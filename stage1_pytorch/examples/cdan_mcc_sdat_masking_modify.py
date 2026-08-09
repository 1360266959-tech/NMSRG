# Credits: https://github.com/thuml/Transfer-Learning-Library
import os, torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import os, sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))     # ...\cls\examples
CLS_DIR = os.path.abspath(os.path.join(CUR_DIR, '..'))   # ...\cls
ROOT_DIR = os.path.abspath(os.path.join(CLS_DIR, '..'))  # ...\MIC-master
sys.path.insert(0, CLS_DIR)   # 让 from dalib... 指到 cls\dalib
sys.path.insert(0, ROOT_DIR)

import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

sys.path.append('../')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from dalib.modules.masking import Masking
from dalib.modules.teacher import EMATeacher
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM

sys.path.append('.')
import utils


def main(args: argparse.Namespace, eps=0.):
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    cudnn.benchmark = True
    device = args.device

    # Data loading code
    train_transform = utils.get_train_transform(
        args.train_resizing,
        random_horizontal_flip=not args.no_hflip,
        random_color_jitter=False,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    val_transform = utils.get_val_transform(
        args.val_resizing,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    train_source_loader = DataLoader(
        train_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    train_target_loader = DataLoader(
        train_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(
        backbone,
        num_classes,
        bottleneck_dim=args.bottleneck_dim,
        pool_layer=pool_layer,
        finetune=not args.scratch,
    ).to(device)
    classifier_feature_dim = classifier.features_dim

    if args.randomized:
        domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    base_optimizer = torch.optim.SGD
    ad_optimizer = SGD(
        domain_discri.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    optimizer = SAM(
        classifier.get_parameters(),
        base_optimizer,
        rho=args.rho,
        adaptive=False,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    decay_fn = lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay)

    lr_scheduler = LambdaLR(optimizer, decay_fn)
    lr_scheduler_ad = LambdaLR(ad_optimizer, decay_fn)

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri,
        entropy_conditioning=args.entropy,
        num_classes=num_classes,
        features_dim=classifier_feature_dim,
        randomized=args.randomized,
        randomized_dim=args.randomized_dim,
        eps=eps,
    ).to(device)

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    teacher = EMATeacher(classifier, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(device)
    masking = Masking(
        block_size=args.mask_block_size,
        ratio=args.mask_ratio,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
        mean=args.norm_mean,
        std=args.norm_std,
    )

    # resume from the best checkpoint
    if args.phase in ('test', 'analysis'):
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    if args.phase == 'pretrain':
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('best'))
        return

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        print("lr_bbone:", lr_scheduler.get_last_lr()[0])
        print("lr_btlnck:", lr_scheduler.get_last_lr()[1])

        if args.phase == 'train':
            print("SDAT Training")
            train(
                train_source_iter,
                train_target_iter,
                classifier,
                teacher,
                domain_adv,
                mcc_loss,
                masking,
                optimizer,
                ad_optimizer,
                lr_scheduler,
                lr_scheduler_ad,
                epoch,
                args,
                domain_discri,
            )
        elif args.phase == 'sourceonly':
            print("Source Only Training")
            sotrain(train_source_iter, classifier, mcc_loss, masking, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            torch.save(classifier.state_dict(), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    return acc1


def sym_kl(log_p, q):
    # log_p = log_softmax(...), q = softmax(...)
    q = q.clamp_min(1e-8)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    p = log_p.exp().clamp_min(1e-8)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)

    kl_pq = F.kl_div(log_p, q, reduction='batchmean', log_target=False)
    kl_qp = F.kl_div(q.log(), p, reduction='batchmean', log_target=False)
    return kl_pq + kl_qp


def train(
    train_source_iter,
    train_target_iter,
    model: ImageClassifier,
    teacher: EMATeacher,
    domain_adv: ConditionalDomainAdversarialLoss,
    mcc,
    masking,
    optimizer,
    ad_optimizer,
    lr_scheduler: LambdaLR,
    lr_scheduler_ad,
    epoch: int,
    args,
    domain_discri,
):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    domain_adv.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)
        x_s, x_t, labels_s = x_s.to(args.device), x_t.to(args.device), labels_s.to(args.device)

        # ===== 互补遮罩（只采样一次）=====
        with torch.no_grad():
            mask1, mask2 = masking.generate_binary_mask(x_t)

        # 强制与 x_t 一致
        mask1 = mask1.to(dtype=x_t.dtype)
        mask2 = mask2.to(dtype=x_t.dtype)

        x_t_m1 = x_t * mask1
        x_t_m2 = x_t * mask2

        x_t_m1 = masking(x_t_m1, aug=args.ms)
        x_t_m2 = masking(x_t_m2, aug=args.ms)
        # 计时
        data_time.update(time.time() - end)

        # ===== Teacher 仅用于未遮挡图的伪标签 =====
        global_iter = epoch * args.iters_per_epoch + i
        teacher.update_weights(model, global_iter)

        with torch.no_grad():
            logits_t_unmasked, _ = teacher.ema_model(x_t)
            prob_t_unmasked = F.softmax(logits_t_unmasked, dim=1)
            conf, pseudo_hard = prob_t_unmasked.max(dim=1)
            confident = conf.ge(args.tau)  # 置信阈值

        # ===== Student 对两路 masked 图的输出（互补一致性）=====
        s_m1, _ = model(x_t_m1)
        s_m2, _ = model(x_t_m2)
        p_m1 = F.softmax(s_m1, dim=1)
        p_m2 = F.softmax(s_m2, dim=1)

        # L_cm：student–student 的对称一致性（论文原意）
        L_cm = 0.5 * (F.mse_loss(p_m1, p_m2) + F.mse_loss(p_m2, p_m1))

        # L_cl：只对高置信未遮挡伪标签做 CE（均衡两路 masked）
        L_cl = torch.tensor(0.0, device=x_t.device)
        if confident.any() and (global_iter >= args.ema_warmup_iters):
            idx = confident.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                L_cl = 0.5 * (
                    F.cross_entropy(s_m1[idx], pseudo_hard[idx]) +
                    F.cross_entropy(s_m2[idx], pseudo_hard[idx])
                )

        # ===== 主干：Source CE + (CDAN + MCC) =====
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, 0)
        f_s, f_t = f.chunk(2, 0)

        cls_loss_s = F.cross_entropy(y_s, labels_s)
        transfer = domain_adv(y_s, f_s, y_t, f_t) + mcc(y_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        cls_acc = accuracy(y_s, labels_s)[0]
        # ===== 分类器阶段：先冻结判别器 =====
        for p in domain_discri.parameters():
            p.requires_grad_(False)
        # 总损失（用于 SAM 第一步）
        loss_all = cls_loss_s + args.trade_off * transfer + args.lambda_cm * L_cm + args.lambda_cl * L_cl
        if not torch.isfinite(loss_all):
            print("[NaN] loss_all exploded",
                float(cls_loss_s), float(transfer), float(L_cm), float(L_cl))
            optimizer.zero_grad(); ad_optimizer.zero_grad()
            continue
        # ===== SAM 第一步（只更分类器）=====
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.first_step(zero_grad=True)

        # ===== SAM 第二步：重算前向（注意互补项也要重算）=====
        y, f = model(x)
        y_s, y_t = y.chunk(2, 0)
        f_s, f_t = f.chunk(2, 0)
        cls_loss_s_2 = F.cross_entropy(y_s, labels_s)
        transfer_2 = domain_adv(y_s, f_s, y_t, f_t) + mcc(y_t)

        s_m1_2, _ = model(x_t_m1)
        s_m2_2, _ = model(x_t_m2)
        p_m1_2 = F.softmax(s_m1_2, dim=1)
        p_m2_2 = F.softmax(s_m2_2, dim=1)
        L_cm_2 = 0.5 * (F.mse_loss(p_m1_2, p_m2_2) + F.mse_loss(p_m2_2, p_m1_2))

        L_cl_2 = torch.tensor(0.0, device=x_t.device)
        if confident.any() and (global_iter >= args.ema_warmup_iters):
            idx = confident.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                L_cl_2 = 0.5 * (
                    F.cross_entropy(s_m1_2[idx], pseudo_hard[idx]) +
                    F.cross_entropy(s_m2_2[idx], pseudo_hard[idx])
                )

        loss_all_2 = cls_loss_s_2 + args.trade_off * transfer_2 + args.lambda_cm * L_cm_2 + args.lambda_cl * L_cl_2

        optimizer.zero_grad()
        loss_all_2.backward()
        optimizer.second_step(zero_grad=True)

        # ===== 判别器单独更新（用 detach，避免回流到分类器）=====
        for p in domain_discri.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            y_s_det, y_t_det = y_s.detach(), y_t.detach()
            f_s_det, f_t_det = f_s.detach(), f_t.detach()
        d_loss = domain_adv(y_s_det, f_s_det, y_t_det, f_t_det)

        ad_optimizer.zero_grad()
        (d_loss * args.trade_off).backward()
        ad_optimizer.step()

        # scheduler
        lr_scheduler.step()
        lr_scheduler_ad.step()

        # 统计
        losses.update(loss_all.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        domain_accs.update(domain_acc, x_s.size(0))
        trans_losses.update(transfer.item(), x_s.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)




def sotrain(
    train_source_iter: ForeverDataIterator,
    model: ImageClassifier,
    mcc,
    masking,
    optimizer,
    lr_scheduler: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch, [batch_time, data_time, losses, trans_losses, cls_accs], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(args.device)
        labels_s = labels_s.to(args.device)

        data_time.update(time.time() - end)

        optimizer.zero_grad()
        y_s, f_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        loss.backward()
        optimizer.first_step(zero_grad=True)

        y_s, f_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))

        loss.backward()
        optimizer.second_step(zero_grad=True)

        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation'
    )
    # dataset parameters
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument(
        '-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
        help='dataset: ' + ' | '.join(utils.get_dataset_names()) + ' (default: Office31)'
    )
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224, help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225), help='normalization std')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--eps', default=1.0, type=float, help='hyper-parameter for environemnt label smoothing.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true', help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=True, action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float, help='the trade-off hyper-parameter for transfer loss')

    # training parameters
    parser.add_argument('--tau', type=float, default=0.88, help='confidence threshold for pseudo labels')
    parser.add_argument('--ema_warmup_iters', type=int, default=1500, help='do not apply CE-to-pseudo until warmup')

    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, metavar='W', dest='weight_decay',
                        help='weight decay (default: 1e-3)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true', help='whether output per-class accuracy during evaluation')
    parser.add_argument('--log', type=str, default='cdan', help='Where to save logs, checkpoints and debugging images.')
    parser.add_argument('--phase', type=str, default='train', choices=['pretrain','sourceonly','train','test','analysis'],
                        help="When phase is 'test', only test the model. When phase is 'analysis', only analysis the model.")
    parser.add_argument('--log_results', action='store_true', help='To log results in wandb')
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID')
    parser.add_argument('--log_name', type=str, default='log', help='log name for wandb')
    parser.add_argument('--rho', type=float, default=0.02, help='GPU ID')
    parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling')

    # weights for the new losses
    parser.add_argument('--lambda_cm', type=float, default=0.01, help='weight for complementary-mask consistency (L_cm)')
    parser.add_argument('--lambda_cl', type=float, default=1.0, help='weight for CE-to-pseudolabels on masked views (L_cl)')

    # masked image consistency
    parser.add_argument('--alpha', default=0.9, type=float)
    parser.add_argument('--pseudo_label_weight', default='prob')
    parser.add_argument('--mask_block_size', default=64, type=int)
    parser.add_argument('--mask_ratio', default=0.7, type=float)
    parser.add_argument('--mask_color_jitter_s', default=0.2, type=float)
    parser.add_argument('--mask_color_jitter_p', default=0.2, type=float)
    parser.add_argument('--mask_blur', default=True, type=bool)
    parser.add_argument('--ms', type=str, default="jb",help="masking augmentation string. Use letters in {'j','b','e','g'}, e.g., jb/je/jbg/jbeg")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    main(args, eps=args.eps)
