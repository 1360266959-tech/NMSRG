# extract_features_like_sample.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import savemat

from dalib.adaptation.cdan import ImageClassifier
import utils  # 你的工程里的 utils.py


def build_loaders(args):
    train_tf = utils.get_train_transform(
        args.train_resizing, resize_size=args.resize_size,
        norm_mean=args.norm_mean, norm_std=args.norm_std
    )
    val_tf = utils.get_val_transform(
        args.val_resizing, resize_size=args.resize_size,
        norm_mean=args.norm_mean, norm_std=args.norm_std
    )
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, _ = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_tf, val_tf)

    # 源域：用 train_source（有标签）
    src_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, drop_last=False)
    # 目标域：为获得真标签，默认用 test_dataset（Office-Home 有 GT）
    tgt_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, drop_last=False)
    return src_loader, tgt_loader, num_classes


@torch.no_grad()
def extract_features(loader, feature_extractor, device):
    """
    输出与样例一致的结构与类型：
      features: (N, D, 1, 1) float32
      labels:   (1, N) int64
    """
    feature_extractor.eval()
    feats, lbs = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None

        x = x.to(device, non_blocking=True)
        f = feature_extractor(x)
        if isinstance(f, (list, tuple)):
            f = f[0]

        # 变成 (B, D, 1, 1)
        if f.dim() == 2:
            f = f.unsqueeze(-1).unsqueeze(-1)
        elif f.dim() == 4 and (f.shape[-1] != 1 or f.shape[-2] != 1):
            f = torch.nn.functional.adaptive_avg_pool2d(f, output_size=(1, 1))
        elif f.dim() not in (2, 4):
            f = f.flatten(1).unsqueeze(-1).unsqueeze(-1)

        feats.append(f.detach().cpu())
        if y is not None:
            lbs.append(y.detach().cpu())
        else:
            lbs.append(torch.zeros(f.size(0), dtype=torch.long))

    feats = torch.cat(feats, dim=0).to(torch.float32).numpy()      # (N, D, 1, 1)
    labels_col = torch.cat(lbs, dim=0).to(torch.int64).numpy()     # (N,)
    labels_row = labels_col.reshape(1, -1)                         # (1, N)
    return feats, labels_row


def main():
    parser = argparse.ArgumentParser(
        description="Extract features to MAT like sample (features:(N,D,1,1) float32, labels:(1,N) int64).")
    parser.add_argument('--root', required=True, help='E:\BaiduNetdiskDownload\MIC特征提取\MIC-master\cls\data\image_CLEF')
    parser.add_argument('--data', default='image_CLEF', choices=utils.get_dataset_names())
    parser.add_argument('--source', nargs='+', required=True, help='c')
    parser.add_argument('--target', nargs='+', required=True, help='i')
    parser.add_argument('--arch', default='resnet50', choices=utils.get_model_names())
    parser.add_argument('--bottleneck-dim', type=int, default=256)
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--train-resizing', default='default')
    parser.add_argument('--val-resizing', default='default')
    parser.add_argument('--norm-mean', nargs='+', type=float, default=(0.485, 0.456, 0.406))
    parser.add_argument('--norm-std', nargs='+', type=float, default=(0.229, 0.224, 0.225))
    parser.add_argument('--no-pool', action='store_true')

    parser.add_argument('--model-path', required=True, help='E:\BaiduNetdiskDownload\imageclef\resnet\jb\ImageCLEF_c2i\checkpoints')
    parser.add_argument('--outdir', required=True, help='E:\BaiduNetdiskDownload\imageclef\特征\jb')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    src_loader, tgt_loader, num_classes = build_loaders(args)

    # 模型
    backbone = utils.get_model(args.arch, pretrain=False)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    classifier.load_state_dict(ckpt)
    classifier.eval()

    # 特征提取器（backbone + pool + bottleneck）
    feature_extractor = nn.Sequential(
        classifier.backbone, classifier.pool_layer, classifier.bottleneck
    ).to(device)
    feature_extractor.eval()

    # 源域
    src_feats, src_labels = extract_features(src_loader, feature_extractor, device)
    mat_src = os.path.join(args.outdir, f"officehome-source-{args.source[0]}-{args.target[0]}-{args.arch}.mat")
    savemat(mat_src, {'features': src_feats, 'labels': src_labels})
    print(f"[✓] Saved {mat_src}")

    # 目标域
    tgt_feats, tgt_labels = extract_features(tgt_loader, feature_extractor, device)
    mat_tgt = os.path.join(args.outdir, f"officehome-{args.source[0]}-{args.target[0]}-{args.arch}.mat")
    savemat(mat_tgt, {'features': tgt_feats, 'labels': tgt_labels})
    print(f"[✓] Saved {mat_tgt}")
    print("[*] Done.")


if __name__ == "__main__":
    main()
