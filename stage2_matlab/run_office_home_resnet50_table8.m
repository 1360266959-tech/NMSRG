
clear all
addpath('./utils/');
addpath('./misc/');
addpath('./function/');

diary('output_log.txt');

total=tic;

noft_dir = 'OfficeHome/resnet50/jb/';
soly_dir = 'OfficeHome/resnet50/je/';
pseudo_dir = 'OfficeHome/resnet50/jg/';

domains = {'Ar','Cl','Pr','Rw'};

for source_domain_index = 1:length(domains)
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([noft_dir 'officehome-source-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50.mat']);
        pre_feat_S = L2Norm(features);
        pre_lb_S = labels+1;
        load([noft_dir 'officehome-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50.mat']);
        pre_feat_T = L2Norm(features);
        pre_lb_T = labels+1;
        load([soly_dir 'officehome-source-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50.mat']);
        so_feat_S = L2Norm(features);
        so_lb_S = labels+1;
        load([soly_dir 'officehome-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50.mat']);
        so_feat_T = L2Norm(features);
        so_lb_T = labels+1;
        load([pseudo_dir 'officehome-source-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50.mat']);
        pseudo_feat_S = L2Norm(features);
        pseudo_lb_S = labels+1;
        load([pseudo_dir 'officehome-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50.mat']);
        pseudo_feat_T = L2Norm(features);
        pseudo_lb_T = labels+1;
        opts.ReducedDim = 128;
        X = double([pre_feat_S;pre_feat_T;so_feat_S;so_feat_T;pseudo_feat_S;pseudo_feat_T]);
        P_pca = PCA(X,opts);
        pre_feat_S = pre_feat_S*P_pca;
        pre_feat_T = pre_feat_T*P_pca;
        pre_feat_S = L2Norm(pre_feat_S);
        pre_feat_T = L2Norm(pre_feat_T);
        so_feat_S = so_feat_S*P_pca;
        so_feat_T = so_feat_T*P_pca;
        so_feat_S = L2Norm(so_feat_S);
        so_feat_T = L2Norm(so_feat_T);
        pseudo_feat_S = pseudo_feat_S*P_pca;
        pseudo_feat_T = pseudo_feat_T*P_pca;
        pseudo_feat_S = L2Norm(pseudo_feat_S);
        pseudo_feat_T = L2Norm(pseudo_feat_T);
        num_class = length(unique(pre_lb_T));
        %% Baseline method: using 1-NN, only labelled source data for training
        fprintf('Baseline method using 1NN:\n');
        classifierType='1nn';
        fprintf('Proposed method using 1NN:\n');
        acc_per_class = Run_GraphReplacement_Table8(pre_feat_S,pre_lb_S,pre_feat_T,pre_lb_T,so_feat_S,so_lb_S, so_feat_T,so_lb_T, pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T);
    end
end

endTime=toc(total);
fprintf('Elapsed time for this iteration: %.2f seconds\n',endTime);

diary off;