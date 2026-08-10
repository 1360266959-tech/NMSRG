

addpath('./utils/');
addpath('./misc/');
addpath('./function/');

diary('output_log.txt'); 



total=tic;

noft_dir = 'Officehome/resnet50/jb/';
soly_dir = 'Officehome/resnet50/je/';
pseudo_dir = 'Officehome/resnet50/jg/';

domains = {'Ar','Pr','Cl','Rw'};

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
        % Hyperparameter selection is performed before target evaluation.
        % It uses only a stratified 80/20 split of the labeled source data.
        % Target labels are not passed to select_params_source_validation.
        [best_q, best_lambda, best_beta, best_source_val_mca, search_result] = ...
            select_params_source_validation(pre_feat_S, pre_lb_S, ...
            so_feat_S, so_lb_S, pseudo_feat_S, pseudo_lb_S);
                            %best_q = 0.5;
                %best_lambda = 1e-4;
                %best_beta = 1e-5;

        fprintf('\nSelected by source validation for %s -> %s:\n', ...
            domains{source_domain_index}, domains{target_domain_index});
        fprintf('q = %g, lambda = %g, beta = %g, source-val MCA = %.4f\n', ...
            best_q, best_lambda, best_beta, best_source_val_mca);

        % Store validation-only search results for sensitivity plots.
        result_file = sprintf('source_validation_%s_to_%s.mat', ...
            domains{source_domain_index}, domains{target_domain_index});
        %save(result_file, 'best_q', 'best_lambda', 'best_beta', ...
            %'best_source_val_mca', 'search_result');
            save(result_file, 'best_q', 'best_lambda', 'best_beta');

        % Final evaluation: re-use all labeled source samples and the
        % unlabeled target features. Target labels enter only here, after
        % source-validation selection is complete.
        rng(2024, 'twister');
        [target_acc, target_mca] = DA_LPP_MV_GLR_quick( ...
            pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ...
            so_feat_S, so_lb_S, so_feat_T, so_lb_T, ...
            pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T, ...
            best_q, best_lambda, best_beta);
        fprintf('Final target accuracy = %.4f, final target MCA = %.4f\n', ...
            target_acc, target_mca);
    end
end

endTime=toc(total);
fprintf('Elapsed time for this iteration: %.2f seconds\n',endTime);

diary off;