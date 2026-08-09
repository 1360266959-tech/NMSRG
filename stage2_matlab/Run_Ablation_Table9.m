function [acc_per_class] = Run_Ablation_Table9(pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, so_feat_S, so_lb_S, so_feat_T, so_lb_T, pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T)
    % 设定的3个随机种子
    seeds = [2024, 2025, 2026];
    num_seeds = length(seeds);
    
    % 初始化存储 Macro Recall (Mean Class Accuracy) 的数组
    res_baseline   = zeros(1, num_seeds);
    res_interview  = zeros(1, num_seeds);
    res_interclass = zeros(1, num_seeds);
    res_full       = zeros(1, num_seeds);
    acc_ord = zeros(1, num_seeds);
recall_ord = zeros(1, num_seeds);
acc_sgn = zeros(1, num_seeds);
recall_sgn = zeros(1, num_seeds);
    
    fprintf('\n========== 开启自动化消融实验 (3个随机种子) ==========\n');
    
    for i = 1:num_seeds
        current_seed = seeds(i);
        fprintf('\n>>> 正在运行 Seed: %d <<<\n', current_seed);
        
        % 1. 运行 Baseline
        [~, res_baseline(i)] = baseline_eval(pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T);
        
        % 统一设置随机种子
        rng(current_seed);
        
         2. 运行 Graph + Inter-view consistency (lambda = 0, beta = 1e-4)
        [~, res_interview(i)]  = graph_variant_eval(pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ...
                                    so_feat_S, so_lb_S, so_feat_T, so_lb_T, pseudo_feat_S, pseudo_lb_S, ...
                                    pseudo_feat_T, pseudo_lb_T, 0, 1e-4, 'signed');
                                    
         3. 运行 Graph + Inter-class separability (lambda = 1e-3, beta = 0)
        [~, res_interclass(i)] = graph_variant_eval(pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ...
                                    so_feat_S, so_lb_S, so_feat_T, so_lb_T, pseudo_feat_S, pseudo_lb_S, ...
                                    pseudo_feat_T, pseudo_lb_T, 1e-3, 0, 'signed');
                                    
        % 4. 运行 Full NMSRG (lambda = 1e-3, beta = 1e-4)
rng(current_seed);
[acc_ord(i), recall_ord(i)] = graph_variant_eval( ...
    pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ...
    so_feat_S, so_lb_S, so_feat_T, so_lb_T, ...
    pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T, ...
    1e-3, 1e-4, 'ordinary');

rng(current_seed);
[acc_sgn(i), recall_sgn(i)] = graph_variant_eval( ...
    pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ...
    so_feat_S, so_lb_S, so_feat_T, so_lb_T, ...
    pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T, ...
    1e-3, 1e-4, 'signed');
    end
    
    % ========== 计算均值和标准差并生成 LaTeX 格式 ==========
    fprintf('\n\n========== 最终结果 (直接复制到 LaTeX Table 9) ==========\n');
    fprintf('Stage-1 Only (Baseline)      : %0.2f \\pm %0.2f\n', mean(res_baseline)*100,   std(res_baseline)*100);
    fprintf('Graph+Inter-view consistency : %0.2f \\pm %0.2f\n', mean(res_interview)*100,  std(res_interview)*100);
    fprintf('Graph+Inter-class separability: %0.2f \\pm %0.2f\n', mean(res_interclass)*100, std(res_interclass)*100);
    fprintf('Full NMSRG (Ours)            : %0.2f \\pm %0.2f\n', mean(recall_sgn)*100,       std(recall_sgn)*100);
    fprintf('=======================================================\n');
        fprintf('\n========== Ordinary vs Signed LPP ==========\n');
fprintf('Ordinary LPP, Overall Accuracy: %.2f \\pm %.2f\n', ...
    mean(acc_ord)*100, std(acc_ord)*100);
fprintf('Ordinary LPP, Macro Recall   : %.2f \\pm %.2f\n', ...
    mean(recall_ord)*100, std(recall_ord)*100);

fprintf('Signed LPP, Overall Accuracy : %.2f \\pm %.2f\n', ...
    mean(acc_sgn)*100, std(acc_sgn)*100);
fprintf('Signed LPP, Macro Recall     : %.2f \\pm %.2f\n', ...
    mean(recall_sgn)*100, std(recall_sgn)*100);
fprintf('============================================\n');
    % 【关键修复】将 Full NMSRG 的均值作为返回值，以满足你在主脚本里的调用
    acc_per_class = mean(recall_sgn);
end

% ==================== 辅助函数 1：Baseline ====================
function [acc, macro_recall] = baseline_eval(pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T)
    num_class = length(unique(pre_lb_S));
    [dist1, dist4] = distmeans(pre_feat_S, pre_feat_T, pre_lb_S, num_class);
    
    expMatrix = exp(-dist1);
    expMatrix2 = exp(-dist4);
    probMatrix = expMatrix ./ repmat(sum(expMatrix, 2), [1 num_class]);
    probMatrix2 = expMatrix2 ./ repmat(sum(expMatrix2, 2), [1 num_class]);
    
    probMatrix = max(probMatrix, probMatrix2);
    [~, predLabels] = max(probMatrix');
    
    acc = sum(predLabels == pre_lb_T(:)') / length(pre_lb_T);
    
    recall_per_class = zeros(1, num_class);
    for i = 1:num_class
        TP = sum((predLabels == i) & (pre_lb_T(:)' == i));
        FN = sum((predLabels ~= i) & (pre_lb_T(:)' == i));
        if (TP + FN) > 0
            recall_per_class(i) = TP / (TP + FN);
        end
    end
    macro_recall = mean(recall_per_class);
end

% ==================== 辅助函数 2：参数化图学习评估 ====================
function [acc, macro_recall] = graph_variant_eval(pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, so_feat_S, so_lb_S, so_feat_T, so_lb_T, pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T, param_lambda, param_beta,laplacian_type)
    options.NeighborMode = 'KNN';
    options.WeightMode = 'HeatKernel';
    options.k = 30;
    options.t = 1;
    options.ReducedDim = 128; 
    options.alpha = 1;
    num_class = length(unique(pre_lb_S));
    
    pretrain_features = [pre_feat_S; pre_feat_T];
    so_features = [so_feat_S; so_feat_T];
    pseudo_features = [pseudo_feat_S; pseudo_feat_T];
    
    max_samples = 1500; 
    total_samples = size(pretrain_features, 1);
    if total_samples > max_samples
        idx = randperm(total_samples, max_samples);
        X_pre = pretrain_features(idx, :);
        X_so  = so_features(idx, :);
        X_pse = pseudo_features(idx, :);
    else
        X_pre = pretrain_features;
        X_so  = so_features;
        X_pse = pseudo_features;
    end
    
    X{1} = X_pre'; X{2} = X_so'; X{3} = X_pse';
    
    nV = length(X);
    Weight = cell(1, nV);
    for v = 1:nV
        for i = 1:size(X{v}, 2)
            X{v}(:,i) = X{v}(:,i) ./ max(1e-12, norm(X{v}(:,i)));
        end
        Lv = X{v}' * X{v};
        sigmav = max(1e-6, mean(mean(1 - Lv)));
        Weight{v} = ones(size(Lv)) - exp(-(ones(size(Lv)) - Lv) / sigmav);
    end
    
    mu = 1e-3; epsilon = 1e-4; rho = 1.5; q = 2/3;
    [Coe, ~, ~] = NLRSC_MSC_Lq_matrix_Cor(X, Weight, mu, rho, epsilon, param_lambda, param_beta, q);
    
    G = Coe + Coe';
    G(1:size(G,1)+1:end) = 0;  % remove self-loops
    
    scale = max(abs(G(:)));
    if scale < 1e-12
        error('The learned signed graph is identically zero.');
    end
    
    W = G / scale;

    % ===== Signed-graph diagnostics =====
    n = size(W, 1);
    offdiag = ~eye(n);
    edge_mask = offdiag & (abs(W) > 1e-12);
    neg_mask = offdiag & (W < -1e-12);
    
    num_edges = nnz(edge_mask);
    num_neg_edges = nnz(neg_mask);
    
    neg_ratio = num_neg_edges / max(num_edges, 1);
    neg_values = W(neg_mask);
    
    if isempty(neg_values)
        neg_mean = 0;
        neg_mean_abs = 0;
        neg_max_abs = 0;
    else
        neg_mean = mean(neg_values);
        neg_mean_abs = mean(abs(neg_values));
        neg_max_abs = max(abs(neg_values));
    end
    
    Dabs = diag(sum(abs(W), 2));
    Lsgn = Dabs - W;
    Lsgn = (Lsgn + Lsgn') / 2;
    
    eigvals_L = eig(full(Lsgn));
    lambda_min_L = min(eigvals_L);
    num_negative_eigs = sum(eigvals_L < -1e-8);
    
    fprintf('\n========== Signed Graph Diagnostics ==========\n');
    fprintf('Nonzero off-diagonal edges : %d\n', num_edges);
    fprintf('Negative edges             : %d\n', num_neg_edges);
    fprintf('Negative-edge ratio        : %.6f\n', neg_ratio);
    fprintf('Mean negative weight       : %.6e\n', neg_mean);
    fprintf('Mean |negative weight|     : %.6e\n', neg_mean_abs);
    fprintf('Max |negative weight|      : %.6e\n', neg_max_abs);
    fprintf('lambda_min(L_sgn)          : %.6e\n', lambda_min_L);
    fprintf('Negative eigenvalues < -1e-8: %d\n', num_negative_eigs);
    fprintf('=============================================\n');
    
    P = LPP(X_pre, W, options,laplacian_type); proj_S_pre = pre_feat_S * P; proj_T_pre = pre_feat_T * P;
    P = LPP(X_so, W, options,laplacian_type); proj_S_source = so_feat_S * P; proj_T_source = so_feat_T * P;
    P = LPP(X_pse, W, options,laplacian_type); proj_S_pseudo = pseudo_feat_S * P; proj_T_pseudo = pseudo_feat_T * P;
    
    [dist1, dist4] = distmeans(proj_S_pre, proj_T_pre, pre_lb_S, num_class);
    [dist2, dist5] = distmeans(proj_S_source, proj_T_source, so_lb_S, num_class);
    [dist3, dist6] = distmeans(proj_S_pseudo, proj_T_pseudo, pseudo_lb_S, num_class);
    
    distClassMeans = (dist2 + dist1 + dist3) / 3;
    distClusterMeans = (dist5 + dist4 + dist6) / 3;
    expMatrix = exp(-distClassMeans); expMatrix2 = exp(-distClusterMeans);
    probMatrix = max(expMatrix ./ repmat(sum(expMatrix, 2), [1 num_class]), ...
                     expMatrix2 ./ repmat(sum(expMatrix2, 2), [1 num_class]));
    [~, predLabels] = max(probMatrix');
    
    acc = sum(predLabels == pre_lb_T(:)') / length(pre_lb_T);
    recall_per_class = zeros(1, num_class);
    for i = 1:num_class
        TP = sum((predLabels == i) & (pre_lb_T(:)' == i));
        FN = sum((predLabels ~= i) & (pre_lb_T(:)' == i));
        if (TP + FN) > 0
            recall_per_class(i) = TP / (TP + FN);
        end
    end
    macro_recall = mean(recall_per_class);

end

