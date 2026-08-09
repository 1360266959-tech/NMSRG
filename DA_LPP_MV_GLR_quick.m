function [acc, macro_recall] = DA_LPP_MV_GLR_quick(pre_feat_S,pre_lb_S,pre_feat_T,pre_lb_T,so_feat_S,so_lb_S,so_feat_T,so_lb_T,pseudo_feat_S,pseudo_lb_S,pseudo_feat_T,pseudo_lb_T,q,lambda,beta,metric_count)
% Evaluate one specified Stage-2 parameter tuple.
% metric_count controls how many leading samples in the second input set are
% evaluated. In source-validation selection, these are held-out source
% samples; appended target samples have NaN labels and are never evaluated.

if nargin < 15
    q = 2/3;
    lambda = 1e-3;
    beta = 1e-4;
end
if nargin < 16
    metric_count = numel(pre_lb_T);
end
if metric_count > numel(pre_lb_T)
    error('metric_count cannot exceed the number of second-set samples.');
end

options.NeighborMode='KNN';
options.WeightMode = 'HeatKernel';
options.k = 30;
options.t = 1;
options.ReducedDim = 128; % 如果报错，记得改小
options.alpha = 1;

num_class = length(unique(pre_lb_S));

pretrain_features = [pre_feat_S;pre_feat_T];
so_features = [so_feat_S;so_feat_T];
pseudo_features = [pseudo_feat_S;pseudo_feat_T];

% ==================== 内存救星：统一采样模块 ====================
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

% NLRSC 算法要求输入格式为 [特征维度 x 样本数]，所以在这里转置！
X{1} = X_pre';
X{2} = X_so';
X{3} = X_pse';
% ================================================================

% 1. NLRSC 特有的 Weight 初始化
nV = length(X);
Weight = cell(1,nV);
for v=1:nV
    [nFea,nSmp] = size(X{v});
    for i = 1:nSmp
        X{v}(:,i) = X{v}(:,i) ./ max(1e-12,norm(X{v}(:,i)));
    end
    Lv = X{v}'*X{v};
    sigmav = mean(mean(1 - Lv));
    if sigmav < 1e-6
        sigmav = 1e-6;
    end
    Weight{v} = ones(size(Lv)) - exp(-(ones(size(Lv))-Lv)/sigmav);
end

% 2. 运行 NLRSC 优化算法
mu = 1e-3;
epsilon = 1e-4;
rho = 1.5;
[Coe,Conver_iter,iter] = NLRSC_MSC_Lq_matrix_Cor(X,Weight,mu,rho,epsilon,lambda,beta,q);

% 3. 构建亲和力图 W
% Signed affinity matrix
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

% 4. 投影
P = LPP(X_pre, W, options,'signed'); 
proj_S_pre = pre_feat_S * P;
proj_T_pre = pre_feat_T * P;

P = LPP(X_so, W, options, 'signed');
proj_S_source = so_feat_S * P;
proj_T_source = so_feat_T * P;

P = LPP(X_pse, W, options ,'signed');
proj_S_pseudo = pseudo_feat_S * P;
proj_T_pseudo = pseudo_feat_T * P;

% 5. 计算各视图的距离并做三路融合
[dist1,dist4] = distmeans(proj_S_pre,proj_T_pre,pre_lb_S,num_class);
[dist2,dist5] = distmeans(proj_S_source,proj_T_source,so_lb_S,num_class);
[dist3,dist6] = distmeans(proj_S_pseudo,proj_T_pseudo,pseudo_lb_S, num_class);

distClassMeans = (dist2+dist1+dist3)/3;
distClusterMeans = (dist5+dist4+dist6)/3;

expMatrix = exp(-distClassMeans);
expMatrix2 = exp(-distClusterMeans);
probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
probMatrix = max(probMatrix,probMatrix2);
[prob,predLabels] = max(probMatrix');

% 6. Restrict metric computation to the designated evaluation subset.
eval_labels = reshape(pre_lb_T(1:metric_count), 1, []);
eval_predictions = reshape(predLabels(1:metric_count), 1, []);

if numel(eval_predictions) ~= numel(eval_labels)
    error('Prediction and evaluation-label counts do not match.');
end

acc = mean(eval_predictions == eval_labels);
fprintf('\n>>> Overall Accuracy: %0.4f <<<\n', acc);

% ==========================================================
% Comprehensive evaluation (Accuracy, Precision, Recall, F1)
% ==========================================================

precision_per_class = zeros(1, num_class);
recall_per_class = zeros(1, num_class);
f1_per_class = zeros(1, num_class);
acc_per_class = zeros(1, num_class);

fprintf('\n--- Per-class evaluation ---\n');
fprintf('Class | Precision | Recall/Acc | F1-Score \n');
fprintf('-----------------------------------------\n');

for i = 1:num_class
    TP = sum((eval_predictions == i) & (eval_labels == i));
    FP = sum((eval_predictions == i) & (eval_labels ~= i));
    FN = sum((eval_predictions ~= i) & (eval_labels == i));

    if (TP + FP) == 0
        precision_per_class(i) = 0;
    else
        precision_per_class(i) = TP / (TP + FP);
    end

    if (TP + FN) == 0
        recall_per_class(i) = 0;
    else
        recall_per_class(i) = TP / (TP + FN);
    end
    acc_per_class(i) = recall_per_class(i);

    if (precision_per_class(i) + recall_per_class(i)) == 0
        f1_per_class(i) = 0;
    else
        f1_per_class(i) = 2 * (precision_per_class(i) * recall_per_class(i)) / (precision_per_class(i) + recall_per_class(i));
    end

    fprintf(' %2d   |  %0.4f   |  %0.4f    |  %0.4f  \n', ...
        i, precision_per_class(i), recall_per_class(i), f1_per_class(i));
end

% 2. 计算宏平均 (Macro-Average)
macro_precision = mean(precision_per_class);
macro_recall = mean(recall_per_class);
macro_f1 = mean(f1_per_class);

fprintf('=========================================\n');
fprintf('Macro Precision : %0.4f\n', macro_precision);
fprintf('Macro Recall    : %0.4f\n', macro_recall);
fprintf('Macro F1-Score  : %0.4f\n', macro_f1);
fprintf('=========================================\n');