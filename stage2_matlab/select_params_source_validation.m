function [best_q, best_lambda, best_beta, best_mca, result] = ...
    select_params_source_validation(pre_feat_S, pre_lb_S, ...
    so_feat_S, so_lb_S, pseudo_feat_S, pseudo_lb_S)
% Select NMSRG hyperparameters without target labels.
% Candidate tuples are ranked only by macro recall on a held-out SOURCE
% validation set. Unlabeled target features remain available, as in UDA.

    if ~isequal(pre_lb_S(:), so_lb_S(:)) || ~isequal(pre_lb_S(:), pseudo_lb_S(:))
        error('Source labels must have identical ordering in all three views.');
    end

    rng(2024, 'twister');
    q_grid = [1/2, 2/3];
    lambda_grid = [1e-4, 1e-3, 1e-2];
    beta_grid = [1e-5, 1e-4, 1e-3];
    [source_train_idx, source_val_idx] = stratified_source_split(pre_lb_S, 0.8);
    scores = zeros(numel(q_grid), numel(lambda_grid), numel(beta_grid));

    fprintf('\n========== Source-validation parameter selection ==========\n');
    fprintf('Stratified source split: 80/20; seed: 2024\n');

    for iq = 1:numel(q_grid)
        for il = 1:numel(lambda_grid)
            for ib = 1:numel(beta_grid)
                % Reset randomness so all candidate tuples use identical
                % source split, graph sampling, and clustering initialization.
                rng(2024, 'twister');
        [~, scores(iq, il, ib)] = DA_LPP_MV_GLR_quick( ...
            pre_feat_S(source_train_idx, :), pre_lb_S(source_train_idx), ...
            pre_feat_S(source_val_idx, :), pre_lb_S(source_val_idx), ...
            so_feat_S(source_train_idx, :), so_lb_S(source_train_idx), ...
            so_feat_S(source_val_idx, :), so_lb_S(source_val_idx), ...
            pseudo_feat_S(source_train_idx, :), pseudo_lb_S(source_train_idx), ...
            pseudo_feat_S(source_val_idx, :), pseudo_lb_S(source_val_idx), ...
            q_grid(iq), lambda_grid(il), beta_grid(ib));
                fprintf('q=%g, lambda=%g, beta=%g, source-val MCA=%.4f\n', ...
                    q_grid(iq), lambda_grid(il), beta_grid(ib), scores(iq, il, ib));
            end
        end
    end

    [best_mca, best_linear_idx] = max(scores(:));
    [best_iq, best_il, best_ib] = ind2sub(size(scores), best_linear_idx);
    best_q = q_grid(best_iq);
    best_lambda = lambda_grid(best_il);
    best_beta = beta_grid(best_ib);

    result = struct('q_grid', q_grid, 'lambda_grid', lambda_grid, ...
        'beta_grid', beta_grid, 'source_val_mca', scores, ...
        'source_train_idx', source_train_idx, 'source_val_idx', source_val_idx, ...
        'best_q', best_q, 'best_lambda', best_lambda, 'best_beta', best_beta, ...
        'best_source_val_mca', best_mca);

    fprintf('Selected tuple: q=%g, lambda=%g, beta=%g, source-val MCA=%.4f\n', ...
        best_q, best_lambda, best_beta, best_mca);
end

function [train_idx, val_idx] = stratified_source_split(labels, train_fraction)
    labels = labels(:);
    train_idx = false(numel(labels), 1);
    val_idx = false(numel(labels), 1);
    classes = unique(labels)';

    for class_id = classes
        class_idx = find(labels == class_id);
        class_idx = class_idx(randperm(numel(class_idx)));
        n_train = max(floor(train_fraction * numel(class_idx)), 1);
        if n_train >= numel(class_idx)
            error('Class %d has fewer than two source samples.', class_id);
        end
        train_idx(class_idx(1:n_train)) = true;
        val_idx(class_idx(n_train + 1:end)) = true;
    end
end

