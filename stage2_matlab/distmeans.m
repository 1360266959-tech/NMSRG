function [distClassMeans,distClusterMeans] = distmeans(domainS_proj,domainT_proj,domainS_labels,num_class)

    options.ReducedDim = size(domainS_proj,2);

    % ===== 1. 强制数值安全 =====
    domainS_proj = real(full(double(domainS_proj)));
    domainT_proj = real(full(double(domainT_proj)));

    domainS_proj(~isfinite(domainS_proj)) = 0;
    domainT_proj(~isfinite(domainT_proj)) = 0;

    % ===== 2. 中心化 =====
    proj_mean = mean([domainS_proj; domainT_proj], 1);
    domainS_proj = domainS_proj - proj_mean;
    domainT_proj = domainT_proj - proj_mean;

    % ===== 3. L2 归一化 =====
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);

    % ===== 4. 类中心 =====
    classMeans = zeros(num_class, options.ReducedDim);
    for i = 1:num_class
        idx = (domainS_labels == i);
        if any(idx)
            classMeans(i,:) = mean(domainS_proj(idx,:), 1);
        end
    end
    classMeans = L2Norm(classMeans);

    % ===== 5. 距离 =====
    distClassMeans = EuDist2(domainT_proj, classMeans);

    % ===== 6. k-means（关键）=====
    targetClusterMeans = vgg_kmeans(domainT_proj', num_class, classMeans')';
    targetClusterMeans = real(targetClusterMeans);
    targetClusterMeans = L2Norm(targetClusterMeans);

    distClusterMeans = EuDist2(domainT_proj, targetClusterMeans);
end
