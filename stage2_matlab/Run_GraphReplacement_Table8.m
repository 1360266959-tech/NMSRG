
function results = Run_GraphReplacement_Table8(pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ...
    so_feat_S, so_lb_S, so_feat_T, so_lb_T, ...
    pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T)
% Fair Stage-2 graph-module replacement experiment for Table 8.
% Target labels are used ONLY at final metric computation.

seeds = [2024, 2025, 2026];
M = 4;  % Stage-1 only, MAPA, MLRGL, NMSRG
methodNames = {'Stage-1 only','MAPA-compatible','MLRGL-compatible','NMSRG'};

% ===== Shared protocol =====
cfg.max_source = 750;                 % total graph subset <= 1500
cfg.max_target = 750;
cfg.reduced_dim = 128;
cfg.ridge = 1e-6;

% ===== NMSRG parameters: retain your reported configuration =====
cfg.nmsrg.mu = 1e-3;
cfg.nmsrg.rho = 1.5;
cfg.nmsrg.epsilon = 1e-4;
cfg.nmsrg.lambda = 1e-3;
cfg.nmsrg.beta = 1e-4;
cfg.nmsrg.q = 2/3;

% ===== MAPA parameters stated in the paper =====
cfg.mapa.alpha = 0.5;
cfg.mapa.sigma = 1.0;
cfg.mapa.pca_dim = 64;                % Office-Home; use 128 for ImageCLEF/VisDA
cfg.mapa.iterations = 10;

% ===== MLRGL source-code parameters =====
% Confirm r1/w1/alpha/max_iter against the MLRGL repository main script
% before using the final manuscript values.
cfg.mlrgl.knn_k = 15;
cfg.mlrgl.r1 = 0.02;
cfg.mlrgl.w1 = 0.5;
cfg.mlrgl.alpha = 6;
cfg.mlrgl.max_iter = 1;
cfg.mlrgl.tol = 1e-4;
cfg.mlrgl.mu = 1e-4;
cfg.mlrgl.rho = 1.1;

FS = {pre_feat_S, so_feat_S, pseudo_feat_S};
FT = {pre_feat_T, so_feat_T, pseudo_feat_T};
YS = {pre_lb_S(:), so_lb_S(:), pseudo_lb_S(:)};
YT = pre_lb_T(:);
assert(isequal(YS{1},YS{2}) && isequal(YS{1},YS{3}), ...
    'Three source-label vectors must agree.');
assert(size(FS{1},1)==size(FS{2},1) && size(FS{1},1)==size(FS{3},1), ...
    'All source views must have identical sample order.');
assert(size(FT{1},1)==size(FT{2},1) && size(FT{1},1)==size(FT{3},1), ...
    'All target views must have identical sample order.');

C = numel(unique(YS{1}));
acc = zeros(M,numel(seeds));
mca = zeros(M,numel(seeds));
timeSec = zeros(M,numel(seeds));
subsetSize = zeros(2,numel(seeds));

fprintf('\n========== Fair Graph Replacement: Table 8 ==========' + "\n");
for s = 1:numel(seeds)
    rng(seeds(s),'twister');

    % Seeded uniform sampling is shared by all methods; source samples precede target samples.
    sourceIdx = randperm(size(FS{1},1), min(cfg.max_source,size(FS{1},1)));
    targetIdx = randperm(size(FT{1},1), min(cfg.max_target,size(FT{1},1)));
    subsetSize(:,s) = [numel(sourceIdx);numel(targetIdx)];

    Xsub = cell(1,3);
    for v = 1:3
        Xsub{v} = [FS{v}(sourceIdx,:); FT{v}(targetIdx,:)]; % samples x features
    end
    ySourceSub = YS{1}(sourceIdx);
    nSourceSub = numel(sourceIdx);

    % 1. No graph: common source prototype + target K-means prediction.
    tic;
    [acc(1,s),mca(1,s)] = eval_stage1(FS,YS,FT,YT,C);
    timeSec(1,s) = toc;

    % 2. MAPA-compatible reimplementation. No target ground truth enters.
    rng(seeds(s),'twister');
    tic;
    Wmapa = mapa_graph(Xsub,nSourceSub,ySourceSub,C,cfg);
    timeSec(2,s) = toc;
    [acc(2,s),mca(2,s)] = eval_with_graph(Wmapa,Xsub,FS,YS,FT,YT,C,'ordinary',cfg);

    % 3. MLRGL-compatible adaptation of TMVC. No SpectralClustering or target labels.
    rng(seeds(s),'twister');
    tic;
    Wmlrgl = mlrgl_graph(Xsub,C,cfg.mlrgl);
    timeSec(3,s) = toc;
    [acc(3,s),mca(3,s)] = eval_with_graph(Wmlrgl,Xsub,FS,YS,FT,YT,C,'ordinary',cfg);

    % 4. Original NMSRG graph solver, followed by the same evaluation rule.
    rng(seeds(s),'twister');
    tic;
    Wnmsrg = nmsrg_graph(Xsub,cfg.nmsrg);
    timeSec(4,s) = toc;
    [acc(4,s),mca(4,s)] = eval_with_graph(Wnmsrg,Xsub,FS,YS,FT,YT,C,'signed',cfg);

    fprintf('\nSeed %d, source/target graph subset = %d/%d\n', ...
        seeds(s),nSourceSub,numel(targetIdx));
    for m = 1:M
        fprintf('%-18s Acc %.2f | MCA %.2f | graph time %.2f s\n', ...
            methodNames{m},100*acc(m,s),100*mca(m,s),timeSec(m,s));
    end
end

fprintf('\n========== Copy this into LaTeX ==========\n');
for m = 1:M
    fprintf('%-18s & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \\\\ \n', ...
        methodNames{m},100*mean(acc(m,:)),100*std(acc(m,:)), ...
        100*mean(mca(m,:)),100*std(mca(m,:)), ...
        mean(timeSec(m,:)),std(timeSec(m,:)));
end

results.method = methodNames;
results.seeds = seeds;
results.source_target_subset_size = subsetSize;
results.accuracy = acc;
results.mca = mca;
results.stage2_seconds = timeSec;
results.config = cfg;
results.note = ['MAPA-compatible is a published-formula Stage-2 reimplementation ', ...
    'on frozen NMSRG features. It is not the unavailable original MAPA implementation.'];
save('Table8_GraphReplacement_results.mat','results');
end

function W = nmsrg_graph(Xsub,p)
if exist('NLRSC_MSC_Lq_matrix_Cor','file') ~= 2
    error('Add NLRSC_MSC_Lq_matrix_Cor.m to the MATLAB path first.');
end
V = numel(Xsub); n = size(Xsub{1},1);
X = cell(1,V); Weight = cell(1,V);
for v = 1:V
    X{v} = normalize_rows(Xsub{v})'; % NMSRG expects feature x sample
    K = X{v}'*X{v};
    kappa = max(mean(1-K,'all'),1e-6);
    Weight{v} = 1-exp(-(1-K)/kappa);
end
[Coe,~,~] = NLRSC_MSC_Lq_matrix_Cor(X,Weight,p.mu,p.rho,p.epsilon, ...
    p.lambda,p.beta,p.q);
W = Coe+Coe';
W(1:n+1:end) = 0;
W = W/max(max(abs(W(:))),eps);
end

function W = mapa_graph(Xsub,nS,yS,C,cfg)
% MAPA Eq. (5)--(11): PCA+normalization, source-target affinity, LPP,
% and progressive confidence filtering. Initial pseudo-labels come from
% frozen feature source prototypes because MAPA EMA teacher probabilities
% are unavailable under the common-feature replacement protocol.
V = numel(Xsub); n = size(Xsub{1},1); nT = n-nS;
pseudo = initial_pseudo(Xsub,nS,yS,C);
keep = true(nT,1);
for it = 1:cfg.mapa.iterations
    Wsum = zeros(n,n); Zproj = cell(1,V);
    for v = 1:V
        Z = pca_and_normalize(Xsub{v},cfg.mapa.pca_dim);
        Wv = mapa_affinity(Z,nS,yS,pseudo,keep,cfg.mapa.alpha,cfg.mapa.sigma);
        Wsum = Wsum+Wv;
        [Zproj{v},~] = lpp_basis(Z,Wv,cfg.reduced_dim,'ordinary',cfg.ridge);
    end
    W = Wsum/V;
    [pseudo,confidence] = pseudo_from_views(Zproj,nS,yS,C);
    % Eq. (11)-style classwise progressive inclusion.
    keepFraction = max(0.05,1-it/cfg.mapa.iterations);
    keep = classwise_filter(pseudo,confidence,C,keepFraction);
end
W = (W+W')/2; W(1:n+1:end)=0;
W = W/max(max(W(:)),eps);
end

function W = mapa_affinity(Z,nS,yS,pseudo,keep,alpha,sigma)
n = size(Z,1); XS=Z(1:nS,:); XT=Z(nS+1:end,:);
D2 = sqdist(XS,XT);
A = (1-alpha)*exp(-D2/(2*sigma^2));
semantic = bsxfun(@eq,yS(:),pseudo(:)') & repmat(keep(:)',nS,1);
A = A + alpha*double(semantic);
W = zeros(n,n);
W(1:nS,nS+1:end)=A; W(nS+1:end,1:nS)=A';
end

function W = mlrgl_graph(X,C,p)
% TMVC adapted to a UDA graph interface. The original final calls to
% SpectralClusteringqi and myNMIACC are intentionally removed.
V=numel(X); n=size(X{1},1);
La=cell(1,V); H=cell(1,V); Q=cell(1,V); HH=cell(1,V);
for v=1:V
    X{v}=normalize_rows(X{v}); S=knn_graph(X{v},p.knn_k);
    D=diag(sum(S,2)); invD=diag(1./sqrt(max(diag(D),eps)));
    La{v}=invD*S*invD; H{v}=zeros(n,C); Q{v}=eye(n); HH{v}=zeros(n,n);
end
w2=1-p.w1; L=zeros(n,n,V); L1=L; L2=L; L3=L;
U1=L; U2=L; U3=L; mu=p.mu;
for it=1:p.max_iter
    X1=zeros(n,n,V);
    for v=1:V
        Z=L(:,:,v);
        G=p.r1*La{v}+Q{v}*(0.5*(Z+Z')-0.5*HH{v})*Q{v};
        H{v}=smallest_eigs(G,C); HHT=H{v}*H{v}';
        Q{v}=diag(1./sqrt(max(diag(HHT),eps)));
        Hhat=Q{v}*H{v}; HH{v}=Hhat*Hhat';
        X1(:,:,v)=abs(HH{v})+abs(HH{v}');
    end
    T=(X1+L+U1/mu)/(1+mu);
    for v=1:V, L1(:,:,v)=prox_nuclear(0.5*(T(:,:,v)+T(:,:,v)'),p.w1/(1+mu)); end
    T=(X1+L+U2/mu)/(1+mu);
    for j=1:n, L2(:,j,:)=reshape(prox_nuclear(reshape(T(:,j,:),n,V),w2/(1+mu)),n,1,V); end
    T=(X1+L+U3/mu)/(1+mu);
    for j=1:n, L3(:,j,:)=reshape(prox_l21(reshape(T(:,j,:),n,V),(w2*p.alpha)/(1+mu)),n,1,V); end
    L=(L1+L2+L3-(U1+U2+U3)/mu)/3;
    change=max([max(abs(X1(:)-L1(:))),max(abs(X1(:)-L2(:))),max(abs(X1(:)-L3(:))), ...
        max(abs(L(:)-L1(:))),max(abs(L(:)-L2(:))),max(abs(L(:)-L3(:)))]);
    if change<p.tol, break; end
    U1=U1+mu*(L-L1); U2=U2+mu*(L-L2); U3=U3+mu*(L-L3);
    mu=min(p.rho*mu,1e10);
end
fprintf('MLRGL iterations: %d\n',it);
W=sum(abs(L),3); W=(W+W')/2; W(1:n+1:end)=0;
W=W/max(max(W(:)),eps);
end

function [acc,mca] = eval_with_graph(W,Xsub,FS,YS,FT,YT,C,type,cfg)
V=numel(FS); p=zeros(size(FT{1},1),C);
for v=1:V
    [~,B]=lpp_basis(Xsub{v},W,cfg.reduced_dim,type,cfg.ridge);
    ZS=normalize_rows(FS{v}*B); ZT=normalize_rows(FT{v}*B);
    p=p+prototype_cluster_posterior(ZS,YS{v},ZT,C);
end
[~,pred]=max(p/V,[],2); [acc,mca]=metrics(pred,YT,C);
end

function [acc,mca] = eval_stage1(FS,YS,FT,YT,C)
p=zeros(size(FT{1},1),C);
for v=1:numel(FS)
    p=p+prototype_cluster_posterior(normalize_rows(FS{v}),YS{v},normalize_rows(FT{v}),C);
end
[~,pred]=max(p/numel(FS),[],2); [acc,mca]=metrics(pred,YT,C);
end

function [Z,B] = lpp_basis(X,W,r,type,ridge)
X=normalize_rows(X); n=size(X,1); d=size(X,2); W=(W+W')/2;
if strcmpi(type,'signed'), D=diag(sum(abs(W),2)); else, D=diag(sum(W,2)); end
L=D-W; A=X'*L*X; Bm=X'*D*X+ridge*eye(d);
r=min([r,d,n-1]); A=(A+A')/2; Bm=(Bm+Bm')/2;
try
    [B,E]=eigs(A,Bm,r,'smallestreal'); [~,o]=sort(diag(E)); B=B(:,o);
catch
    [V,E]=eig(A,Bm); [~,o]=sort(real(diag(E))); B=real(V(:,o(1:r)));
end
Z=X*B;
end

function p = prototype_cluster_posterior(ZS,yS,ZT,C)
centers=zeros(C,size(ZS,2));
for c=1:C, centers(c,:)=mean(ZS(yS==c,:),1); end
Dsrc=sqdist(ZT,centers);
try
    [id,ct]=kmeans(ZT,C,'Start',centers,'Replicates',1,'MaxIter',100,'EmptyAction','singleton');
    mapped=zeros(C,size(ZT,2)); filled=false(C,1);
    for k=1:C
        [~,c]=min(sqdist(ct(k,:),centers),[],2); mapped(c,:)=ct(k,:); filled(c)=true;
    end
    mapped(~filled,:)=centers(~filled,:); Dtgt=sqdist(ZT,mapped);
catch
    Dtgt=Dsrc;
end
p=max(softmax_rows(-Dsrc),softmax_rows(-Dtgt)); p=p./max(sum(p,2),eps);
end

function pseudo=initial_pseudo(X,nS,yS,C)
p=zeros(size(X{1},1)-nS,C);
for v=1:numel(X), p=p+prototype_cluster_posterior(normalize_rows(X{v}(1:nS,:)),yS,normalize_rows(X{v}(nS+1:end,:)),C); end
[~,pseudo]=max(p,[],2);
end

function [label,conf]=pseudo_from_views(Z,nS,yS,C)
p=zeros(size(Z{1},1)-nS,C);
for v=1:numel(Z), p=p+prototype_cluster_posterior(normalize_rows(Z{v}(1:nS,:)),yS,normalize_rows(Z{v}(nS+1:end,:)),C); end
p=p/numel(Z); [conf,label]=max(p,[],2);
end

function keep=classwise_filter(label,conf,C,fraction)
keep=false(size(label));
for c=1:C
    ix=find(label==c); if isempty(ix), continue; end
    q=quantile(conf(ix),1-fraction); keep(ix)=conf(ix)>=q;
end
end

function W=knn_graph(X,k)
n=size(X,1); k=min(k,n-1); W=zeros(n,n); D=sqdist(X,X);
for i=1:n
    D(i,i)=inf; [ds,ix]=sort(D(i,:),'ascend'); ix=ix(1:k); ds=ds(1:k);
    W(i,ix)=exp(-ds/max(mean(ds),eps));
end
W=max(W,W'); W(1:n+1:end)=0;
end

function U=smallest_eigs(A,r)
A=(A+A')/2; n=size(A,1); r=min(r,n-1);
try, [U,~]=eigs(A,r,'smallestreal');
catch, [V,D]=eig(A); [~,o]=sort(real(diag(D))); U=real(V(:,o(1:r))); end
end

function Z=prox_nuclear(X,tau)
[U,S,V]=svd(X,'econ'); Z=U*diag(max(diag(S)-tau,0))*V';
end

function Z=prox_l21(X,tau)
norms=sqrt(sum(X.^2,1)); Z=X.*max(0,1-tau./max(norms,eps));
end

function X=normalize_rows(X)
X=double(X); X=X./max(sqrt(sum(X.^2,2)),eps);
end

function Z=pca_and_normalize(X,r)
X=double(X)-mean(X,1); r=min([r,size(X,1)-1,size(X,2)]);
try, [~,~,V]=svd(X,'econ'); Z=X*V(:,1:r);
catch, Z=X; end
Z=normalize_rows(Z);
end

function D=sqdist(A,B)
D=max(sum(A.^2,2)+sum(B.^2,2)'-2*A*B',0);
end

function P=softmax_rows(A)
A=A-max(A,[],2); E=exp(A); P=E./max(sum(E,2),eps);
end

function [acc,mca]=metrics(pred,truth,C)
acc=mean(pred(:)==truth(:)); rec=zeros(C,1);
for c=1:C, ix=truth==c; if any(ix), rec(c)=mean(pred(ix)==c); end, end
mca=mean(rec);
end
