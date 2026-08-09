function P = LPP(data, W, options, laplacian_type)
data = double(data);
W = double(W);

if strcmpi(laplacian_type, 'ordinary')
    D = diag(sum(W, 2));
    L = D - W;

    Sl = data' * L * data;
    Sd = data' * D * data;

elseif strcmpi(laplacian_type, 'signed')
    Dabs = diag(sum(abs(W), 2));
    Lsgn = Dabs - W;

    Sl = data' * Lsgn * data;
    Sd = data' * Dabs * data;

else
    error('Unknown laplacian_type: %s', laplacian_type);
end

Sl = (Sl + Sl') / 2;
Sd = (Sd + Sd') / 2;
Sl = Sl + options.alpha * eye(size(Sl, 2));

opts.disp = 0;
[P, ~] = eigs(double(Sd), double(Sl), ...
              options.ReducedDim, 'la', opts);

for i = 1:size(P, 2)
    if P(1, i) < 0
        P(:, i) = -P(:, i);
    end
end