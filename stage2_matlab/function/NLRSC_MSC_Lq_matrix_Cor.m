function [Coe,Conver_iter,iter] = NLRSC_MSC_Lq_matrix_Cor(X,W,mu,rho,epsilon,lambda,beta,q)
nV = length(X);
n = size(X{1},2);  
for v = 1:nV
    C{v} = zeros(n,n);  
    P{v} = zeros(n,n); 
    S{v} = zeros(n,n); 
    M1{v} = zeros(n,n); 
    M2{v} = zeros(n,n); 
    dV = size(X{v},1);
    M3{v} = zeros(dV,n); 
    E{v} = zeros(dV,n);
end
Coe = zeros(n,n);
mu_max = 1e10;
maxIter = 100;  % Reduced from 500 to 100 for faster convergence
iter = 0;

fprintf('Starting NLRSC optimization (nV=%d, n=%d)...\n', nV, n);
while iter<maxIter
%     finish = 0;
    iter = iter + 1;
    if mod(iter, 10) == 0
        fprintf('  Iteration %d/%d\n', iter, maxIter);
    end

    % Update for each view
    for v = 1:nV
        % update P
        A = C{v} + M1{v}./mu;
        if q == 2/3
            [U,sigma,V] = Two_thirds_norm(A,2/mu);
            P{v} = U*diag(sigma)*V';
        elseif q == 1/2
            [U,sigma,V] = Half_norm(A,2/mu);
            P{v} = U*diag(sigma)*V';
        end
        
        % update S
        B = C{v}+M2{v}./mu;
        tao = ((2*lambda)/mu).*W{v};
        if q == 2/3
            S{v} = L23_norm_matrix(B,tao);
        elseif q == 1/2
            S{v} = Half_norm_matrix(B,tao);
        end
        
        % Update C under the exact constraint diag(C) = 0
        Chat = X{v}' * (mu .* X{v} - mu .* E{v} + M3{v}) ...
             + mu .* P{v} - M1{v} + mu .* S{v} - M2{v};
        
        Bmat = (2 * mu) .* eye(n) + mu .* (X{v}' * X{v});
        C{v} = zeros(n, n);
        
        for j = 1:n
            idx_free = [1:j-1, j+1:n];
            C{v}(idx_free, j) = Bmat(idx_free, idx_free) \ Chat(idx_free, j);
        end

        % Update E: exact proximal update for beta * ||E||_{2,1}
        tempE = X{v} - X{v} * C{v} + M3{v} ./ mu;
        E{v} = solve_l1l2(tempE, beta / mu);
   
        % Check
        stopC = max([max(max(abs(C{v} - P{v}))),max(max(abs(C{v} - S{v}))),max(max(abs(X{v} - X{v}*C{v}-E{v})))]); %����Լ����
        stopC_Views(1,v) = stopC;

        % Update multipliers
        M1{v} = M1{v} + mu.*(C{v} - P{v});
        M2{v} = M2{v} + mu.*(C{v} - S{v});
        M3{v} = M3{v} + mu.*(X{v} - X{v}*C{v}-E{v});
    end

    % update mu (should be outside the view loop)
    mu = min(rho*mu, mu_max);

    Conver_max = max(stopC_Views);
    Conver_iter(iter,1) = Conver_max;
    
    finish = 1;
    for v = 1:nV
        if stopC_Views(1,v) >= epsilon 
            finish = 0;
            break;
        end
    end
    if finish == 1
        for v = 1:nV
            Coe = Coe + C{v};
        end
        fprintf('NLRSC converged at iteration %d\n', iter);
        break;
    end
end

if iter >= maxIter
    fprintf('NLRSC reached maxIter=%d without full convergence\n', maxIter);
    for v = 1:nV
        Coe = Coe + C{v};
    end
end

end