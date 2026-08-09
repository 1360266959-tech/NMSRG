function [ProcessData]=NormalizeData_matrix(X)

[nFea,nSmp] = size(X);
for i = 1:nSmp
     ProcessData(:,i) = X(:,i) ./ max(1e-12,norm(X(:,i)));
end