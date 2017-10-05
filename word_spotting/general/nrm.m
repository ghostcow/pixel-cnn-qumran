function y=nrm(x)

y=x./repmat(sqrt(sum(x.^2,1)),size(x,1),1);