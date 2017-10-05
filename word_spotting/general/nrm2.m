function y=nrm2(x)

y=x./repmat(sqrt(sum(x.^2,2)),1,size(x,2));