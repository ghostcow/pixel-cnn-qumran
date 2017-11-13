function V=get_max_pooling(U,P)

V=zeros(length(P),size(U,2));
for i=1:length(P)
    V(i,:)=max(U(P{i},:),[],1);
end

