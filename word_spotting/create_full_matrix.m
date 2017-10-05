function [y] = create_full_matrix(b,BW,sParam,P)

n=size(b,1);
X=cell(n,1);
for j=1:n
    num_doc = b(j,5);
    BWName = BW{num_doc};
    B = imread(BWName);
    X{j} = B(b(j,3):b(j,4),b(j,1):b(j,2));

%     imshow(X{j})
%     close
end

    y=single(get_feature_descriptor(X, sParam,P));
end

