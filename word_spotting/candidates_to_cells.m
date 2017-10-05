function X = candidates_to_cells(BWName,DOC, b )

n=size(b,1);
X=cell(n,1);
BW = imread(BWName);
for j=1:n  
    X{j} = BW(b(j,3):b(j,4),b(j,1):b(j,2));
% X{j} = DOC(b(j,3):b(j,4),b(j,1):b(j,2));
% figure
% imshow(X{j})
% close
end
end

