function XD =  cellArray2mat(m,num)

XD = single(zeros(m,250));
a = matfile('XD.mat');
p=0;
for i=1:num
    fprintf('itter %d\n', i);
    b = a.XD(i,1);
    b = b{1};
    b = b';
    sz=size(b);
    XD(p+1:p+sz(1),1:sz(2))=b;
    p=p+sz(1);
end
