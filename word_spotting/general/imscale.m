function I=imscale(I,h,w)

[m,n]=size(I);
newn=n/m*h;
if nargin>2 && newn>w
   newn=w;
end

I=imresize(single(I),[h,newn]);
if nargin>2 && newn<w
    J=single(zeros(h,w));
    x=round((w-newn)/2);
    J(:,x+1:x+size(I,2))=I;
    I=J;
end
