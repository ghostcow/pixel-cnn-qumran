function I=myimresize(I,h,w)

I=single(imresize(I,[h,w]));
%I=imresize(single(I),[h,w]);