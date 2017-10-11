function create_result_image(numpic,numLines,queries)

num_col = numpic/numLines;

ResIm.ImagesNumber=numpic;
ResIm.Images = cell( numpic,1);
for qq = 1:numel(queries)
    q = queries(qq).gttext{1};
    for i=1:numpic
        picName = fullfile(sprintf('results/%d_%s/res_%d.jpg',qq,q,i));
        pic = imread(picName);
 
        H = size(pic,1);
        W = size(pic,2);
        
        ResIm.H(i)= H;
        ResIm.W(i) = W;
        ResIm.Images{i}  = pic;
    end

    maxH = max(ResIm.H);
    maxW = max(ResIm.W);

    meanH = mean(ResIm.H);
    meanW = mean(ResIm.W);
    H = ceil(meanH)*5+15;
    W = ceil(meanW)*num_col+15;

    for m = 1:num_col
        maxW_vec(m) = max(ResIm.W((m-1)*10+1:10*m));
    end
    Res1 = zeros(H,W);
    Res1 = im2uint8(Res1);
    col = 1;

    for j=1:num_col

        row = 1;
        for i=1:10
            H = ResIm.H((j-1)*10+i);
            W = ResIm.W((j-1)*10+i);
            Res1(row:row+H-1,col:col+W-1) =(ResIm.Images{(j-1)*10+i});
            row = row + H +2;
        end

        col = col+ maxW_vec(j) + 2;
    
    end
%     imshow(Res1)
    file = sprintf('results/%s.png', q);
    imwrite(Res1,file,'png');

end
disp '  -  Done'
    
    