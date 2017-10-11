function Bench=build_prepare2 (docs,sParam,pagesDir)

Bench.PagesNumber=length(docs);
Bench.PagesName = cell( Bench.PagesNumber,1);
Bench.BWName = cell( Bench.PagesNumber,1);
Bench.ccPixels = cell( Bench.PagesNumber,1);
Bench.ccBox = cell( Bench.PagesNumber,1);
Bench.ccCentroid = cell( Bench.PagesNumber,1);

%directories for saving the new images
Pages_dir = 'Pages';
BW_dir='BW';

num_proccess = Bench.PagesNumber;
for i=1:num_proccess
    
    img2=docs(i).Im;
    if isempty(img2)
        continue
    end
    if islogical(img2)
        bimg=~img2;
    else
        img2 = im2double(img2);
%         r=ceil(rand(ceil(sqrt(numel(img2))),1)*numel(img2)); %for performance we calc median for sqrt amount of elemnts
        bimg = img2<median(img2(:))*sParam.bwMedian;
    end
    BW=uint8(bimg*255);   %make image BlackWhite inverted
    BW_path = fullfile(pagesDir,BW_dir,sprintf( 'BW_%d.png', i));

    %save the BW page
    imwrite(BW,BW_path);
    Bench.BWName{i}=BW_path;

    %preprocessing the pages - removing noise and big CC.
    CC = bwconncomp(BW);
    BB=regionprops(CC,'BoundingBox','Centroid');
    BF=reshape(cell2mat({BB(:).('BoundingBox')}),4,[]);%extract field - Bounding boxes
    BC=reshape(cell2mat({BB(:).('Centroid')}),2,[]);%extract field - centroids

    BF=BF+repmat([0.5;0.5;-1;-1],1,size(BF,2));
    BI=(BF(3,:)<sParam.bwWTooBig & BF(4,:)<sParam.bwHTooBig & BF(4,:)>sParam.HTooSmall & BF(3,:)>sParam.WTooSmall );
        BI(cellfun(@length,CC.PixelIdxList,'uni',true)<sParam.bwTooSmall)=0;
    BI(cellfun(@length,CC.PixelIdxList,'uni',true)>sParam.bwTooBig)=0;
    
    Bench.ccCentroid{i}=BC(:,BI);
    BFI=BF(:,BI);
    Bench.ccBox{i}=[BFI(1,:);BFI(1,:)+BFI(3,:);BFI(2,:);BFI(2,:)+BFI(4,:);repmat(i,1,size(BFI,2))];
    Bench.ccPixels{i} = CC.PixelIdxList(BI)';
    Bench.ccSize{i}= cellfun(@length,  Bench.ccPixels{i});
    newBW=false(size(BW));
    newBW(cell2mat(Bench.ccPixels{i}))=1;
    Pages_path = fullfile(pagesDir,Pages_dir,sprintf( 'Page_%d.png', i));

    %save the new BW page
    imwrite(newBW,Pages_path);
    imshow(newBW)
    Bench.PagesName{i} = Pages_path;

    % for debugging:
%         figure
%         imshow(BW);
%         figure
%         imshow(newBW);
%         newBW=false(size(BW)); newBW(cell2mat(CC.PixelIdxList(~BI)'))=1;
%         figure
%         imshow(newBW);   %     DEBUG: show what is filtered out
%         close all
end
end
