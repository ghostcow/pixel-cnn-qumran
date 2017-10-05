bwMedian = ones(17)*0.5;
imgExt = '.png';
seg_free   = 'comp/Segmentation-free_';
dataset = 'Dataset';
pathToDocs = 'Documents/';                                                 % documents directory (inside segmentation free directory)
pagesDir = [seg_free, dataset];
d2 = dir( fullfile(pagesDir,pathToDocs,['*' imgExt]) );    % list all files
d2 = {d2.name}';
numDocs=17;


% reading the documents and creating the docs struct
for i=1:numDocs
    pathToImage = fullfile(pagesDir,pathToDocs,d2{i});
    [I,map] = imread(pathToImage);
    if ~isempty(map)
        I = ind2gray(I,map);
    end
    if (size(I,3)==4),         I=I(:,:,1:3);              end
    if (size(I,3)>1),            I=rgb2gray(I);    end
    img2 = im2double(I);
    bimg = img2<median(img2(:))*bwMedian(i);
    BW=uint8(bimg*255);   %make image BlackWhite inverted
    imshow(BW)
    fprintf('%d %s\n',i,pathToImage);
end

% 
% img2 = im2double(img2);
% bimg = img2<median(img2(:))*bwMedian;
% 
% BW=uint8(bimg*255);   %make image BlackWhite inverted
% BW_path = fullfile(pagesDir,BW_dir,sprintf( 'BW_%d.png', i));