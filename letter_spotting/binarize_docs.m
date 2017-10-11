% cd ('/specific/disk2/home/adis2/NEW_word_spotting');
addpath('binarization');
ImagesDir = 'images_genizah';
files = dir( fullfile(ImagesDir,['*' 'jpg']) );    % list all files
files = {files.name}';

for i=1:numel(files)
    close all;
    fname = fullfile(ImagesDir,files{i}); 
    I=imread(fname);   
%     I_resized = imresize(I,0.5);
    figure
    imshow(I);
    % Base algorithm with static parameters
    tic
    bimg = binarizeImage(I);
%     [bimg,c,thi] = binarizeImageAlg2(I_resized); toc
    a = ~bimg;
    figure; 
    imshow(a)
%     imwrite(a,'bin_resized1.jpg');
end
