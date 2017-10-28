function  [docs, dic] =read_pages(pagesDir, imgExt, cache,pathToDocs,pfile)
% OUTPUT: docs - struct array . every element in docs is a struct (corresponds to a specific page) with the following fields: 
% H - height
% W = width
% path image - full path to the image location
% Im - image of the document
% Name - document's name

% dic is a dictionary with keys as documents names and values as documents numbers.

% Load the pfile if exist and return, if not continue
if cache==1 && exist(pfile, 'file')
    disp(' - Loading docs file');
    load(pfile);
    return
end

disp(' - Reading documents');

if length(imgExt) == 2
    d2 = [ dir(fullfile(pagesDir,pathToDocs, ['*' imgExt{1}])) ; dir(fullfile(pagesDir, ['*' imgExt{2}]))];
else
    d2 = dir( fullfile(pagesDir,pathToDocs,['*' imgExt]) );    % list all files
end

d2 = {d2.name}';
numDocs = length(d2);

docs = struct('H',cell(numDocs,1),'W',cell(numDocs,1),'pathImage',cell(numDocs,1),'Name',cell(numDocs,1),'Im', cell(numDocs,1));
dic = [];

% reading the documents and creating the docs struct
for i=1:numDocs
        pathToImage = fullfile(pagesDir,pathToDocs,d2{i});
       [docs,dic] = create_struct_docs(docs,d2,pathToImage,dic,i,imgExt);
end


%saving the struct to a mat file
if cache > 0
    save (pfile, 'docs', 'dic', '-v7.3');
end
end


function [docs,dic] = create_struct_docs(docs,d2,pathToImage,dic,j,imgExt)

len = numel(imgExt);

docs(j).pathImage =pathToImage; % full path to file
docs(j).Name = d2{j}(1:end-len);
[I,map] = imread(pathToImage);
if ~isempty(map)
    I = ind2gray(I,map);
end
if (size(I,3)==4),         I=I(:,:,1:3);              end
if (size(I,3)>1),            I=rgb2gray(I);    end

[docs(j).H, docs(j).W] = size(I);
docs(j).Im = I;
new = containers.Map(docs(j).Name,j);
dic = [dic;new];
end



