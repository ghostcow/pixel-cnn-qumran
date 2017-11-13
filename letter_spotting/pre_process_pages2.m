function PPP=pre_process_pages2(docs,sParam,pagesDir,cache,PPP_file)
% OUTPUT: PPP - struct with the following fields: 
% PagesNumber - a scalar 
% PagesName (cell array) - every cell is a string
% BWName (cell array) - every cell is a string
% ccPixels (cell array) - every cell is a matrix
% ccBox (cell array) - every cell is a matrix
% ccCentroid (cell array) - every cell is a matrix
% ccSize (cell array) - every cell is a matrix

%Load the PPP_file if exist and return, if not continue

if cache==1 && exist(PPP_file, 'file')
    disp(' - Loading Pages file')
    load(PPP_file);
    return
end

if (sParam.showDebug)
    disp '  -  Preprocessing Pages'
end

% preprocessing the pages and creating the pages struct
PPP = build_prepare2(docs,sParam,pagesDir);

% save PPP struct
if cache > 0
    save (PPP_file, 'PPP', '-v7.3');
end
