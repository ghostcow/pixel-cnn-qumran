function [queries]=read_queries2(queriesDirPath, queriesFile, cache)
% OUTPUT: queries - struct array. Each element correspond to one query.
%                   each struct contains the following fields - 
%                   Im - query image
%                   gttext - string with query name

qfile = fullfile(queriesDirPath, 'Q.mat');
% Load queries file if exists and return, if not continue
if cache==1 && exist(qfile, 'file')
    load(qfile);
    return
end

fileQueries= fullfile(queriesDirPath, queriesFile);
fid = fopen(fileQueries, 'rt');
input = textscan(fid, '%s %s');
nWords = length(input{1});
            
for i=1:nWords
    Path_Im = fullfile(queriesDirPath,input{1}(i));
    Path_Im = Path_Im{1};
    Im = imread(Path_Im);  
    queries(i).Im = Im;
    queries(i).gttext = input{2}(i);
end

% save queries
if cache > 0
    save(qfile, 'queries');
end