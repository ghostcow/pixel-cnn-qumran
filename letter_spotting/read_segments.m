function S=read_segments(segDirPath, pagesNames, segExt, cache)

sfile = fullfile(segDirPath, 'S.mat');
if cache==1 && exist(sfile, 'file')
    load(sfile);
    return
end

numPages=numel(pagesNames);
S = cell(numPages,1);               

for i=1:numPages
    fname = fullfile(segDirPath,  [pagesNames{i} segExt]);    % full path to file
    xml=xml2struct(xmlread(fname));
    n = length(xml.words.word);
    loc = zeros(n,4);
    for w=1:n
        word = xml.words.word{w}.Attributes;
        loc(w,:)=[ str2double(word.x),  str2double(word.y),  str2double(word.width),  str2double(word.height)];
    end
    S{i}=loc;
end

if cache > 0
    save(sfile, 'S');
end
