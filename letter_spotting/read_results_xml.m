function XML=read_results_xml(file)

queriesXml = xml2struct(xmlread(file));
queries=queriesXml.RelevanceListings.Rel;
XML   = cell(numel(queries),1);                   

for i=1:numel(queries)
    q=queries{i}.Attributes;
    XML{i}.QueryName=q.queryid;
    words=queries{i}.word;
    XML{i}.Results=cell(numel(words),1);
    for j=1:numel(words)
        word = words{j}.Attributes;
        XML{i}.Results{j}=struct('doc',word.document,'x',str2double(word.x),'y',str2double(word.y),'w',str2double(word.width),'h',str2double(word.height));
    end
end
