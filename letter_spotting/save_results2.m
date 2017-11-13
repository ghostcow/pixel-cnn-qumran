function XML=save_results2(Results, queries, docs, Locations, file ,n)
% OUTPUT: XML - cell array. length of array is the number of queries.
%               each cell is a struct array containing all the results for one query.
%               each struct in the array contains the following fields - 
%               doc - full path to the document which contains the current result
%               x - x coordinate of the result
%               y - y coordinate of the result
%               w - width  of the result
%               h - height of the result

XML=cell(numel(Results),1);
for i=1:numel(Results)
    XML{i}.QueryName = queries(i).gttext;
    k=min(n,size(Results{i},1));
    XML{i}.Results=cell(k,1);
    for j=1:k
        doc = Results{i}(j,1);
        loc=Locations{doc}(Results{i}(j,2),:);
        XML{i}.Results{j}=struct('doc',docs(doc).pathImage,'x',loc(1),'y',loc(2),'w',loc(3),'h',loc(4));
    end
end
if ~isempty(file)
    save_results_xml(XML, file);
end