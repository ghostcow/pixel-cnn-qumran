function save_results_xml(XML, file)

F = fopen(file,'W','n','UTF-8');
fprintf(F,'<?xml version="1.0" encoding="utf-8"?>\n');
fprintf(F,'<RelevanceListings xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n');
for i=1:numel(XML)
    q = XML{i}.QueryName;
    q = q{1};
    fprintf(F,'  <Rel queryid="%s">\n',q);
    for j=1:size( XML{i}.Results,1);
        r=XML{i}.Results{j};
        fprintf(F, '    <word document="%s" x="%d" y="%d" width="%d" height="%d" />\n',r.doc,r.x,r.y,r.w,r.h);
    end
    fprintf(F,'  </Rel>\n');
end
fprintf(F,'</RelevanceListings>\n');
fclose(F);
