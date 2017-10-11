function save_images_results(XML, docs,Results,sParam)

b = exist('results');
if b ~= 7
    mkdir('results');
     cd('results');
else
    d = dir('results');
    if length(d) <=2
        cd('results');
    else
        disp('results directory already exists and it is not empty - returning...');
        return
    end
end

for i=1:numel(XML)
    a = XML(i);
    a = a{1};
    Name = a.QueryName{1};
    dir_name = sprintf('%d_%s' ,i, Name);
    mkdir(dir_name);
    cd(dir_name);
    for j=1:min(sParam.numImagesToSave,length(a.Results))
        res = a.Results(j);
        res = res{1};
        doc = Results{i}(j,1);
        new = docs(doc).Im;
        b = new(res.y:res.y+res.h-1,res.x:res.x+res.w-1);
        file = sprintf('res_%d.jpg', j);
        imwrite(b, file, 'jpg');
    end
    cd('../');
end
cd('../');
end
