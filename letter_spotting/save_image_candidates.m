
function save_image_candidates(Candidates, docs,PPP,segDir)

if ~exist(sprintf('%s/Cand3',segDir))
    mkdir(sprintf('%s/Cand3',segDir));
end

p = PPP.BWName;
disp('Start parfor');
for i=1:numel(Candidates)
    fprintf('Process doc %d\n',i);
    a = Candidates(i);
    a = a{1};
    new = imread(p{i});
    name = sprintf('%s/letters_%d',segDir,i);
    mkdir(name);

    for j=1:size(a,1)
        y1 = max(1,a(j,3));
        y2 = min(size(new,1),a(j,4));
        x1 = max(1,a(j,1));
        x2 = min(size(new,2),a(j,2));
        b = new(y1:y2,x1:x2);
        name = sprintf('%s/letters_%d/%s_%d.jpg', segDir, i, PPP.Names{i}, j);
        imwrite(b, name, 'jpg');
    end
end

end
