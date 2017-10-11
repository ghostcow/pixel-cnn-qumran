
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
%    new = docs(i).Im;
    new = imread(p{i});
    name = sprintf('%s/Cand3/candidates_%d',segDir,i);
    mkdir(name);

    for j=1:size(a,1)
%         y1 = max(1,a(j,3)-5);
%         y2 = min(size(new,1),a(j,4)+5);
%         x1 = max(1,a(j,1)-5);
%         x2 = min(size(new,2),a(j,2)+5);
        y1 = max(1,a(j,3));
        y2 = min(size(new,1),a(j,4));
        x1 = max(1,a(j,1));
        x2 = min(size(new,2),a(j,2));
        b = new(y1:y2,x1:x2);
        name = sprintf('%s/Cand3/candidates_%d/cand%d.jpg',segDir,i, j);
        %         b2 = pic2(a(j,3):a(j,4),a(j,1):a(j,2));
        %name2 = sprintf('%s/Cand2/candidates2_%d/cand%d.jpg',segDir,i, j);
        imwrite(b, name, 'jpg');
        %         imwrite(b2, name2, 'jpg');
    end
end

end
