function draw_squares(Segments,PPP,docs,segDir)
M = cell(numel(Segments),1);     %segments
num = numel(Segments);
if ~exist(sprintf('%s/Cand',segDir))
    mkdir(sprintf('%s/Cand',segDir));
end

if ~exist(sprintf('%s/Cand2',segDir))
    mkdir(sprintf('%s/Cand2',segDir));
end

% if ~exist(sprintf('%s/crops',segDir))
%     mkdir(sprintf('%s/crops',segDir));
% end

for pg = 1:num
    loc=Segments{pg};
    if isempty(loc)
        continue
    end
    M{pg} = [loc(:,1), loc(:,2),loc(:,3), loc(:,4)];  %x1,x2,width,height,page,word
    rectangles = M{pg};
    new = docs(pg).Im;
    
    RGB = repmat(new,[1,1,3]); % convert I to an RGB image
    
    % crop rectangles and save them to disk
%     for j=1:size(rectangles,1)
%         im = imcrop(new,rectangles(j,:));
%         name = sprintf('%s/crops/image%d_%03d.jpg',segDir, pg, j);
%         imwrite(im, name);
%     end
    
    J = insertShape(RGB, 'Rectangle', rectangles, 'Color', 'red');
    name = sprintf('%s/Cand/image%d_squars_new.jpg',segDir, pg);
    imwrite(J,name);
        
    I2 = imread(PPP.PagesName{pg});
    RGB2 = repmat(I2,[1,1,3]); % convert I to an RGB image
    
    J = insertShape(double(RGB2), 'Rectangle', rectangles, 'Color', 'red');
    name = sprintf('%s/Cand2/image%d_squars_new_BW.jpg',segDir, pg);
    imwrite(J,name);  
end
