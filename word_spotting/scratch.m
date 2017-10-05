function [  ] = scratch( t )
root = '/home/lioruzan/tehilim_data/11Q5/976-needs stitching/IR';
addpath(root);
I = imread('P976-Fg004-R-C01-R01-D04082013-T144100-LR924 _012.jpg');
I = imresize(I, 0.1);
I = ~(I>135);
%imshow(I)
%%

CC = bwconncomp(I);
numPixels = cellfun(@numel,CC.PixelIdxList);
[~,idx] = max(numPixels);
I(CC.PixelIdxList{idx}) = 0;
U = sort(unique(numPixels));
[S,IX] = sort(numPixels);
i=1;

for j=1:numel(U)
    oim = zeros(size(I));
    v = U(j);
    c = 0;
    fprintf('got here, j: %d numel(U): %d, v: %d\n',j,numel(U),v)
    while i <= numel(S) && S(i) == v 
        if c < 21 && v > 5
            idx = IX(i);
            oim(CC.PixelIdxList{idx}) = 1;
            
            if mod(c,7) == 0
                figure
                imshow(oim)
            end
            c = c+1;
        end
        i = i+1;
    end
    close all
    
    
end

end

