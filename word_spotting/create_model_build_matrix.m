function M = create_model_build_matrix(PPP,Segments,docs,sParam)
% OUTPUT: M - struct with the following fields: 

% Candidates (cell array) - each cell correspond to one document and contains a matrix. 
%                           every line in the matrix represent a candidate -
%                           [x1 (left),x2 (right), y1 (top), y2 (bottom), page number, candidate number, largest CC number]

% AllCandidates (martix)- one matrix containing all the candidates.

% Model (struct) - Model.H is the model matrix
%                  Model.P = 1
%                  Model.n = 1
%                  Model.Y - a variable for max-pooling


[M.Candidates,M.AllCandidates] = prepare_segments (Segments);
clear Segments

if (sParam.showDebug)
    disp '  -  Building'
    disp '  -    -  Train model';tic
end

%Building the model matrix
M.Model = build_model(PPP, M.Candidates, docs,sParam);
if (sParam.showDebug)
    toc; disp '  -    -  Word Vectors Building'; tic
end

end


function [Candidates, AllCandidates]=prepare_segments (Segments)
Candidates = cell(numel(Segments),1);                                       
for pg = 1:numel(Segments)
    if isempty(Segments{pg})
        Candidates{pg} = [];
        continue
    end
    loc=Segments{pg};
    wn = size(loc,1);
    if size(loc,2) == 6
        addr=[loc(:,5), (1:wn)', loc(:,6)];
    else
        addr=[loc(:,5), (1:wn)'];
    end
    Candidates{pg} = [loc(:,1),loc(:,1)+loc(:,3)-1, loc(:,2),loc(:,2)+loc(:,4)-1, addr];
end
Candidates2 = Candidates(~cellfun('isempty',Candidates))  ;
AllCandidates=cell2mat(Candidates2);
clear Candidates2
end
