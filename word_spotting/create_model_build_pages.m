function Model = create_model_build_pages(Model,PPP,modelFile,cache,docs,sParam)
% OUTPUT: M - struct with the following fields: 

% Candidates (cell array) - each cell correspond to one document and contains a matrix. 
%                           every line in the matrix represent a candidate -
%                           [x1 (left),x2 (right), y1 (top), y2 (bottom), page number, candidate number, largest CC number]

% AllCandidates (martix)- one matrix containing all the candidates.

% Model (struct) - Model.H is the model matrix
%                  Model.P = 1
%                  Model.n = 1
%                  Model.Y - a variable for max-pooling
% 
% CY - final candidates matrix.
% XD - full length descriptors candidate matrix


%Building candidates matrix
tic
[Model.CY ,Model.XD] = build_pages_new(PPP, Model.Model, Model.Candidates, docs,sParam);
    if (sParam.showDebug)
        toc; disp '  -  Building - DONE'
    end

%save model file
    if cache >0
        save (fullfile(modelFile), 'Model', '-v7.3');
    end

end

