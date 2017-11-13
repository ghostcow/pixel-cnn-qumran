function [Candidates, AllCandidates]=prepare_segments (Segments)
    Candidates = cell(numel(Segments),1);                                           %segments
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
            addr=[repmat(pg,wn,1), (1:wn)', (1:wn)'];
        end
        Candidates{pg} = [loc(:,1),loc(:,1)+loc(:,3)-1, loc(:,2),loc(:,2)+loc(:,4)-1, addr];  %x1,x2,y1,y2,page,word
    end
    Candidates2 = Candidates(~cellfun('isempty',Candidates))  ;
    AllCandidates=cell2mat(Candidates2);                   
end