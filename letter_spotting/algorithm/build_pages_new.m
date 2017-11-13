function [CY, XDmat]=build_pages_new(PPP,model,Candidates,docs,sParam)

CYarr = cell(PPP.PagesNumber,1);
XDmatArr = cell(PPP.PagesNumber,1);
BW = PPP.BWName;
for pg=1:PPP.PagesNumber 
     fprintf('building page %d\n',pg);
    BWName = BW{pg};
    DOC = docs(pg).Im;
    C = Candidates{pg};
    if size(C,1)==0
        continue
    end    
     X = candidates_to_cells(BWName,DOC,C);   
    [y,XD]=classifier(X,model,sParam);
    XDmatArr{pg}=XD;
    CYarr{pg}=y';
end
 
CY=cell2mat(CYarr);
CY(isnan(CY)) = 0 ;
XDmat=cell2mat(XDmatArr);
CY(isnan(CY)) = 0 ;

