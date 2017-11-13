function model = build_model(PPP,Candidates,docs,sParam)

R=cell(sParam.modelFrom,1);
rp=1+floor(rand(sParam.modelFrom,1)*(PPP.PagesNumber));
rw=rand(sParam.modelFrom,1);
% choosing random candidates
for i=1:sParam.modelFrom
    pg=rp(i);
    if size(Candidates{pg},1)==0        
        w=1+floor(rw(i)*size(Candidates{pg-1},1));
        c=Candidates{pg-1}(w,:);
        BW = imread(PPP.BWName{pg-1});
%         DOC = docs(pg-1).Im;
    else
        w=1+floor(rw(i)*size(Candidates{pg},1));
        c=Candidates{pg}(w,:);
        BW = imread(PPP.BWName{pg});
%         DOC = docs(pg).Im;
    end
    R{i}=BW(c(3):c(4), c(1):c(2));
%       R{i}=DOC(c(3):c(4), c(1):c(2));
%       figure
%       imshow(R{i})
%       close
end

%Creating the model matrix
model.H=get_feature_descriptor(R, sParam);
model.P=1;
model.n=1;

model.Y=cell(sParam.modelTo,1);
fl=floor(sParam.modelFrom/sParam.modelTo);
for i=1:length(model.Y)
    model.Y{i}=(i-1)*fl+1:i*fl;
end
