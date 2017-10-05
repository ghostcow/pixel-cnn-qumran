function M=get_feature_descriptor(X, sParam,p)
    if nargin<3
        p=0;usedp=0;
    else
        usedp=1;
    end
    n=length(X);
    D=build_descriptor(X{1}, sParam,usedp,p);
    M=single(zeros(n,length(D)));
    M(1,:)=D;
    for i=2:n
        if isempty(X{i})
            disp i
        end
%         fprintf('building descriptor %d\n', i)
        M(i,:) = build_descriptor(X{i}, sParam,usedp,p);
    end
    t=31*(sParam.H/sParam.CS)*(sParam.W/sParam.CS);
    M=[nrm2(M(:,1:t)), nrm2(M(:,t+1:end))];
    if usedp
        M=M*p;
    end
end

function D=build_descriptor(im, sParam,usedp,p)
    im = double(im);

%     figure
%     imshow(im)
    I = single(imresize(im, [sParam.H,sParam.W]));
%     I=myimresize(double(im),sParam.H,sParam.W);
%     figure
%       imshow(uint8(I));
%       close all;
    if strcmp(sParam.fdesc,'hog')
        D = fdesc(I, 1, sParam.CS);
    elseif strcmp(sParam.fdesc,'lbp')
        D = fdesc(I, 2, sParam.CS);
    else %hog+lbp
        D = [fdesc(I, 1, sParam.CS), fdesc(I, 2, sParam.CS)];
    end
    if usedp
        D=D*p;
    end
end

function D=fdesc(I, type, cs)
    if type==1
        D = vl_hog(I, cs) ;
    else
        D = vl_lbp(I, cs) ;
    end
     D=reshape(D,1,[]);
end