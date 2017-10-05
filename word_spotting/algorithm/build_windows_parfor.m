function Segments=build_windows_parfor (PPP,sParam,segDir,segFile,cache)
% OUTPUT: Segments - cell array. each cell correspond to one document and contains a matrix.
%  every line in the matrix represent a candidate -
% [x1 (left),y1 (top),width,height,page number, largest CC number]

%Load the segFile if exist and return, if not continue
if cache==1 && exist([segDir,segFile], 'file')
    disp(' - Loading Segments file')
    load(fullfile(segDir, segFile));
    return
end

if (sParam.showDebug)
    disp '  -  Building Windows'
end

Segments = cell( PPP.PagesNumber,1);
SP=getSpace(PPP, sParam); %calculating space threshold

disp('Start parfor');
ccSize_cell=PPP.ccSize;
BB_cell=PPP.ccBox;
ccC_cell = PPP.ccCentroid;
num = PPP.PagesNumber;
PagesName = PPP.PagesName;
%can change parfor to for if needed
parfor i=1:num
    fprintf('Process doc %d\n',i);
    if isempty(PagesName{i})
        bndry=getCandidates([],[],[],[],i,SP(i),sParam,0,0);
        Segments{i} = bndry;
    else
%         avgLineH = docs(i).avgLineHeight;
%         avgLineSpc = docs(i).avgLineSpacing;
        P = imread(PagesName{i});
        ccSize = ccSize_cell{i};
        BB=BB_cell{i};
        ccC=ccC_cell{i};
        % getting words candidates for current page
        bndry=getCandidates(ccSize,ccC,BB,P,i,SP(i),sParam,[],[]);
        Segments{i}=[bndry(:,1),bndry(:,3),bndry(:,2)-bndry(:,1)+1,bndry(:,4)-bndry(:,3)+1,bndry(:,5),bndry(:,6)];
    end
    disp '  -  Finish windows page'
    disp(i)
end

%save Segments
if cache > 0
    save (fullfile(segDir, segFile), 'Segments', '-v7.3');
end

end

function S=getSpace(PPP, sParam)
S=zeros(PPP.PagesNumber,1);
for pg=1:PPP.PagesNumber
    if isempty(PPP.PagesName{pg})
        continue
    end
    P=imread(PPP.PagesName{pg});
    BB=PPP.ccBox{pg};
    spc=sParam.findSpace;
    T=zeros(size(BB,2),1);
    for i=1:size(BB,2)
        b=BB(:,i);
        T(i)=sum(sum(P(b(3):b(4),b(2)+1:min(b(2)+spc,size(P,2)))));
    end
    S(pg)=mean(T)*0.5;
end
end

function C=getCandidates(ccSize,ccC,BB,P, pg,sp, sParam, avgLine,avgLineS)
if isempty(P)
    C = [];
else
    
    RES=single(zeros(6,sParam.segMaxWin));
    [hh,ww]=size(P);
    count=0;
    %Using the documents data if available
    if ~isempty(avgLine)
        fprintf('%s\n',avgLine);
        findH = 1.8*avgLine;
        findHMin = avgLine-5;
        findT = avgLineS - avgLine - 2;
    else
        findH = sParam.findH;
        findHMin = sParam.findHMin;
        findT = sParam.findT;
    end
    findW = sParam.findW;
    findWMin = sParam.findWMin;
    noSpace = sParam.noSpace;
    findSpace = sParam.findSpace;
    
    for i=1:size(BB,2)
        
        b=BB(:,i);
        F=(b(2)-BB(1,:)<findW) & (BB(2,:)<b(2));
        F=F & (BB(4,:)-b(3)<findH) & (abs(b(3)-BB(3,:))<findT);
        F(i)=1;
        FN=find(F);
        BFF=BB(:,FN);
        
        %if no_space=1 the next line is disabled
        if (sum(sum(P(b(3):b(4),b(2)+1:min(b(2)+findSpace,ww))))>sp) & (noSpace~=1)
            continue % no space before the word (means it is not the beginning of the word)
        end
        
        for top=unique(BFF(3,BFF(3,:)<=b(3)))
            ii=BFF(3,:)<=top;
            for left=unique(BFF(1,BFF(1,:)<=max(BFF(1,ii))))
                ij=BFF(1,:)>=left;
                bottoms=BFF(4,ij & BFF(4,:)>=mean(BFF(4, ij & BFF(3,:)<=b(4))));
                if ~isscalar(bottoms), bottoms=unique(bottoms);end
                for bottom=bottoms
                    if bottom<=top
                        continue  %bottom<=top means empty box
                    end
                    mrg=(bottom-top)/4;
                    tm=top+mrg;
                    bm=bottom-mrg;
                    iii=ij & BFF(3,:)<=bm & BFF(4,:)>=tm;
                    
                    if ((min(BFF(3,iii))<top) | (max(BFF(4,iii))>bottom))
                        continue
                    end
                    
                    if ((bottom-top<findHMin || b(2)-left<findWMin) || sum(iii)==0)
                        continue   %box is too small
                    end
                    
                    if (max_space(P(top:bottom,left:b(2)))>findSpace)
                        continue % big space inside the word (means those are two words)
                    end
                    
                    %if no_space=1 the next line is disabled
                    if (sum(sum(P(top:bottom,max(left-findSpace,1):left-1)))>sp) && (noSpace~=1)
                        continue   % no space after the word (means it is not the end of the word)
                    end
                    
                    f1=find(ccC(1,:)<b(2) & ccC(1,:)>left);
                    f2=f1(ccC(2,f1)>top & ccC(2,f1)<bottom);
                    BFB=BB(:,f2);
                    nbb=[min(BFB(1,:));max(BFB(2,:));min(BFB(3,:));max(BFB(4,:))];
                    if isempty(nbb)
                        continue
                    end
                    
                    %inside the boundary of the word there are objects that have centroids inside the box but pixels outside
                    if (nbb(2)>b(2) || nbb(1)<left  || nbb(3)<top || nbb(4)>bottom)
                        continue
                    end
                    count=count+1;
                    left2=max(left-3,1);
                    bottom2=min(bottom+3,hh);
                    top2=max(top-3,1);
                    right2=min(b(2)+3,ww);
                    indf=FN(iii);
                    [~,ind]=max(ccSize(indf));
                    RES(:,count)=[left2; right2; top2; bottom2; pg; 10*indf(ind)];
                    %DEBUG
                    %imshow(P(top2:bottom2,left2:right2));
                    % imshow(P);
                    %rectangle('Position',[left2,top2,right2-left2+1,bottom2-top2+1],'EdgeColor','b');
                    %c=[b(1),b(3),b(2)-b(1)+1,b(4)-b(3)+1];
                    %rectangle('Position',c,'EdgeColor','r');
                end
            end
        end
    end
    C=RES(:,1:count)';
end
end

function maxlen=max_space(pic)
t=diff([false,sum(pic)==0,false]);
p = find(t==1);
if isempty(p),  maxlen=0;return; end
%DEBUG
%imshow(pic)
q = find(t==-1);
maxlen=max(q-p);
end

