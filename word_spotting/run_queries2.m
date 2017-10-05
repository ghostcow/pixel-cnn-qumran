function [Results,scores,QXD] = run_queries2(M,queries,sParam,resToSave,queriesDir,cache)
%OUTPUT: Results (cell array) - number of elements is the number of queries 
%         Each cell is a matrix with the results sorted in descending order
%         Each line int the matrix is one result - [page number, candidate number, largest CC number]

Rfile = fullfile(queriesDir, 'Results.mat');
%loading the results file if exists
if cache==1 && exist(Rfile, 'file')
    load(Rfile);
    return
end

qn = length(queries);
if (sParam.showDebug) ,            disp (['  -  Running ' num2str(qn) ' Queries']);   tic; end
bwimages = improve_query_images(queries,sParam);
D=double(zeros(size(M.CY,1), qn));
QXD=cell(numel(sParam.q_move_i)*numel(sParam.q_move_j)+8*numel(sParam.q_move_k),1);
qxdi=1;

%calculating the distances between the queries and all the candidates
% using jittering proceess
for i=sParam.q_move_i
    for j=sParam.q_move_j
        W=immove(bwimages,i,j);
        emptyCells = cellfun(@isempty,W);
        if sum(emptyCells)~=0
            disp sum
        end
        [queries,QXD{qxdi}] = classifier(W,M.Model,sParam);
        qxdi=qxdi+1;
        dist=M.CY*queries;
        D=max(D,dist);
    end
end
for i=sParam.q_move_k
    for z=1:8
        W=immove(bwimages,i,0,z);
        emptyCells = cellfun(@isempty,W);
        if sum(emptyCells)~=0
            disp sum
        end
        [queries,QXD{qxdi}] = classifier(W,M.Model,sParam);
        qxdi=qxdi+1;
        dist=M.CY*queries;
        D=max(D,dist);
    end
end

[score,ind]=sort(D,'descend');
ind=ind(1:resToSave,:);
Results = cell(qn,1);
RRR = cell(qn,1);

%re-ranking if sParam.runPsotProcessing does not equal zero
if sParam.runPostProcessing~=0
    n=min(sParam.runPostProcessing,resToSave);
    %MXD=cell2mat(M.XD);
    for i=1:qn
        iq=ind(1:n,i);
        ww=M.XD(iq,:);
        qq=cell2mat(cellfun(@(x) x(i,:), QXD, 'UniformOutput', false));
        wwq=qq*ww';
        [score2,qind]=sort(max(wwq),'descend');
        Results{i} = M.AllCandidates([iq(qind);ind(n+1:end,i)],5:end);
        scores{i} = score2;
    end
else
    for i=1:qn
        Results{i} = M.AllCandidates(ind(1:resToSave,i),5:end);
        scores{i} = score(1:resToSave,i);
    end
end

%removing overlapping windows
for i=1:qn
    RRR{i} = Results{i}(:,1:2:3);
    [~,ind]=unique(RRR{i},'rows','stable');
    Results{i}=Results{i}(ind,:);
end

%saving the results
if (sParam.showDebug) ,               toc; disp '  -  Running Queries - DONE';    end
if cache > 0
    save(Rfile, 'Results','scores','QXD');
end
end


function J=improve_query_images(queries,sParam)

for i=1:length(queries)
    im=queries(i).Im;
    if islogical(im)
        bimg=~im;
    else
        im = im2double(im);
        bimg = im<median(im(:))*sParam.bwMedian;
    end
    bw=uint8(bimg*255);   %make image BlackWhite inverted
    J{i}=bw;
% J{i}=im;

end

end
