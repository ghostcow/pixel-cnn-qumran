function W=immove(X,i,j,z)
    if i==0&&j==0
        W=X;
        return
    end
    W=cell(length(X),1);
    if nargin>=4
        switch z
            case 1
                for a=1:length(X);W{a}=X{a}(i+1:end,:);end
            case 2
                for a=1:length(X);W{a}=X{a}(:,i+1:end);end
            case 3
                for a=1:length(X);W{a}=X{a}(1:end-i,:);end
            case 4
                for a=1:length(X);W{a}=X{a}(:,1:end-i);end
            case 5
                for a=1:length(X);W{a}=X{a}(i+1:end,i+1:end);end
            case 6
                for a=1:length(X);W{a}=X{a}(1:end-i,i+1:end);end
            case 7
                for a=1:length(X);W{a}=X{a}(1:end-i,1:end-i);end
            case 8
                for a=1:length(X);W{a}=X{a}(i+1:end,1:end-i);end
        end
        return
    end
    x=abs(i);y=abs(j);
    if i>0
        if j>0
            for a=1:length(X);W{a}=immove1(X{a},x,y);end
        else
            for a=1:length(X);W{a}=immove3(X{a},x,y);end
        end
    else
        if j>0
            for a=1:length(X);W{a}=immove2(X{a},x,y);end
        else
            for a=1:length(X);W{a}=immove4(X{a},x,y);end
        end
    end
end

function Y=immove1(I,i,j)
    [m,n]=size(I);
    Y=false(m+j,n+i);
    Y(1:m,1:n)=I;
end
function Y=immove2(I,i,j)
    [m,n]=size(I);
    Y=false(m+j,n+i);
    Y(1:m,1+i:n+i)=I;
end
function Y=immove3(I,i,j)
    [m,n]=size(I);
    Y=false(m+j,n+i);
    Y(1+j:m+j,1:n)=I;
end
function Y=immove4(I,i,j)
    [m,n]=size(I);
    Y=false(m+j,n+i);
    Y(1+j:m+j,1+i:n+i)=I;
end