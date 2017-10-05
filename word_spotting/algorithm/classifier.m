function [y, XD]=classifier(X,model,sParam)

    XD=get_feature_descriptor(X, sParam,model.P);
    R=model.H*XD';          
%     clear XD;
%     XD = [];
    [m,n] = size(R);
    F=reshape(R,[model.n,m/model.n,n]);       clear R;
    %max-pooling
    v=reshape(max(F,[],1),[],n);                              clear F;
    %vector normalization
    y=nrm(get_max_pooling(v,model.Y));
    
end


