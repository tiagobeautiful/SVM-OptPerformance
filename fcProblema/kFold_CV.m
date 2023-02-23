function [DATASET] = kFold_CV(X,k_fold,DATASET)
%% K-Fold Cross Validation
n_row = size(X,1);

% *** encontrando os indices ALEATORIO
c = cvpartition(n_row,'KFold',k_fold);

for k = 1:k_fold
    
    logi_val = test(c,k); %indices logicos de teste
    logi_train = (logi_val == 0);
    
    DATASET.idx_train{k,1} = find(logi_train == 1);
    DATASET.idx_val{k,1} = find(logi_val == 1);
    
end

%DATASET.X = X; DATASET.y = y;
