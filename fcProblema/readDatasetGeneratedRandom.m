function [DATASET,STR] = readDatasetGeneratedRandom(d,PARAMETERS,STR)

%% *** loading datasets
STR.datasetAtual = ['Problem_' num2str(d)];
fprintf('***************************************************\n')
fprintf('************ %s ******************\n',STR.datasetAtual)
fprintf('***************************************************\n')


X = xlsread([STR.caminho_dataset '\_aleatorios\' STR.datasetAtual '.xlsx']);
y = X(:,end);
X(:,end) = [];
y_test = [];
X_test = [];

%% *** Applying a scalling, if requested
switch PARAMETERS.scalling
    case {1;'gaussian'}
        X = normalize(X);
        
    case {2;'minMax'}
        if ~isfield(PARAMETERS,'max'), lim_sup = 1;else,lim_sup=PARAMETERS.max;end
        if ~isfield(PARAMETERS,'min'), lim_inf = 0;else,lim_inf=PARAMETERS.min;end                        
        
        X_std = (X - min(X)) ./ (max(X) - min(X));
        X = X_std * (lim_sup - lim_inf) + lim_inf;
end

%% *** Organizing the struct DATASET
DATASET = [];

if isempty(X_test)
    fprintf('\t ### MAKING A TRAIN/TEST SPLIT (80/20)\n');
    %choosing randomly the testing samples
    n_row = size(X,1);
    c = cvpartition(n_row,'Holdout',0.2);
    idxTrain = training(c);
    DATASET.idxTrain = find(idxTrain == 1);
    DATASET.idxTest = find(idxTrain == 0);
    clear idxTrain c n_row
    
    %creating the arrays for the testing samples
    X_test = X(DATASET.idxTest,:);
    y_test = y(DATASET.idxTest,1);
    
    %deleting the test samples from the training matrix
    X_train = X(DATASET.idxTrain,:);
    y_train = y(DATASET.idxTrain,1);   
end

% allocating
DATASET.X_train = X_train; DATASET.y_train = y_train;
DATASET.X_test = X_test; DATASET.y_test = y_test;   
[DATASET.n,DATASET.p] = size(X);

if STR.doKfold && PARAMETERS.k_fold > 1
    [DATASET] = kFold_CV(DATASET.X_train,PARAMETERS.k_fold,DATASET);
end