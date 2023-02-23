function [DATASET,STR] = readDatasetLIBSVM(d,PARAMETERS,STR)
addpath(genpath([pwd '\libsvm-3.3']))

%% *** loading datasets
STR.datasetAtual = STR.dataset{d,:};
fprintf('***************************************************\n')
fprintf('************ %s ******************\n',STR.datasetAtual)
fprintf('***************************************************\n')

[y, X] = libsvmread([STR.caminho_dataset '\' STR.datasetAtual '.dat']);
[y_test, X_test] = libsvmread([STR.caminho_dataset '\' STR.datasetAtual '.t.dat']);

rmpath(genpath([pwd '\libsvm-3.3']))
[n_train,~] = size(X);
[n_test,~] = size(X_test);

%%% *** fixing bugs with datasets aXa.dat
if strcmp(STR.datasetAtual(1,1),'a') == 1
    [n_train,p_train] = size(X);
    [~,p_test] = size(X_test);
    
    diferenca = p_test - p_train;
    bloco = zeros(n_train,diferenca);
    
    X = [X bloco];
end

% Verifying the label array to mantain -1 e 1
aux = unique(y);
aux_min = min(aux); 
aux1 = (y == aux_min);
y(aux1) = -1;
y(~aux1) = 1;

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
else
    X_train = X;
    y_train = y;
    DATASET.idxTrain = [1:n_train]';
    DATASET.idxTest = [1:n_test]';
end

% allocating
DATASET.X_train = X_train; DATASET.y_train = y_train;
DATASET.X_test = X_test; DATASET.y_test = y_test;   
[DATASET.n,DATASET.p] = size(X);

if STR.doKfold && PARAMETERS.k_fold > 1
    [DATASET] = kFold_CV(DATASET.X_train,PARAMETERS.k_fold,DATASET);
end