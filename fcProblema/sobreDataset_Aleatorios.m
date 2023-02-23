function [DATASET,STR,X,y] = sobreDataset_Aleatorios(d,PARAMETROS,STR)
%% *** Carrega os conjuntos de dados
STR.datasetAtual = STR.dataset{d,:};
% fprintf('************ %s ******************\n',STR.datasetAtual)

% [y, X] = libsvmread([STR.caminho_dataset '\' STR.datasetAtual '.dat']);
% X = full(X); %tira o sparse do conjunto de dados
X = xlsread([STR.caminho_dataset '\_aleatorios\make_classification_p=2.xlsx'],STR.datasetAtual);
y = X(:,end);
X(:,end) = [];

% Verificao no vetor de respostas para ser -1 e 1
aux = unique(y);
aux_min = min(aux); 
aux1 = (y == aux_min);
y(aux1) = -1;
y(~aux1) = 1;

%% *** Aplicando alguma transformacao no conjunto de dados
switch PARAMETROS.scalling
    case {1;'gaussiano'}
        X = normalize(X);
        
    case {2;'minMax'}
        if ~isfield(PARAMETROS,'max'), lim_sup = 1;else,lim_sup=PARAMETROS.max;end
        if ~isfield(PARAMETROS,'min'), lim_inf = 0;else,lim_inf=PARAMETROS.min;end                        
        
        X_std = (X - min(X)) ./ (max(X) - min(X));
        X = X_std * (lim_sup - lim_inf) + lim_inf;
        
    case {3}
        X = X ./ max(X);
end

%% *** Validation set (80% treino, 20% teste)
n_row = size(X,1);
c = cvpartition(n_row,'Holdout',0.2); %
idxTrain = training(c);%ones(n_row,1);
DATASET.idx_treino = find(idxTrain == 1);
DATASET.idx_teste = find(idxTrain == 0);
clear idxTrain c n_row

DATASET.X_treino = X(DATASET.idx_treino,:); DATASET.y_treino = y(DATASET.idx_treino,1);
DATASET.X_teste = X(DATASET.idx_teste,:); DATASET.y_teste = y(DATASET.idx_teste,1);
DATASET.n_train = size(DATASET.idx_treino,1);
DATASET.p = size(X,2);

%% *** Retorna os indices de cada fold
[DATASET] = kFold_CV(DATASET.X_treino,DATASET.y_treino,PARAMETROS.k_fold,DATASET);
