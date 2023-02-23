function [GERAL,model] = callLIBLINEAR(PARAMETERS,DATASET)

trainData_X = sparse(DATASET.X_train);
trainData_y = sparse(DATASET.y_train);


% *** Building options
PARAMETERS.options_liblinear = ['-q -s 3 -c ' num2str(PARAMETERS.C)];    


% *** Training
startTime = tic();
model = train(trainData_y, trainData_X, PARAMETERS.options_liblinear);
trainingTime = toc(startTime);

% *** test
[predicao, accuracy, fitted] = ...
            predict(sparse(DATASET.y_test), sparse(DATASET.X_test), model);
        
GERAL.accuracy_fcLib = accuracy(1);
GERAL.bStar_fcLib = model.bias;
GERAL.balancedAcc = NaN;
GERAL.precision = NaN;
GERAL.recall = NaN;
GERAL.f1_score = NaN;
GERAL.ConfMatr = [NaN NaN NaN NaN];
GERAL.predicao = predicao;
GERAL.valores_decisao = fitted;

GERAL.accuracy = NaN;
GERAL.bStar = NaN;


GERAL.alpha = model.w;
GERAL.index = NaN;
GERAL.outputs = [NaN,NaN,NaN,NaN,1,NaN,NaN,NaN,NaN,NaN,trainingTime];
GERAL.norm_Sistemas = [NaN NaN];

