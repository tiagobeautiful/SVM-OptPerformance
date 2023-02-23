function [TESTPHASE,RESULTS] = testingPhase2(d,kernel,RESULTS,TESTPHASE,PARAMETERS,DATASET,STR)
[DATASET.n_train,DATASET.p_train] = size(DATASET.X_train);
[DATASET.n_test,DATASET.p_test] = size(DATASET.X_test);
fprintf('\t --> TEST PHASE: dataset %s, kernel %s \n',STR.datasetAtual,PARAMETERS.chosenKernel)

message = '';

if strcmp(PARAMETERS.chosenKernel,'rbf')
    PARAMETERS.gammaKernel = RESULTS.bestParameters{d,kernel}(10,1);% *** calcula a matriz kernel da fase de treino
elseif strcmp(PARAMETERS.chosenKernel,'linear')
    PARAMETERS.gammaKernel = RESULTS.bestParameters{d,kernel}(10,1);% *** calcula a matriz kernel da fase de treino
elseif strcmp(PARAMETERS.chosenKernel,'poly')
    PARAMETERS.gammaKernel = RESULTS.bestParameters{d,kernel}(10,1);% *** calcula a matriz kernel da fase de treino
    PARAMETERS.d = RESULTS.bestParameters{d,kernel}(10,5);
    PARAMETERS.r =RESULTS.bestParameters{d,kernel}(10,6);
end
PARAMETERS = makingStructs(PARAMETERS,DATASET);

% *** Majoring the objective function
PARAMETERS.majorValue = 1;%max(max(PARAMETERS.P));
PARAMETERS.P = PARAMETERS.P / PARAMETERS.majorValue;
PARAMETERS.a = PARAMETERS.a / PARAMETERS.majorValue;

% *** regularizing matrix P
PARAMETERS.Poriginal = PARAMETERS.P;
[PARAMETERS.P,PARAMETERS.constant] = verifyingPD(PARAMETERS.P,DATASET.n_train,PARAMETERS.timeLimit);


% *** cria o limite superior da caixa e configura as opcoes do libsvm
PARAMETERS.C = RESULTS.bestParameters{d,kernel}(10,2);
PARAMETERS.ub = PARAMETERS.C*ones(DATASET.n_train,1);


%% Augmented Lagragian based
addpath([pwd '\augLag'])
try
    [STR] = alteratingFlag('Knapsack_SVM',STR);
    [K_SVM,~,~,~,~,~,~,~,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);
    
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME
    
    K_SVM = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %knapsack_SVM

try
    message = '';

    [STR] = alteratingFlag('Knapsack_ProjGrad',STR);
    [~,K_PG,~,~,~,~,~,~,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME    
    K_PG = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %knapsack_projGrad

rmpath([pwd '\augLag'])

%% Projected gradient (Cauchy or Newton)
addpath([pwd '\gradProj'])
try
    message = '';

    [STR] = alteratingFlag('ProjGrad_cauchy',STR);
    [~,~,PG_C,~,~,~,~,~,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);
    
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    PG_C = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %ProjGrad_cauchy

try
    message = '';

    [STR] = alteratingFlag('ProjGrad_newton',STR);
    [~,~,~,PG_N,~,~,~,~,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    PG_N = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %ProjGrad_newton

rmpath([pwd '\gradProj'])

%% Spectral Projected Gradient
addpath([pwd '\spg'])
try
    message = '';

    [STR] = alteratingFlag('SPG_exactSearch',STR);
    [~,~,~,~,SPG_es,~,~,~,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    SPG_es = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %SPG_exactSearch

try
    message = '';

    [STR] = alteratingFlag('SPG_augLag',STR);
    [~,~,~,~,~,SPG_AugLag,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    SPG_AugLag = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %SPG_augmentedLagrangian

try
    message = '';
    
    [STR] = alteratingFlag('SPG_augLag_rho',STR);
    [~,~,~,~,~,~,SPG_AugLag_rho,~,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    SPG_AugLag_rho = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %SPG_augmentedLagrangian_rhoUpdate
rmpath([pwd '\spg'])

%% Filter
addpath([pwd '\filtro'])
try
    message = '';

    [STR] = alteratingFlag('Filter_qp',STR);
    [~,~,~,~,~,~,~,F_QP,~,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);
    
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    F_QP = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %Filter_quadprog

try
    message = '';
    
    [STR] = alteratingFlag('Filter_AL',STR);
    [~,~,~,~,~,~,~,~,F_AL,~,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    F_AL = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %Filter_augmentedLagrangian

rmpath([pwd '\filtro'])

%% LIBSVM
addpath(genpath([pwd '\libsvm-3.3']))
try
    message = '';

    [STR] = alteratingFlag('Libsvm',STR);
    [~,~,~,~,~,~,~,~,~,LIB,~,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    LIB = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end%libsvm

rmpath(genpath([pwd '\libsvm-3.3']))

%% quadprog
try
    message = '';

    [STR] = alteratingFlag('Quadprog',STR);
    [~,~,~,~,~,~,~,~,~,~,QP,~,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    QP = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end%quadprog

try
    message = '';

    [STR] = alteratingFlag('Quadprog_reg',STR);
    [~,~,~,~,~,~,~,~,~,~,~,QP_reg,~,~,~] = ...
        callMethods(PARAMETERS,DATASET,STR);
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    QP = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end%quadprog_reg

%% liblinear
addpath(genpath([pwd '\liblinear-2.45']))
try
    message = '';

    if strcmp(PARAMETERS.chosenKernel,'linear') 
        [STR] = alteratingFlag('Liblinear',STR);
        [~,~,~,~,~,~,~,~,~,~,~,~,LIBLI,~,~] = ...
            callMethods(PARAMETERS,DATASET,STR);
    else, error('vai pro catch')
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    LIBLI = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN,...
        'accuracy_fcLib',NaN,'bStar_fcLib',NaN);
end%liblinear

rmpath(genpath([pwd '\liblinear-2.45']))

%% SMO and My LIBSVM
addpath([pwd '\decomnposicao'])
try
    message = '';

    [STR] = alteratingFlag('SMO',STR);
    [~,~,~,~,~,~,~,~,~,~,~,~,~,SMO,~] = ...
        callMethods(PARAMETERS,DATASET,STR);
    
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    SMO = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %SMO

try
    message = '';
    
    [STR] = alteratingFlag('My_LIBSVM',STR);
    [~,~,~,~,~,~,~,~,~,~,~,~,~,~,MY_LIB] = ...
        callMethods(PARAMETERS,DATASET,STR);

catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME   
    
    MY_LIB = struct('bStar',NaN,'alpha',NaN*ones(DATASET.n_train,1),'index',NaN*ones(DATASET.n_train,1), ...
        'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
        'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(DATASET.n_test,1),...
        'valores_decisao',NaN*ones(DATASET.n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %MYLIB

rmpath([pwd '\decomnposicao'])
fprintf('\n');

%% *** analyzing the best values for the test phase
%build excel test results
classProportionPos_train = 100*size(find(DATASET.y_train == 1),1)/size(DATASET.y_train,1);
classProportionNeg_train = 100*size(find(DATASET.y_train == -1),1)/size(DATASET.y_train,1);
classProportionPos_test = 100*size(find(DATASET.y_test == 1),1)/size(DATASET.y_test,1);
classProportionNeg_test = 100*size(find(DATASET.y_test == -1),1)/size(DATASET.y_test,1);

RESULTS.tableTestPhase = [RESULTS.tableTestPhase;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(1,1) RESULTS.bestParameters{d,kernel}(1,2) RESULTS.bestParameters{d,kernel}(1,5:6) ...
          K_SVM.outputs      sum(K_SVM.outputs(10:11))      K_SVM.norm_Sistemas      K_SVM.bStar      K_SVM.lambda      K_SVM.accuracy      K_SVM.balancedAcc      K_SVM.precision      K_SVM.recall      K_SVM.f1_score      K_SVM.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(2,1) RESULTS.bestParameters{d,kernel}(2,2) RESULTS.bestParameters{d,kernel}(2,5:6) ...
          K_PG.outputs       sum(K_PG.outputs(10:11))       K_PG.norm_Sistemas       K_PG.bStar       K_PG.lambda       K_PG.accuracy       K_PG.balancedAcc       K_PG.precision       K_PG.recall       K_PG.f1_score       K_PG.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(3,1) RESULTS.bestParameters{d,kernel}(3,2) RESULTS.bestParameters{d,kernel}(3,5:6) ...
          PG_C.outputs       sum(PG_C.outputs(10:11))       PG_C.norm_Sistemas       PG_C.bStar       NaN               PG_C.accuracy       PG_C.balancedAcc       PG_C.precision       PG_C.recall       PG_C.f1_score       PG_C.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(4,1) RESULTS.bestParameters{d,kernel}(4,2) RESULTS.bestParameters{d,kernel}(4,5:6) ...
          PG_N.outputs       sum(PG_N.outputs(10:11))       PG_N.norm_Sistemas       PG_N.bStar       NaN               PG_N.accuracy       PG_N.balancedAcc       PG_N.precision       PG_N.recall       PG_N.f1_score       PG_N.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(5,1) RESULTS.bestParameters{d,kernel}(5,2) RESULTS.bestParameters{d,kernel}(5,5:6) ...
          SPG_es.outputs     sum(SPG_es.outputs(10:11))     SPG_es.norm_Sistemas     SPG_es.bStar     NaN               SPG_es.accuracy     SPG_es.balancedAcc     SPG_es.precision     SPG_es.recall     SPG_es.f1_score     SPG_es.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(6,1) RESULTS.bestParameters{d,kernel}(6,2) RESULTS.bestParameters{d,kernel}(6,5:6) ...
          SPG_AugLag.outputs sum(SPG_AugLag.outputs(10:11)) SPG_AugLag.norm_Sistemas SPG_AugLag.bStar SPG_AugLag.lambda SPG_AugLag.accuracy SPG_AugLag.balancedAcc SPG_AugLag.precision SPG_AugLag.recall SPG_AugLag.f1_score SPG_AugLag.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(7,1) RESULTS.bestParameters{d,kernel}(7,2) RESULTS.bestParameters{d,kernel}(7,5:6) ...
          SPG_AugLag_rho.outputs sum(SPG_AugLag_rho.outputs(10:11)) SPG_AugLag_rho.norm_Sistemas SPG_AugLag_rho.bStar SPG_AugLag_rho.lambda SPG_AugLag_rho.accuracy SPG_AugLag_rho.balancedAcc SPG_AugLag_rho.precision SPG_AugLag_rho.recall SPG_AugLag_rho.f1_score SPG_AugLag_rho.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(8,1) RESULTS.bestParameters{d,kernel}(8,2) RESULTS.bestParameters{d,kernel}(8,5:6) ...
          F_QP.outputs       sum(F_QP.outputs(10:11))       F_QP.norm_Sistemas       F_QP.bStar       F_QP.lambda       F_QP.accuracy       F_QP.balancedAcc       F_QP.precision       F_QP.recall       F_QP.f1_score       F_QP.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(9,1) RESULTS.bestParameters{d,kernel}(9,2) RESULTS.bestParameters{d,kernel}(9,5:6) ...
          F_AL.outputs       sum(F_AL.outputs(10:11))       F_AL.norm_Sistemas       F_AL.bStar       F_AL.lambda       F_AL.accuracy       F_AL.balancedAcc       F_AL.precision       F_AL.recall       F_AL.f1_score       F_AL.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(10,1) RESULTS.bestParameters{d,kernel}(10,2) RESULTS.bestParameters{d,kernel}(10,5:6) ...
          LIB.outputs        LIB.outputs(11)                LIB.norm_Sistemas        LIB.bStar        NaN               LIB.accuracy        LIB.balancedAcc        LIB.precision        LIB.recall        LIB.f1_score        LIB.ConfMatr      LIB.acc_fcLIB     LIB.bStar_fcLib  ;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(11,1) RESULTS.bestParameters{d,kernel}(11,2) RESULTS.bestParameters{d,kernel}(11,5:6) ... 
          QP.outputs         sum(QP.outputs(10:11))         QP.norm_Sistemas         QP.bStar         QP.lambda         QP.accuracy         QP.balancedAcc         QP.precision         QP.recall         QP.f1_score         QP.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(12,1) RESULTS.bestParameters{d,kernel}(12,2) RESULTS.bestParameters{d,kernel}(12,5:6) ... 
          QP_reg.outputs     sum(QP_reg.outputs(10:11))     QP_reg.norm_Sistemas     QP_reg.bStar     QP_reg.lambda     QP_reg.accuracy     QP_reg.balancedAcc     QP_reg.precision     QP_reg.recall     QP_reg.f1_score     QP_reg.ConfMatr      NaN       NaN;...
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(13,1) RESULTS.bestParameters{d,kernel}(13,2) RESULTS.bestParameters{d,kernel}(13,5:6) ... 
          LIBLI.outputs      LIBLI.outputs(11)      LIBLI.norm_Sistemas      LIBLI.bStar      NaN               LIBLI.accuracy      LIBLI.balancedAcc      LIBLI.precision      LIBLI.recall      LIBLI.f1_score      LIBLI.ConfMatr      LIBLI.accuracy_fcLib       LIBLI.bStar_fcLib; ...        
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(14,1) RESULTS.bestParameters{d,kernel}(14,2) RESULTS.bestParameters{d,kernel}(14,5:6) ... 
          SMO.outputs         sum(SMO.outputs(10:11))         SMO.norm_Sistemas         SMO.bStar         SMO.lambda         SMO.accuracy         SMO.balancedAcc         SMO.precision         SMO.recall         SMO.f1_score         SMO.ConfMatr      NaN       NaN;...        
        kernel DATASET.n_train DATASET.n_test DATASET.p_train classProportionPos_train classProportionNeg_train classProportionPos_test classProportionNeg_test RESULTS.bestParameters{d,kernel}(15,1) RESULTS.bestParameters{d,kernel}(15,2) RESULTS.bestParameters{d,kernel}(15,5:6) ... 
          MY_LIB.outputs         sum(MY_LIB.outputs(10:11))         MY_LIB.norm_Sistemas         MY_LIB.bStar         MY_LIB.lambda         MY_LIB.accuracy         MY_LIB.balancedAcc         MY_LIB.precision         MY_LIB.recall         MY_LIB.f1_score         MY_LIB.ConfMatr      NaN       NaN ...
    ];



RESULTS.tableTestPhaseError{d,kernel} = {...
        'knapsack_svm',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(1,1:2) RESULTS.bestParameters{d,kernel}(1,5:6)],K_SVM.Error;...
        'knapsack_projGrad',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(2,1:2) RESULTS.bestParameters{d,kernel}(2,5:6)],K_PG.Error;...
        'projGrad_cauchy',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(3,1:2) RESULTS.bestParameters{d,kernel}(3,5:6)],PG_C.Error;...
        'projGrad_newton',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(4,1:2) RESULTS.bestParameters{d,kernel}(4,5:6)],PG_N.Error;...
        'spg_exactSearch',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(5,1:2) RESULTS.bestParameters{d,kernel}(5,5:6)],SPG_es.Error;...
        'spg_augLag',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(6,1:2) RESULTS.bestParameters{d,kernel}(6,5:6)],SPG_AugLag.Error;...
        'spg_augLag_rho',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(7,1:2) RESULTS.bestParameters{d,kernel}(7,5:6)],SPG_AugLag_rho.Error;...
        'filter_quadprog',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(8,1:2) RESULTS.bestParameters{d,kernel}(8,5:6)],F_QP.Error;...
        'filter_augLag',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(9,1:2) RESULTS.bestParameters{d,kernel}(9,5:6)],F_AL.Error;...
        'libsvm',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(10,1:2) RESULTS.bestParameters{d,kernel}(10,5:6)],LIB.Error;...
        'quadprog',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(11,1:2) RESULTS.bestParameters{d,kernel}(11,5:6)],QP.Error;...
        'quadprog_reg',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(12,1:2) RESULTS.bestParameters{d,kernel}(12,5:6)],QP_reg.Error;...
        'liblinear',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(13,1:2) RESULTS.bestParameters{d,kernel}(13,5:6)],LIBLI.Error;...
        'SMO',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(14,1:2) RESULTS.bestParameters{d,kernel}(14,5:6)],LIBLI.Error;... 
        'My_LIBSVM',[DATASET.n_train,DATASET.n_test DATASET.p_train RESULTS.bestParameters{d,kernel}(15,1:2) RESULTS.bestParameters{d,kernel}(15,5:6)],LIBLI.Error;... 
        };

TESTPHASE.K_SVM{d,kernel} = K_SVM;
TESTPHASE.K_PG{d,kernel} = K_PG;
TESTPHASE.PG_C{d,kernel} = PG_C;
TESTPHASE.PG_N{d,kernel} = PG_N;
TESTPHASE.SPG_es{d,kernel} = SPG_es;
TESTPHASE.SPG_AugLag{d,kernel} = SPG_AugLag;
TESTPHASE.SPG_AugLag_rho{d,kernel} = SPG_AugLag_rho;
TESTPHASE.F_QP{d,kernel} = F_QP;
TESTPHASE.F_AL{d,kernel} = F_AL;
TESTPHASE.LIB{d,kernel} = LIB;
TESTPHASE.QP{d,kernel} = QP;
TESTPHASE.QP_reg{d,kernel} = QP_reg;
TESTPHASE.LIBI{d,kernel} = LIBLI;
TESTPHASE.SMO{d,kernel} = SMO;
TESTPHASE.MY_LIB{d,kernel} = MY_LIB;


end