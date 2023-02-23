%Objective: define some parameters in the numerical experiment

%% some parameters for the kernel function, upper bound and grid search
PARAMETERS.type_kernel = {'rbf';'linear'};          % implemented: rbf, linear, polinomial
PARAMETERS.k_fold = 5;                              % how many folds?
PARAMETERS.scalling = 0;                            % 1: normal; 2: minMax; 0: nothing
PARAMETERS.min = 0;                                 % Parameter for scalling 2: minMax;
PARAMETERS.max = 0;                                 % Parameter for scalling 2: minMax
PARAMETERS.upperBounds = [0.1 1 10 100 1000 1e4];   % upper bounds for the optimization problem
PARAMETERS.timeLimit = 20*60;                       % runtime limit (seconds)
PARAMETERS.toleranceSV = 1e-6;
PARAMETERS.tolerance = 1e-4;
PARAMETERS.kmax = 100000;

RESULTS = struct('tableTestPhase',[]);
GRIDSEARCH = struct();
TESTPHASE = struct();

%% Criando pastas para armazenar resultados
% *** cria a pasta do resultado
% STR.caminho = [pwd '\_results\benchmarkDatasets_libsvmDefaultParameters_seed' num2str(seed)];
STR.caminho = [pwd '\_results\randomGeneratedDatasets_libsvmDefaultsParameters_seed' num2str(seed)];
mkdir(STR.caminho)

STR.caminho_dataset = 'G:\Meu Drive\PPGMNE\Doutorado\Tese\Algoritmos\_dataset'; % datasets archives folder

%%
header = {'dataset','method','kernel function','n_train','n_test','attributes', 'classProportionPos_train','classProportionNeg_train','classProportionPos_test','classProportionNeg_test',...
        'gammaKernel','upperBound','degree (poly kernel only)','r (poly kernel only)','iterations','constantDP','fval','supportVectors','exitflag','stopCrit1','stopCrit2','stopCrit3','stopCrit4',...
        'buildStructTime','trainingTime (loop only)','trainingTimeTotal','normLinearSystem1','normLinearSystem2','bStar','LagrangeMultiplier','accuracy','balacendAcc','precision',...
        'recall','f1Score','TP','TN','FP','FN','accuracy_fcLIBSVM','bstar_fcLIBSVM'};
methods = {'knapsack_SVM';'knapsack_projGrad';'projGrad_cauchy';'projGrad_newton';'SPG_exactSearch';'SPG_augmentedLagrangian';'SPG_augmentedLagrangian_rhoUpdate';'filter_quadprog';'filter_augmentedLagrangian';...
    'libsvm';'quadprog';'quadprog_reg';'liblinear';'SMO';'My_libsvm'};
nameDS = {};
columnMethods = {};
columnKernel = {};

%% Flags
STR.doKfold               = 1;
STR.runKnapsack_SVM       = 1;
STR.runKnapsack_ProjGrad  = 1;
STR.runProjGrad_cauchy    = 1;
STR.runProjGrad_newton    = 1;
STR.runSPG_exactSearch    = 1;
STR.runSPG_augLag         = 1;
STR.runSPG_augLag_rho     = 1;
STR.runFilter_qp          = 1;
STR.runFilter_AL          = 1;
STR.runLibsvm             = 1;
STR.runQuadprog           = 1;
STR.runQuadprog_reg       = 1;
STR.runLiblinear          = 1;
STR.runSMO                = 1;
STR.runMyLibsvm           = 1;



%% *** Datasets
STR.dataset = {'leukemia';'duke';'colon.cancer';'liver.disorders';'sonar';'heart';'heart.scale';...
    'breast.cancer';'breast.cancer.scale';'australian';'australian.scale';...
    'diabetes';'diabetes.scale';'fourclass';'fourclass.scale';'madelon';...
    'german.numer';'german.numer.scale';'splice';...
    'svmguide3';'svmguide1';'mushrooms';'a1a';'a2a';'a3a';'a4a';'a5a';'a6a';'w1a';'w2a';'w3a';'w4a';'w5a'};