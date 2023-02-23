%Objective: call all the implemented methods
% k-> actual fold
function [K_SVM,K_PG,...
          PG_C,PG_N, ...
          SPG_es,SPG_AugLag,SPG_AugLag_rho,...
          F_QP,F_AL,...
          LIB,QP,QP_reg,LIBLI,...
          SMO,MY_LIB] = callMethods(PARAMETERS,DATASET,STR)

% exitflags:
%      0 -> surpass time limit
%      1 -> converged
%      2 -> surpass max iteration,
%      3 -> SQP subproblem did not converge (Filter only)
%      4 -> SQP subproblem surpass time limit

n_train=DATASET.n_train;
n_test = size(DATASET.X_test,1);

%% Augmented Lagragian based
addpath([pwd '\augLag'])
try
    message = '';
    if STR.runKnapsack_SVM
        [K_SVM] = knapsack_SVM(PARAMETERS);
        
        if isnan(K_SVM.alpha), error('Solucao deu NaN');
        else,[K_SVM] = classificationTask(K_SVM,DATASET,PARAMETERS);
        fprintf('Knapsack_svm OK; ');
        K_SVM.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    K_SVM = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %knapsack_SVM

try
    if STR.runKnapsack_ProjGrad
        message = '';
        [K_PG] = knapsack_ProjGrad(PARAMETERS);
        if isnan(K_PG.alpha), error('Solucao deu NaN');
        else,[K_PG] = classificationTask(K_PG,DATASET,PARAMETERS);
        fprintf('Knapsack_ProjGrad OK; ');
        K_PG.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    K_PG = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %knapsack_projGrad

rmpath([pwd '\augLag'])

%% Projected gradient (Cauchy or Newton)
addpath([pwd '\gradProj'])
try
    if STR.runProjGrad_cauchy
        message = '';
        [PG_C] = ProjGrad_cauchy(PARAMETERS);
        if isnan(PG_C.alpha), error('Solucao deu NaN');
        else,[PG_C] = classificationTask(PG_C,DATASET,PARAMETERS);
        fprintf('ProjGrad_cauchy OK; ');
        PG_C.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    PG_C = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %ProjGrad_cauchy

try
    if STR.runProjGrad_newton
        message = '';
        [PG_N] = ProjGrad_newton(PARAMETERS);
        if isnan(PG_N.alpha), error('Solucao deu NaN');
        else,[PG_N] = classificationTask(PG_N,DATASET,PARAMETERS);
        fprintf('ProjGrad_newton OK; ');
        PG_N.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    PG_N = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %ProjGrad_newton

rmpath([pwd '\gradProj'])

%% Spectral Projected Gradient
addpath([pwd '\spg'])
try
    if STR.runSPG_exactSearch
        message = '';
        [SPG_es] = SPG(PARAMETERS);
        if isnan(SPG_es.alpha), error('Solucao deu NaN');
        else,[SPG_es] = classificationTask(SPG_es,DATASET,PARAMETERS);
        fprintf('SPG_exactSearch OK; ');
        SPG_es.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    SPG_es = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %SPG_exactSearch

try
    if STR.runSPG_augLag
        PARAMETERS.atualizaRho = 0;
        message = '';
        [SPG_AugLag] = SPG_AL(PARAMETERS);
        if isnan(SPG_AugLag.alpha), error('Solucao deu NaN');
        else,[SPG_AugLag] = classificationTask(SPG_AugLag,DATASET,PARAMETERS);
        fprintf('SPG_AugLagrangian OK; ');
        SPG_AugLag.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    SPG_AugLag = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %SPG_augmentedLagrangian

try
    if STR.runSPG_augLag_rho
        message = '';
        [SPG_AugLag_rho] = SPG_AL_rho(PARAMETERS);
        
        if isnan(SPG_AugLag_rho.alpha), error('Solucao deu NaN');
        else,[SPG_AugLag_rho] = classificationTask(SPG_AugLag_rho,DATASET,PARAMETERS);
        fprintf('SPG_AugLagrangian_rhoUpdate OK; ');
        SPG_AugLag_rho.Error = message;
        end
        
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    SPG_AugLag_rho = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %SPG_augmentedLagrangian_rhoPenaltyUpdate

rmpath([pwd '\spg'])

%% Filter
addpath([pwd '\filtro'])
try
    if STR.runFilter_qp
        message = '';
        [F_QP] = filtro(PARAMETERS,zeros(DATASET.n_train,1),1);
        if isnan(F_QP.alpha), error('Solucao deu NaN');
        else,[F_QP] = classificationTask(F_QP,DATASET,PARAMETERS);
        fprintf('Filter_quadprog OK; ');
        F_QP.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    F_QP = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %Filter_quadprog

try
    if STR.runFilter_AL
        message = '';
        [F_AL] = filtro(PARAMETERS,zeros(DATASET.n_train,1),0);
        if isnan(F_AL.alpha), error('Solucao deu NaN');
        else,[F_AL] = classificationTask(F_AL,DATASET,PARAMETERS);
        fprintf('Filter_augLagrangian OK; ');
        F_AL.Error = message;
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    F_AL = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %Filter_augmentedLagrangian

rmpath([pwd '\filtro'])

%% LIBSVM
addpath(genpath([pwd '\libsvm-3.3']))
try
    if STR.runLibsvm
        message = '';
        [LIB,model] = callLIBSVM(PARAMETERS,DATASET);
        if isnan(LIB.alpha), error('Solucao deu NaN');
        else,[LIB] = classificationTask(LIB,DATASET,PARAMETERS);
        fprintf('libsvm OK; ');
        LIB.Error = message;

        % *** Fase de teste (funcao libsvm)
        [~, accuracy, ~] = ...
            svmpredict(DATASET.y_test, DATASET.X_test, model);
        LIB.acc_fcLIB = accuracy(1);
        LIB.bStar_fcLib = -model.rho;
        clear accuracy model
        end
    else, error('vai pro catch');
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    LIB = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'acc_fcLIB',NaN,'bStar_fcLib',NaN,'lambda',NaN);
end%libsvm

rmpath(genpath([pwd '\libsvm-3.3']))

%% quadprog
try
    if STR.runQuadprog
        message = '';
        [QP] = callQuadprog(PARAMETERS,0);
        QP.outputs(2) = NaN;
        if isnan(QP.alpha), error('Solucao deu NaN');
        else,[QP] = classificationTask(QP,DATASET,PARAMETERS);
        fprintf('Quadprog OK; ');
        QP.Error = message;        
        end
    else, error('vai pro catch')
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    QP = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end%quadprog

try
    if STR.runQuadprog_reg       
        message = '';
        [QP_reg] = callQuadprog(PARAMETERS,1);
        QP_reg.outputs(2) = PARAMETERS.constant;
        if isnan(QP_reg.alpha), error('Solucao deu NaN');
        else,[QP_reg] = classificationTask(QP_reg,DATASET,PARAMETERS);
        fprintf('Quadprog_reg OK; ');
        QP_reg.Error = message;
        end
    else, error('vai pro catch')
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
            
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    QP_reg = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end%quadprog_reg


%% liblinear
% For L2-regularized L1-loss SVC dual (-s 3), we solve
% min_alpha  0.5(alpha^T Q alpha) - e^T alpha
%     s.t.   0 <= alpha_i <= C,

addpath(genpath([pwd '\liblinear-2.45']))
try
    if strcmp(PARAMETERS.chosenKernel,'linear') && STR.runLiblinear
        message = '';
        [LIBLI,~] = callLIBLINEAR(PARAMETERS,DATASET);
        fprintf('Liblinear OK; ');
        LIBLI.Error = message;    
    else, error('vai pro catch')
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    LIBLI = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN,...
            'accuracy_fcLib',NaN,'bStar_fcLib',NaN);
end%liblinear

rmpath(genpath([pwd '\liblinear-2.45']))
fprintf('\n');

%% SMO and LIBSVM (my implementation)
addpath([pwd '\decomposicao'])
try
     if STR.runSMO
        message = '';
        [SMO] = SMO_decomposition(DATASET,PARAMETERS);
        if isnan(SMO.alpha), error('Solucao deu NaN');
        else,[SMO] = classificationTask(SMO,DATASET,PARAMETERS);
        fprintf('SMO OK; ');
        SMO.Error = message;        
        end
     else, error('vai pro catch')
     end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    SMO = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %smo

try
    if STR.runMyLibsvm
        message = '';
        [MY_LIB] = LIBSVM_decomposition(DATASET,PARAMETERS);
        if isnan(MY_LIB.alpha), error('Solucao deu NaN');
        else,[MY_LIB] = classificationTask(MY_LIB,DATASET,PARAMETERS);
        fprintf('MY_LIB OK; ');
        MY_LIB.Error = message;        
        end
     else, error('vai pro catch')
    end
catch ME
    names = []; lines = [];
    for idx = [size(ME.stack,1):1]
        names = [names ' -> ' ME.stack(idx).name];
        lines = [lines ' -> ' num2str(ME.stack(idx).line)];
    end
    
    message = [ME.message ' ; Files: ' names ' ; Lines: ' lines]; clear ME       
    MY_LIB = struct('bStar',NaN,'alpha',NaN*ones(n_train,1),'index',NaN*ones(n_train,1), ...
            'outputs',NaN*ones(1,11), 'accuracy',NaN, 'balancedAcc',NaN, 'precision',NaN,...
            'recall',NaN, 'f1_score', NaN, 'ConfMatr',NaN*ones(1,4), 'predicao', NaN*ones(n_test,1),...
            'valores_decisao',NaN*ones(n_test,1),'Error',message,'norm_Sistemas',[NaN NaN],'lambda',NaN);
end %MY_LIBSVM

rmpath([pwd '\decomposicao'])