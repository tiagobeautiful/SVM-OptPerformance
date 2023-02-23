% Objective: do a grid search with RBF KERNEL aiming to find the best training parameters

function [RESULTS,GRIDSEARCH] = trainingPhase_linear(d,kernel,DATASET,PARAMETERS,STR,RESULTS,GRIDSEARCH)

%counters
fold = 1; box = 1;

% *** making the grid search only for LIBSVM
[STR] = alteratingFlag('Libsvm',STR);

%% loop k-fold
for k = 1:PARAMETERS.k_fold %****** k-FOLD
    % *** training and validation sets split for the k-fold
    AUX.X_train = DATASET.X_train(DATASET.idx_train{k,1},:); AUX.y_train = DATASET.y_train(DATASET.idx_train{k,1},:);
    AUX.X_test  = DATASET.X_train(DATASET.idx_val{k,1},:); AUX.y_test = DATASET.y_train(DATASET.idx_val{k,1},:);
    [AUX.n_train,AUX.p_train] = size(AUX.X_train);
    
    % *** create matrix and others elements of the optimization problem
    PARAMETERS.gammaKernel = NaN;
    PARAMETERS = makingStructs(PARAMETERS,AUX);
    
    % *** Majoring the objective function
    PARAMETERS.majorValue = 1;%max(max(PARAMETERS.P));
    PARAMETERS.P = PARAMETERS.P / PARAMETERS.majorValue;
    PARAMETERS.a = PARAMETERS.a / PARAMETERS.majorValue;
    
    for c = PARAMETERS.upperBounds %****** upper bounds values
        % ########### creating the upper bound
        PARAMETERS.C = c;
        PARAMETERS.ub = sparse(PARAMETERS.C*ones(AUX.n_train,1));
        
        fprintf('***************************\n')
        fprintf('%s, %s, FOLD %d, C = %d\n',STR.datasetAtual,PARAMETERS.chosenKernel,k,PARAMETERS.C)
        fprintf('***************************\n')
        
        % *** Call methods
        [K_SVM{fold,box},...      %knapsack_Torrealba
            K_PG{fold,box},...       %knapsack_ProjectedGradient
            PG_C{fold,box},...       %projectedGradient_cauchy
            PG_N{fold,box},...       %projectedGradient_newton
            SPG_es{fold,box},...     %spg_exactSearch
            SPG_AugLag{fold,box},... %spg_augmentedLagr
            SPG_AugLag_rho{fold,box},... %spg_augmentedLagr_rhoUpdate           
            F_QP{fold,box},...       %filter_qp
            F_AL{fold,box},...       %filter_augmentedLagr
            LIB{fold,box},...        %libsvm
            QP{fold,box},...         %quadprog
            QP_reg{fold,box},...     %quadprog_reg
            LIBI{fold,box}, ...      %liblinear
            SMO{fold,box},...     %SMO
            MY_LIB{fold,box}] = ...    %MY_LIBSVM
            callMethods(PARAMETERS,AUX,STR);
        
        box = box+1;
        
        % *** cleaning for the next upper bound value
        PARAMETERS = rmfield(PARAMETERS,{'C';'ub'});
        clear c
    end
    box = 1;
    
    % *** clearning values in the PARAMETERS struct
    PARAMETERS = rmfield(PARAMETERS,...
        {'P';'gammaKernel';'kernelX';'n_train';...
        'p_train';'a';'b';'c';'lb';'buildStructTime'});
    
    clear n p
    
    %% *** Cleaning the validation matrix created
    AUX = rmfield(AUX,{'X_train';'y_train';'X_test';'y_test'});
    fold = fold+1;
end
clear k c fold box


%% *** analyzing the best values for the test phase
[RESULTS] = buildGrid_linear(d,kernel,K_SVM,K_PG,PG_C,PG_N,SPG_es,SPG_AugLag,SPG_AugLag_rho,F_QP,F_AL,LIB,QP,QP_reg,LIBI,SMO,MY_LIB,PARAMETERS,RESULTS,DATASET,STR);

GRIDSEARCH.K_SVM{d,kernel} = K_SVM;
GRIDSEARCH.K_PG{d,kernel} = K_PG;
GRIDSEARCH.PG_C{d,kernel} = PG_C;
GRIDSEARCH.PG_N{d,kernel} = PG_N;
GRIDSEARCH.SPG_es{d,kernel} = SPG_es;
GRIDSEARCH.SPG_AugLag{d,kernel} = SPG_AugLag;
GRIDSEARCH.SPG_AugLag_rho{d,kernel} = SPG_AugLag_rho;
GRIDSEARCH.F_QP{d,kernel} = F_QP;
GRIDSEARCH.F_AL{d,kernel} = F_AL;
GRIDSEARCH.LIB{d,kernel} = LIB;
GRIDSEARCH.QP{d,kernel} = QP;
GRIDSEARCH.QP_reg{d,kernel} = QP_reg;
GRIDSEARCH.LIBI{d,kernel} = LIBI;
GRIDSEARCH.SMO{d,kernel} = SMO;
GRIDSEARCH.MY_LIB{d,kernel} = MY_LIB;
