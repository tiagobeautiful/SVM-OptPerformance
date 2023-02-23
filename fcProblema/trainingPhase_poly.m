% Objective: do a grid search with RBF KERNEL aiming to find the best training parameters

function [RESULTS,GRIDSEARCH] = trainingPhase_poly(d,kernel,DATASET,PARAMETERS,STR,RESULTS,GRIDSEARCH)

%counters
fold = 1; gmKrn = 1; box = 1; grau = 1; valorR = 1;
PARAMETERS.majorValue = 1;

% *** making the grid search only for LIBSVM
[STR] = alteratingFlag('Libsvm',STR);

%% loop k-fold
for k = 1:PARAMETERS.k_fold %****** k-FOLD
    % *** training and validation sets split for the k-fold
    AUX.X_train = DATASET.X_train(DATASET.idx_train{k,1},:); AUX.y_train = DATASET.y_train(DATASET.idx_train{k,1},:);
    AUX.X_test  = DATASET.X_train(DATASET.idx_val{k,1},:); AUX.y_test = DATASET.y_train(DATASET.idx_val{k,1},:);
    [AUX.n_train,AUX.p_train] = size(AUX.X_train);
    
    % ########### Polynomial kernel parameters: ( gammaKernel<xi,xj> +r )^d
    PARAMETERS.valores_d = [2 3];
    PARAMETERS.valores_r = [0 0.5 1];
    PARAMETERS.valoresGmKrnl = [1 0.25 1/DATASET.p];
    
    for grauD = PARAMETERS.valores_d
        PARAMETERS.d = grauD;
        
        for deslocR=PARAMETERS.valores_r
            PARAMETERS.r = deslocR;
            
            for gammaKernel = PARAMETERS.valoresGmKrnl %****** GAMMA_KERNEL
                % *** create matrix and others elements of the optimization problem
                PARAMETERS.gammaKernel = gammaKernel;
                PARAMETERS = makingStructs(PARAMETERS,AUX);

                % *** Majoring the objective function
                PARAMETERS.majorValue = 1;%1e3*max(max(PARAMETERS.P));
                PARAMETERS.P = PARAMETERS.P / PARAMETERS.majorValue;
                PARAMETERS.a = PARAMETERS.a / PARAMETERS.majorValue;

                for c = PARAMETERS.upperBounds %****** upper bounds values
                    % ########### creating the upper bound
                    PARAMETERS.C = c;
                    PARAMETERS.ub = sparse(PARAMETERS.C*ones(AUX.n_train,1));

                    fprintf('***************************\n')
                    fprintf('%s, %s, FOLD %d, C = %d, gamma_kernel = %d, d= %.2f, r = %.2f\n',STR.datasetAtual,PARAMETERS.chosenKernel,k,PARAMETERS.C,PARAMETERS.gammaKernel,PARAMETERS.d,PARAMETERS.r)
                    fprintf('***************************\n')

                    % *** Call methods
                    [K_SVM{fold,box,grau,valorR,gmKrn},...      %knapsack_Torrealba
                     K_PG{fold,box,grau,valorR,gmKrn},...       %knapsack_ProjectedGradient
                     PG_C{fold,box,grau,valorR,gmKrn},...       %projectedGradient_cauchy
                     PG_N{fold,box,grau,valorR,gmKrn},...       %projectedGradient_newton
                     SPG_es{fold,box,grau,valorR,gmKrn},...     %spg_exactSearch
                     SPG_AugLag{fold,box,grau,valorR,gmKrn},... %spg_augmentedLagr
                     SPG_AugLag_rho{fold,box,grau,valorR,gmKrn},... %spg_augmentedLagr_rhoUpdate
                     F_QP{fold,box,grau,valorR,gmKrn},...       %filter_qp
                     F_AL{fold,box,grau,valorR,gmKrn},...       %filter_augmentedLagr
                     LIB{fold,box,grau,valorR,gmKrn},...        %libsvm
                     QP{fold,box,grau,valorR,gmKrn},...         %quadprog
                     QP_reg{fold,box,grau,valorR,gmKrn},...         %quadprog regularized
                     LIBI{fold,box,grau,valorR,gmKrn}, ...      %liblinear
                     SMO{fold,box,gmKrn},...     %SMO
                     MY_LIB{fold,box,gmKrn}] = ...    %MY_LIBSVM
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

                gmKrn = gmKrn + 1;
            end
            gmKrn = 1;
            valorR = valorR +1;
            PARAMETERS = rmfield(PARAMETERS,{'r'});

        end
        valorR = 1;
        grau = grau + 1;
        PARAMETERS = rmfield(PARAMETERS,{'d'});
        
    end
    grau = 1;
    
    %% *** Cleaning the validation matrix created
    AUX = rmfield(AUX,{'X_train';'y_train';'X_test';'y_test'});
    fold = fold+1;
end
clear k gammaKernel c cont_c cont_gmKrn cont_k


%% *** analyzing the best values for the test phase
[RESULTS] = buildGrid_poly(d,kernel,K_SVM,K_PG,PG_C,PG_N,SPG_es,SPG_AugLag,SPG_AugLag_rho,F_QP,F_AL,LIB,QP,QP_reg,LIBI,SMO,MY_LIB,PARAMETERS,RESULTS,DATASET,STR);

GRIDSEARCH.K_SVM{d,kernel} = K_SVM;
GRIDSEARCH.K_PG{d,kernel} = K_PG;
GRIDSEARCH.PG_C{d,kernel} = PG_C;
GRIDSEARCH.PG_N{d,kernel} = PG_N;
GRIDSEARCH.SPG_es{d,kernel} = SPG_es;
GRIDSEARCH.SPG_AugLag{d,kernel} = SPG_AugLag;
GRIDSEARCH.F_QP{d,kernel} = F_QP;
GRIDSEARCH.F_AL{d,kernel} = F_AL;
GRIDSEARCH.LIB{d,kernel} = LIB;
GRIDSEARCH.QP{d,kernel} = QP;
GRIDSEARCH.LIBI{d,kernel} = LIBI;
GRIDSEARCH.SMO{d,kernel} = SMO;
GRIDSEARCH.MY_LIB{d,kernel} = MY_LIB;
