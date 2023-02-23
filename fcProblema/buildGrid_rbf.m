% Objetivo: encontrar os melhores parametros de acordo com a acuracia
%           obtida na validacao

function [RESULTS] = buildGrid_rbf(d,kernel,K_SVM,K_PG,PG_C,PG_N,SPG_es,SPG_AugLag,SPG_AugLag_rho,F_QP,F_AL,LIB,QP,QP_reg,LIBI,SMO,MY_LIB,PARAMETERS,RESULTS,DATASET,STR)

qtdGmKr = size(QP,3);
qtdCaixa= size(QP,2);
p = DATASET.p;
valoresGmKrnl = PARAMETERS.valoresGmKrnl;
valoresCaixa = PARAMETERS.upperBounds;

aux = []; aux_error = [];
monta = []; monta_error = [];

%% building the matrix of all k-fold problem solved
for g = 1:qtdGmKr %****** GAMMA_KERNEL
    for c = 1:qtdCaixa %****** CAIXA
        for k = 1:PARAMETERS.k_fold %****** k-FOLD

            % accuracy
            acc_K_SVM(c,g,k)       = K_SVM{k,c,g}.accuracy;
            acc_K_PG(c,g,k)        = K_PG{k,c,g}.accuracy;
            acc_PG_C(c,g,k)        = PG_C{k,c,g}.accuracy;
            acc_PG_N(c,g,k)        = PG_N{k,c,g}.accuracy;
            acc_SPG_es(c,g,k)      = SPG_es{k,c,g}.accuracy;
            acc_SPG_AugLag(c,g,k)  = SPG_AugLag{k,c,g}.accuracy;
            acc_SPG_AugLag_rho(c,g,k)  = SPG_AugLag_rho{k,c,g}.accuracy;
            acc_F_QP(c,g,k)        = F_QP{k,c,g}.accuracy;
            acc_F_AL(c,g,k)        = F_AL{k,c,g}.accuracy;
            acc_LIB(c,g,k)         = LIB{k,c,g}.accuracy;
            acc_QP(c,g,k)          = QP{k,c,g}.accuracy;
            acc_QP_reg(c,g,k)      = QP_reg{k,c,g}.accuracy;
            acc_LIBI(c,g,k)        = LIBI{k,c,g}.accuracy;
            acc_SMO(c,g,k)         = SMO{k,c,g}.accuracy;
            acc_MyLib(c,g,k)       = MY_LIB{k,c,g}.accuracy;

            % balanced accuracy
            accBal_K_SVM(c,g,k)       = K_SVM{k,c,g}.balancedAcc;
            accBal_K_PG(c,g,k)        = K_PG{k,c,g}.balancedAcc;
            accBal_PG_C(c,g,k)        = PG_C{k,c,g}.balancedAcc;
            accBal_PG_N(c,g,k)        = PG_N{k,c,g}.balancedAcc;
            accBal_SPG_es(c,g,k)      = SPG_es{k,c,g}.balancedAcc;
            accBal_SPG_AugLag(c,g,k)  = SPG_AugLag{k,c,g}.balancedAcc;
            accBal_SPG_AugLag_rho(c,g,k)  = SPG_AugLag_rho{k,c,g}.balancedAcc;
            accBal_F_QP(c,g,k)        = F_QP{k,c,g}.balancedAcc;
            accBal_F_AL(c,g,k)        = F_AL{k,c,g}.balancedAcc;
            accBal_LIB(c,g,k)         = LIB{k,c,g}.balancedAcc;
            accBal_QP(c,g,k)          = QP{k,c,g}.balancedAcc;
            accBal_QP_reg(c,g,k)      = QP_reg{k,c,g}.balancedAcc;
            accBal_LIBI(c,g,k)        = LIBI{k,c,g}.balancedAcc;
            accBal_SMO(c,g,k)        = SMO{k,c,g}.balancedAcc;
            accBal_MyLib(c,g,k)        = MY_LIB{k,c,g}.balancedAcc;


            classProportionPos_train = 100*size(find(DATASET.y_train(DATASET.idx_train{k,1}) == 1),1)/size(DATASET.idx_train{k,1},1);
            classProportionNeg_train = 100*size(find(DATASET.y_train(DATASET.idx_train{k,1}) == -1),1)/size(DATASET.idx_train{k,1},1);
            classProportionPos_val = 100*size(find(DATASET.y_train(DATASET.idx_val{k,1}) == 1),1)/size(DATASET.idx_val{k,1},1);
            classProportionNeg_val = 100*size(find(DATASET.y_train(DATASET.idx_val{k,1}) == -1),1)/size(DATASET.idx_val{k,1},1);
            
            aux = [aux;...
                %fold n_train n_test p gammaKernel C
                k size(DATASET.idx_train{k,1},1) size(DATASET.idx_val{k,1},1) p classProportionPos_train classProportionNeg_train classProportionPos_val classProportionNeg_val valoresGmKrnl(g) valoresCaixa(c) ...
                K_SVM{k,c,g}.outputs(10) sum(K_SVM{k,c,g}.outputs(10:11)) sum(K_PG{k,c,g}.outputs(10:11)) sum(PG_C{k,c,g}.outputs(10:11)) ...
                sum(PG_N{k,c,g}.outputs(10:11)) sum(SPG_es{k,c,g}.outputs(10:11)) sum(SPG_AugLag{k,c,g}.outputs(10:11)) sum(SPG_AugLag_rho{k,c,g}.outputs(10:11)) sum(F_QP{k,c,g}.outputs(10:11)) ...
                sum(F_AL{k,c,g}.outputs(10:11)) sum(LIB{k,c,g}.outputs(10:11)) sum(QP{k,c,g}.outputs(10:11)) sum(QP_reg{k,c,g}.outputs(10:11)) sum(LIBI{k,c,g}.outputs(10:11)) ...
                sum(SMO{k,c,g}.outputs(10:11)) sum(MY_LIB{k,c,g}.outputs(10:11)) ...
                K_SVM{k,c,g}.outputs(1) K_PG{k,c,g}.outputs(1) PG_C{k,c,g}.outputs(1) ...
                PG_N{k,c,g}.outputs(1) SPG_es{k,c,g}.outputs(1) SPG_AugLag{k,c,g}.outputs(1) SPG_AugLag_rho{k,c,g}.outputs(1) F_QP{k,c,g}.outputs(1) ...
                F_AL{k,c,g}.outputs(1) LIB{k,c,g}.outputs(1) QP{k,c,g}.outputs(1) QP_reg{k,c,g}.outputs(1) LIBI{k,c,g}.outputs(1) ...
                SMO{k,c,g}.outputs(1) MY_LIB{k,c,g}.outputs(1) ...
                K_SVM{k,c,g}.outputs(5) K_PG{k,c,g}.outputs(5) PG_C{k,c,g}.outputs(5) ...
                PG_N{k,c,g}.outputs(5) SPG_es{k,c,g}.outputs(5) SPG_AugLag{k,c,g}.outputs(5) SPG_AugLag_rho{k,c,g}.outputs(5) F_QP{k,c,g}.outputs(5) ...
                F_AL{k,c,g}.outputs(5) LIB{k,c,g}.outputs(5) QP{k,c,g}.outputs(5) QP_reg{k,c,g}.outputs(5) LIBI{k,c,g}.outputs(5) ...
                SMO{k,c,g}.outputs(5) MY_LIB{k,c,g}.outputs(5) ...
                K_SVM{k,c,g}.outputs(6:9) K_PG{k,c,g}.outputs(6:9) PG_C{k,c,g}.outputs(6:9) ...
                PG_N{k,c,g}.outputs(6:9) SPG_es{k,c,g}.outputs(6:9) SPG_AugLag{k,c,g}.outputs(6:9) SPG_AugLag_rho{k,c,g}.outputs(6:9) F_QP{k,c,g}.outputs(6:9) ...
                F_AL{k,c,g}.outputs(6:9) LIB{k,c,g}.outputs(6:9) QP{k,c,g}.outputs(6:9) QP_reg{k,c,g}.outputs(6:9) LIBI{k,c,g}.outputs(6:9) ...
                SMO{k,c,g}.outputs(6:9) MY_LIB{k,c,g}.outputs(6:9) ...
                K_SVM{k,c,g}.outputs(3) K_PG{k,c,g}.outputs(3) PG_C{k,c,g}.outputs(3) ...
                PG_N{k,c,g}.outputs(3) SPG_es{k,c,g}.outputs(3) SPG_AugLag{k,c,g}.outputs(3) SPG_AugLag_rho{k,c,g}.outputs(3) F_QP{k,c,g}.outputs(3) ...
                F_AL{k,c,g}.outputs(3) LIB{k,c,g}.outputs(3) QP{k,c,g}.outputs(3) QP_reg{k,c,g}.outputs(3) LIBI{k,c,g}.outputs(3) ...
                SMO{k,c,g}.outputs(3) MY_LIB{k,c,g}.outputs(3) ...
                K_SVM{k,c,g}.outputs(4) K_PG{k,c,g}.outputs(4) PG_C{k,c,g}.outputs(4) ...
                PG_N{k,c,g}.outputs(4) SPG_es{k,c,g}.outputs(4) SPG_AugLag{k,c,g}.outputs(4) SPG_AugLag_rho{k,c,g}.outputs(4) F_QP{k,c,g}.outputs(4) ...
                F_AL{k,c,g}.outputs(4) LIB{k,c,g}.outputs(4) QP{k,c,g}.outputs(4) QP_reg{k,c,g}.outputs(4) LIBI{k,c,g}.outputs(4) ...
                SMO{k,c,g}.outputs(4) MY_LIB{k,c,g}.outputs(4) ...
                K_SVM{k,c,g}.accuracy K_PG{k,c,g}.accuracy PG_C{k,c,g}.accuracy ...
                PG_N{k,c,g}.accuracy SPG_es{k,c,g}.accuracy SPG_AugLag{k,c,g}.accuracy SPG_AugLag_rho{k,c,g}.accuracy F_QP{k,c,g}.accuracy ...
                F_AL{k,c,g}.accuracy LIB{k,c,g}.accuracy QP{k,c,g}.accuracy QP_reg{k,c,g}.accuracy LIBI{k,c,g}.accuracy ...
                SMO{k,c,g}.accuracy MY_LIB{k,c,g}.accuracy ...
                K_SVM{k,c,g}.balancedAcc K_PG{k,c,g}.balancedAcc PG_C{k,c,g}.balancedAcc ...
                PG_N{k,c,g}.balancedAcc SPG_es{k,c,g}.balancedAcc SPG_AugLag{k,c,g}.balancedAcc SPG_AugLag_rho{k,c,g}.balancedAcc F_QP{k,c,g}.balancedAcc ...
                F_AL{k,c,g}.balancedAcc LIB{k,c,g}.balancedAcc QP{k,c,g}.balancedAcc QP_reg{k,c,g}.balancedAcc LIBI{k,c,g}.balancedAcc ...
                SMO{k,c,g}.balancedAcc MY_LIB{k,c,g}.balancedAcc ...
                K_SVM{k,c,g}.f1_score K_PG{k,c,g}.f1_score PG_C{k,c,g}.f1_score ...
                PG_N{k,c,g}.f1_score SPG_es{k,c,g}.f1_score SPG_AugLag{k,c,g}.f1_score SPG_AugLag_rho{k,c,g}.f1_score F_QP{k,c,g}.f1_score ...
                F_AL{k,c,g}.f1_score LIB{k,c,g}.f1_score QP{k,c,g}.f1_score QP_reg{k,c,g}.f1_score LIBI{k,c,g}.f1_score ...
                SMO{k,c,g}.f1_score MY_LIB{k,c,g}.f1_score ...
                K_SVM{k,c,g}.ConfMatr K_PG{k,c,g}.ConfMatr PG_C{k,c,g}.ConfMatr ...
                PG_N{k,c,g}.ConfMatr SPG_es{k,c,g}.ConfMatr SPG_AugLag{k,c,g}.ConfMatr SPG_AugLag_rho{k,c,g}.ConfMatr F_QP{k,c,g}.ConfMatr ...
                F_AL{k,c,g}.ConfMatr LIB{k,c,g}.ConfMatr QP{k,c,g}.ConfMatr QP_reg{k,c,g}.ConfMatr LIBI{k,c,g}.ConfMatr ...
                SMO{k,c,g}.ConfMatr MY_LIB{k,c,g}.ConfMatr ...
                K_SVM{k,c,g}.outputs(2) K_PG{k,c,g}.outputs(2) PG_C{k,c,g}.outputs(2) ...
                PG_N{k,c,g}.outputs(2) SPG_es{k,c,g}.outputs(2) SPG_AugLag{k,c,g}.outputs(2) SPG_AugLag_rho{k,c,g}.outputs(2) F_QP{k,c,g}.outputs(2) ...
                F_AL{k,c,g}.outputs(2) LIB{k,c,g}.outputs(2) QP{k,c,g}.outputs(2) QP_reg{k,c,g}.outputs(2) LIBI{k,c,g}.outputs(2) ...
                SMO{k,c,g}.outputs(2) MY_LIB{k,c,g}.outputs(2) ...
                ];
            
            aux_error = [aux_error;...
                K_SVM{k,c,g}.Error K_PG{k,c,g}.Error PG_C{k,c,g}.Error ...
                PG_N{k,c,g}.Error SPG_es{k,c,g}.Error SPG_AugLag{k,c,g}.Error F_QP{k,c,g}.Error ...
                F_AL{k,c,g}.Error LIB{k,c,g}.Error QP{k,c,g}.Error LIBI{k,c,g}.Error ...
                SMO{k,c,g}.Error MY_LIB{k,c,g}.Error ...
                ];
            
            
        end
    end
    
    monta = [monta;aux];
    aux = [];
    
    monta_error = [monta_error;aux_error];
    aux_error = [];
end

%%
% *** valores medios de acuracia
for g = 1:qtdGmKr %****** GAMMA_KERNEL
    for c = 1:qtdCaixa %****** CAIXA
        m_acc_K_SVM(c,g) = mean(acc_K_SVM(c,g,:));
        m_acc_K_PG(c,g) = mean(acc_K_PG(c,g,:));
        m_acc_PG_C(c,g) = mean(acc_PG_C(c,g,:));
        m_acc_PG_N(c,g) = mean(acc_PG_N(c,g,:));
        m_acc_SPG_es(c,g) = mean(acc_SPG_es(c,g,:));
        m_acc_SPG_AugLag(c,g) = mean(acc_SPG_AugLag(c,g,:));
        m_acc_SPG_AugLag_rho(c,g) = mean(acc_SPG_AugLag_rho(c,g,:));
        m_acc_F_QP(c,g) = mean(acc_F_QP(c,g,:));
        m_acc_F_AL(c,g) = mean(acc_F_AL(c,g,:));
        m_acc_LIB(c,g) = mean(acc_LIB(c,g,:));
        m_acc_QP(c,g) = mean(acc_QP(c,g,:));
        m_acc_QP_reg(c,g) = mean(acc_QP_reg(c,g,:));
        m_acc_LIBI(c,g) = mean(acc_LIBI(c,g,:));
        m_acc_SMO(c,g) = mean(acc_SMO(c,g,:));
        m_acc_MyLib(c,g) = mean(acc_MyLib(c,g,:));
        
        
        m_accBal_K_SVM(c,g) = mean(accBal_K_SVM(c,g,:));
        m_accBal_K_PG(c,g) = mean(accBal_K_PG(c,g,:));
        m_accBal_PG_C(c,g) = mean(accBal_PG_C(c,g,:));
        m_accBal_PG_N(c,g) = mean(accBal_PG_N(c,g,:));
        m_accBal_SPG_es(c,g) = mean(accBal_SPG_es(c,g,:));
        m_accBal_SPG_AugLag(c,g) = mean(accBal_SPG_AugLag(c,g,:));
        m_accBal_SPG_AugLag_rho(c,g) = mean(accBal_SPG_AugLag_rho(c,g,:));
        m_accBal_F_QP(c,g) = mean(accBal_F_QP(c,g,:));
        m_accBal_F_AL(c,g) = mean(accBal_F_AL(c,g,:));
        m_accBal_LIB(c,g) = mean(accBal_LIB(c,g,:));
        m_accBal_QP(c,g) = mean(accBal_QP(c,g,:));
        m_accBal_QP_reg(c,g) = mean(accBal_QP_reg(c,g,:));
        m_accBal_LIBI(c,g) = mean(accBal_LIBI(c,g,:));   
        m_accBal_SMO(c,g) = mean(accBal_SMO(c,g,:));
        m_accBal_MyLib(c,g) = mean(accBal_MyLib(c,g,:));
        
    end
end

% *** find the best values based in the higher mean accuracy
[result_LIB] = organizingBestParameters(m_acc_LIB,valoresCaixa,valoresGmKrnl);

if STR.runKnapsack_SVM, [result_K_SVM] = organizingBestParameters(m_acc_K_SVM,valoresCaixa,valoresGmKrnl);
else, result_K_SVM = result_LIB; end
    
if STR.runKnapsack_ProjGrad,[result_K_PG] = organizingBestParameters(m_acc_K_PG,valoresCaixa,valoresGmKrnl);
else,result_K_PG=result_LIB; end
    
if STR.runProjGrad_cauchy,[result_PG_C] = organizingBestParameters(m_acc_PG_C,valoresCaixa,valoresGmKrnl);
else, result_PG_C = result_LIB;end
    
if STR.runProjGrad_newton, [result_PG_N] = organizingBestParameters(m_acc_PG_N,valoresCaixa,valoresGmKrnl);
else,result_PG_N=result_LIB; end

if STR.runSPG_exactSearch,[result_SPG_es] = organizingBestParameters(m_acc_SPG_es,valoresCaixa,valoresGmKrnl);
else, result_SPG_es=result_LIB; end

if STR.runSPG_augLag,[result_SPG_AugLag] = organizingBestParameters(m_acc_SPG_AugLag,valoresCaixa,valoresGmKrnl);
else, result_SPG_AugLag=result_LIB; end
    
if STR.runSPG_augLag_rho,[result_SPG_AugLag_rho] = organizingBestParameters(m_acc_SPG_AugLag_rho,valoresCaixa,valoresGmKrnl);
else, result_SPG_AugLag_rho=result_LIB; end

if STR.runFilter_qp,[result_F_QP] = organizingBestParameters(m_acc_F_QP,valoresCaixa,valoresGmKrnl);
else, result_F_QP=result_LIB; end
    
if STR.runFilter_AL,[result_F_AL] = organizingBestParameters(m_acc_F_AL,valoresCaixa,valoresGmKrnl);
else, result_F_AL=result_LIB; end

if STR.runQuadprog,[result_QP] = organizingBestParameters(m_acc_QP,valoresCaixa,valoresGmKrnl);
else, result_QP=result_LIB; end

if STR.runQuadprog_reg, [result_QP_reg] = organizingBestParameters(m_acc_QP_reg,valoresCaixa,valoresGmKrnl);
else, result_QP_reg=result_LIB; end

if STR.runLiblinear,[result_LIBI] = organizingBestParameters(m_acc_LIBI,valoresCaixa,[]);
else, result_LIBI=result_LIB; end

if STR.runSMO,[result_SMO] = organizingBestParameters(m_acc_SMO,valoresCaixa,valoresGmKrnl);
else, result_SMO=result_LIB; end

if STR.runMyLibsvm,[result_MyLib] = organizingBestParameters(m_acc_MyLib,valoresCaixa,valoresGmKrnl);
else, result_MyLib=result_LIB; end

bestParameters = [result_K_SVM;result_K_PG;result_PG_C;result_PG_N;...
                  result_SPG_es;result_SPG_AugLag;result_SPG_AugLag_rho;result_F_QP;...
                  result_F_AL;result_LIB;result_QP;result_QP_reg;result_LIBI;...
                  result_SMO;result_MyLib];


RESULTS.tableGridSearch{d,kernel} = full(monta);
RESULTS.tableGridSearchError{d,kernel} = monta_error;
RESULTS.bestParameters{d,kernel} = bestParameters;


