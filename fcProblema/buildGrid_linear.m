% Objetivo: encontrar os melhores parametros de acordo com a acuracia
%           obtida na validacao

function [RESULTS] = buildGrid_linear(d,kernel,K_SVM,K_PG,PG_C,PG_N,SPG_es,SPG_AugLag,SPG_AugLag_rho,F_QP,F_AL,LIB,QP,QP_reg,LIBI,SMO,MY_LIB,PARAMETERS,RESULTS,DATASET,STR)

qtdCaixa= size(QP,2);
p = DATASET.p;
valoresCaixa = PARAMETERS.upperBounds;

aux = []; aux_error = [];
monta = []; monta_error = [];
%% building the matrix of all k-fold problem solved
    for c = 1:qtdCaixa %****** CAIXA
        for k = 1:PARAMETERS.k_fold %****** k-FOLD
            
            % accuracy
            acc_K_SVM(c,k)       = K_SVM{k,c}.accuracy;
            acc_K_PG(c,k)        = K_PG{k,c}.accuracy;
            acc_PG_C(c,k)        = PG_C{k,c}.accuracy;
            acc_PG_N(c,k)        = PG_N{k,c}.accuracy;
            acc_SPG_es(c,k)      = SPG_es{k,c}.accuracy;
            acc_SPG_AugLag(c,k)  = SPG_AugLag{k,c}.accuracy;
            acc_SPG_AugLag_rho(c,k) = SPG_AugLag_rho{k,c}.accuracy;
            acc_F_QP(c,k)        = F_QP{k,c}.accuracy;
            acc_F_AL(c,k)        = F_AL{k,c}.accuracy;
            acc_LIB(c,k)         = LIB{k,c}.accuracy;
            acc_QP(c,k)          = QP{k,c}.accuracy;
            acc_QP_reg(c,k)      = QP_reg{k,c}.accuracy;
            acc_LIBI(c,k)        = LIBI{k,c}.accuracy;
            acc_SMO(c,k)         = SMO{k,c}.accuracy;
            acc_MyLib(c,k)       = MY_LIB{k,c}.accuracy;

            % balanced accuracy
            accBal_K_SVM(c,k)       = K_SVM{k,c}.balancedAcc;
            accBal_K_PG(c,k)        = K_PG{k,c}.balancedAcc;
            accBal_PG_C(c,k)        = PG_C{k,c}.balancedAcc;
            accBal_PG_N(c,k)        = PG_N{k,c}.balancedAcc;
            accBal_SPG_es(c,k)      = SPG_es{k,c}.balancedAcc;
            accBal_SPG_AugLag(c,k)  = SPG_AugLag{k,c}.balancedAcc;
            accBal_SPG_AugLag_rho(c,k)  = SPG_AugLag_rho{k,c}.balancedAcc;
            accBal_F_QP(c,k)        = F_QP{k,c}.balancedAcc;
            accBal_F_AL(c,k)        = F_AL{k,c}.balancedAcc;
            accBal_LIB(c,k)         = LIB{k,c}.balancedAcc;
            accBal_QP(c,k)          = QP{k,c}.balancedAcc;
            accBal_QP_reg(c,k)      = QP_reg{k,c}.balancedAcc;
            accBal_LIBI(c,k)        = LIBI{k,c}.balancedAcc;
            accBal_SMO(c,k)        = SMO{k,c}.balancedAcc;
            accBal_MyLib(c,k)        = MY_LIB{k,c}.balancedAcc;
            

            classProportionPos_train = 100*size(find(DATASET.y_train(DATASET.idx_train{k,1}) == 1),1)/size(DATASET.idx_train{k,1},1);
            classProportionNeg_train = 100*size(find(DATASET.y_train(DATASET.idx_train{k,1}) == -1),1)/size(DATASET.idx_train{k,1},1);
            classProportionPos_val = 100*size(find(DATASET.y_train(DATASET.idx_val{k,1}) == 1),1)/size(DATASET.idx_val{k,1},1);
            classProportionNeg_val = 100*size(find(DATASET.y_train(DATASET.idx_val{k,1}) == -1),1)/size(DATASET.idx_val{k,1},1);
            
            aux = [aux;...
                %fold n_train n_test p gammaKernel C
                k size(DATASET.idx_train{k,1},1) size(DATASET.idx_val{k,1},1) p classProportionPos_train classProportionNeg_train classProportionPos_val classProportionNeg_val valoresCaixa(c) ...
                K_SVM{k,c}.outputs(10) sum(K_SVM{k,c}.outputs(10:11)) sum(K_PG{k,c}.outputs(10:11)) sum(PG_C{k,c}.outputs(10:11)) ...
                sum(PG_N{k,c}.outputs(10:11)) sum(SPG_es{k,c}.outputs(10:11)) sum(SPG_AugLag{k,c}.outputs(10:11)) sum(SPG_AugLag_rho{k,c}.outputs(10:11)) sum(F_QP{k,c}.outputs(10:11)) ...
                sum(F_AL{k,c}.outputs(10:11)) sum(LIB{k,c}.outputs(10:11)) sum(QP{k,c}.outputs(10:11)) sum(QP_reg{k,c}.outputs(10:11)) sum(LIBI{k,c}.outputs(10:11)) ...
                sum(SMO{k,c}.outputs(10:11)) sum(MY_LIB{k,c}.outputs(10:11)) ...
                K_SVM{k,c}.outputs(1) K_PG{k,c}.outputs(1) PG_C{k,c}.outputs(1) ...
                PG_N{k,c}.outputs(1) SPG_es{k,c}.outputs(1) SPG_AugLag{k,c}.outputs(1) SPG_AugLag_rho{k,c}.outputs(1) F_QP{k,c}.outputs(1) ...
                F_AL{k,c}.outputs(1) LIB{k,c}.outputs(1) QP{k,c}.outputs(1) QP_reg{k,c}.outputs(1) LIBI{k,c}.outputs(1) ...
                SMO{k,c}.outputs(1) MY_LIB{k,c}.outputs(1) ...
                K_SVM{k,c}.outputs(5) K_PG{k,c}.outputs(5) PG_C{k,c}.outputs(5) ...
                PG_N{k,c}.outputs(5) SPG_es{k,c}.outputs(5) SPG_AugLag{k,c}.outputs(5) SPG_AugLag_rho{k,c}.outputs(5) F_QP{k,c}.outputs(5) ...
                F_AL{k,c}.outputs(5) LIB{k,c}.outputs(5) QP{k,c}.outputs(5) QP_reg{k,c}.outputs(5) LIBI{k,c}.outputs(5) ...
                SMO{k,c}.outputs(5) MY_LIB{k,c}.outputs(5) ...
                K_SVM{k,c}.outputs(6:9) K_PG{k,c}.outputs(6:9) PG_C{k,c}.outputs(6:9) ...
                PG_N{k,c}.outputs(6:9) SPG_es{k,c}.outputs(6:9) SPG_AugLag{k,c}.outputs(6:9) SPG_AugLag_rho{k,c}.outputs(6:9) F_QP{k,c}.outputs(6:9) ...
                F_AL{k,c}.outputs(6:9) LIB{k,c}.outputs(6:9) QP{k,c}.outputs(6:9) QP_reg{k,c}.outputs(6:9) LIBI{k,c}.outputs(6:9) ...
                SMO{k,c}.outputs(6:9) MY_LIB{k,c}.outputs(6:9) ...
                K_SVM{k,c}.outputs(3) K_PG{k,c}.outputs(3) PG_C{k,c}.outputs(3) ...
                PG_N{k,c}.outputs(3) SPG_es{k,c}.outputs(3) SPG_AugLag{k,c}.outputs(3) SPG_AugLag_rho{k,c}.outputs(3) F_QP{k,c}.outputs(3) ...
                F_AL{k,c}.outputs(3) LIB{k,c}.outputs(3) QP{k,c}.outputs(3) QP_reg{k,c}.outputs(3) LIBI{k,c}.outputs(3) ...
                SMO{k,c}.outputs(3) MY_LIB{k,c}.outputs(3) ...
                K_SVM{k,c}.outputs(4) K_PG{k,c}.outputs(4) PG_C{k,c}.outputs(4) ...
                PG_N{k,c}.outputs(4) SPG_es{k,c}.outputs(4) SPG_AugLag{k,c}.outputs(4) SPG_AugLag_rho{k,c}.outputs(4) F_QP{k,c}.outputs(4) ...
                F_AL{k,c}.outputs(4) LIB{k,c}.outputs(4) QP{k,c}.outputs(4) QP_reg{k,c}.outputs(4) LIBI{k,c}.outputs(4) ...
                SMO{k,c}.outputs(4) MY_LIB{k,c}.outputs(4) ...
                K_SVM{k,c}.accuracy K_PG{k,c}.accuracy PG_C{k,c}.accuracy ...
                PG_N{k,c}.accuracy SPG_es{k,c}.accuracy SPG_AugLag{k,c}.accuracy SPG_AugLag_rho{k,c}.accuracy F_QP{k,c}.accuracy ...
                F_AL{k,c}.accuracy LIB{k,c}.accuracy QP{k,c}.accuracy QP_reg{k,c}.accuracy LIBI{k,c}.accuracy ...
                SMO{k,c}.accuracy MY_LIB{k,c}.accuracy ...
                K_SVM{k,c}.balancedAcc K_PG{k,c}.balancedAcc PG_C{k,c}.balancedAcc ...
                PG_N{k,c}.balancedAcc SPG_es{k,c}.balancedAcc SPG_AugLag{k,c}.balancedAcc SPG_AugLag_rho{k,c}.balancedAcc F_QP{k,c}.balancedAcc ...
                F_AL{k,c}.balancedAcc LIB{k,c}.balancedAcc QP{k,c}.balancedAcc QP_reg{k,c}.balancedAcc LIBI{k,c}.balancedAcc ...
                SMO{k,c}.balancedAcc MY_LIB{k,c}.balancedAcc ...
                K_SVM{k,c}.f1_score K_PG{k,c}.f1_score PG_C{k,c}.f1_score ...
                PG_N{k,c}.f1_score SPG_es{k,c}.f1_score SPG_AugLag{k,c}.f1_score SPG_AugLag_rho{k,c}.f1_score F_QP{k,c}.f1_score ...
                F_AL{k,c}.f1_score LIB{k,c}.f1_score QP{k,c}.f1_score QP_reg{k,c}.f1_score LIBI{k,c}.f1_score ...
                SMO{k,c}.f1_score MY_LIB{k,c}.f1_score ...
                K_SVM{k,c}.ConfMatr K_PG{k,c}.ConfMatr PG_C{k,c}.ConfMatr ...
                PG_N{k,c}.ConfMatr SPG_es{k,c}.ConfMatr SPG_AugLag{k,c}.ConfMatr SPG_AugLag_rho{k,c}.ConfMatr F_QP{k,c}.ConfMatr ...
                F_AL{k,c}.ConfMatr LIB{k,c}.ConfMatr QP{k,c}.ConfMatr QP_reg{k,c}.ConfMatr LIBI{k,c}.ConfMatr ...
                SMO{k,c}.ConfMatr MY_LIB{k,c}.ConfMatr ...
                K_SVM{k,c}.outputs(2) K_PG{k,c}.outputs(2) PG_C{k,c}.outputs(2) ...
                PG_N{k,c}.outputs(2) SPG_es{k,c}.outputs(2) SPG_AugLag{k,c}.outputs(2) SPG_AugLag_rho{k,c}.outputs(2) F_QP{k,c}.outputs(2) ...
                F_AL{k,c}.outputs(2) LIB{k,c}.outputs(2) QP{k,c}.outputs(2) QP_reg{k,c}.outputs(2) LIBI{k,c}.outputs(2) ...
                SMO{k,c}.outputs(2) MY_LIB{k,c}.outputs(2) ...
                ];
            
            aux_error = [aux_error;...
                K_SVM{k,c}.Error K_PG{k,c}.Error PG_C{k,c}.Error ...
                PG_N{k,c}.Error SPG_es{k,c}.Error SPG_AugLag{k,c}.Error SPG_AugLag_rho{k,c}.Error F_QP{k,c}.Error ...
                F_AL{k,c}.Error LIB{k,c}.Error QP{k,c}.Error QP_reg{k,c}.Error LIBI{k,c}.Error ...
                SMO{k,c}.Error MY_LIB{k,c}.Error ...
                ];
            
        end
    end
    
    monta = [monta;aux];
    aux = [];
    
    monta_error = [monta_error;aux_error];
    aux_error = [];


%%
% *** valores medios de acuracia
    for c = 1:qtdCaixa %****** CAIXA
        m_acc_K_SVM(c) = mean(acc_K_SVM(c,:));
        m_acc_K_PG(c) = mean(acc_K_PG(c,:));
        m_acc_PG_C(c) = mean(acc_PG_C(c,:));
        m_acc_PG_N(c) = mean(acc_PG_N(c,:));
        m_acc_SPG_es(c) = mean(acc_SPG_es(c,:));
        m_acc_SPG_AugLag(c) = mean(acc_SPG_AugLag(c,:));
        m_acc_SPG_AugLag_rho(c) = mean(acc_SPG_AugLag_rho(c,:));
        m_acc_F_QP(c) = mean(acc_F_QP(c,:));
        m_acc_F_AL(c) = mean(acc_F_AL(c,:));
        m_acc_LIB(c) = mean(acc_LIB(c,:));
        m_acc_QP(c) = mean(acc_QP(c,:));
        m_acc_QP_reg(c) = mean(acc_QP_reg(c,:));
        m_acc_LIBI(c) = mean(acc_LIBI(c,:));
        m_acc_SMO(c) = mean(acc_SMO(c,:));
        m_acc_MyLib(c) = mean(acc_MyLib(c,:));
        
        m_accBal_K_SVM(c) = mean(accBal_K_SVM(c,:));
        m_accBal_K_PG(c) = mean(accBal_K_PG(c,:));
        m_accBal_PG_C(c) = mean(accBal_PG_C(c,:));
        m_accBal_PG_N(c) = mean(accBal_PG_N(c,:));
        m_accBal_SPG_es(c) = mean(accBal_SPG_es(c,:));
        m_accBal_SPG_AugLag(c) = mean(accBal_SPG_AugLag(c,:));
        m_accBal_SPG_AugLag_rho(c) = mean(accBal_SPG_AugLag_rho(c,:));
        m_accBal_F_QP(c) = mean(accBal_F_QP(c,:));
        m_accBal_F_AL(c) = mean(accBal_F_AL(c,:));
        m_accBal_LIB(c) = mean(accBal_LIB(c,:));
        m_accBal_QP(c) = mean(accBal_QP(c,:));
        m_accBal_QP_reg(c) = mean(accBal_QP_reg(c,:));
        m_accBal_LIBI(c) = mean(accBal_LIBI(c,:));
        m_accBal_SMO(c) = mean(accBal_SMO(c,:));
        m_accBal_MyLib(c) = mean(accBal_MyLib(c,:));        
    end


% *** find the best values based in the higher mean accuracy
[result_LIB] = organizingBestParameters(m_acc_LIB,valoresCaixa,[]);

if STR.runKnapsack_SVM, [result_K_SVM] = organizingBestParameters(m_acc_K_SVM,valoresCaixa,[]);
else, result_K_SVM = result_LIB; end
    
if STR.runKnapsack_ProjGrad,[result_K_PG] = organizingBestParameters(m_acc_K_PG,valoresCaixa,[]);
else,result_K_PG=result_LIB; end
    
if STR.runProjGrad_cauchy,[result_PG_C] = organizingBestParameters(m_acc_PG_C,valoresCaixa,[]);
else, result_PG_C = result_LIB;end
    
if STR.runProjGrad_newton, [result_PG_N] = organizingBestParameters(m_acc_PG_N,valoresCaixa,[]);
else,result_PG_N=result_LIB; end

if STR.runSPG_exactSearch,[result_SPG_es] = organizingBestParameters(m_acc_SPG_es,valoresCaixa,[]);
else, result_SPG_es=result_LIB; end

if STR.runSPG_augLag,[result_SPG_AugLag] = organizingBestParameters(m_acc_SPG_AugLag,valoresCaixa,[]);
else, result_SPG_AugLag=result_LIB; end

if STR.runSPG_augLag_rho,[result_SPG_AugLag_rho] = organizingBestParameters(m_acc_SPG_AugLag_rho,valoresCaixa,[]);
else, result_SPG_AugLag_rho=result_LIB; end
    
if STR.runFilter_qp,[result_F_QP] = organizingBestParameters(m_acc_F_QP,valoresCaixa,[]);
else, result_F_QP=result_LIB; end
    
if STR.runFilter_AL,[result_F_AL] = organizingBestParameters(m_acc_F_AL,valoresCaixa,[]);
else, result_F_AL=result_LIB; end

if STR.runQuadprog, [result_QP] = organizingBestParameters(m_acc_QP,valoresCaixa,[]);
else, result_QP=result_LIB; end

if STR.runQuadprog_reg,[result_QP_reg] = organizingBestParameters(m_acc_QP_reg,valoresCaixa,[]);
else, result_QP_reg=result_LIB; end

if STR.runLiblinear,[result_LIBI] = organizingBestParameters(m_acc_LIBI,valoresCaixa,[]);
else, result_LIBI=result_LIB; end

if STR.runSMO,[result_SMO] = organizingBestParameters(m_acc_SMO,valoresCaixa,[]);
else, result_SMO=result_LIB; end

if STR.runMyLibsvm,[result_MyLib] = organizingBestParameters(m_acc_MyLib,valoresCaixa,[]);
else, result_MyLib=result_LIB; end


bestParameters = [result_K_SVM;result_K_PG;result_PG_C;result_PG_N;...
                  result_SPG_es;result_SPG_AugLag;result_SPG_AugLag_rho;result_F_QP;...
                  result_F_AL;result_LIB;result_QP;result_QP_reg;result_LIBI;...
                  result_SMO;result_MyLib];

RESULTS.tableGridSearch{d,kernel} = full(monta);
RESULTS.tableGridSearchError{d,kernel} = monta_error;
RESULTS.bestParameters{d,kernel} = bestParameters;

end



