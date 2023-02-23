% Objetivo: encontrar os melhores parametros de acordo com a acuracia
%           obtida na validacao

function [RESULTS] = buildGrid_poly(d,kernel,K_SVM,K_PG,PG_C,PG_N,SPG_es,SPG_AugLag,SPG_AugLag_rho,F_QP,F_AL,LIB,QP,QP_reg,LIBI,SMO,MY_LIB,PARAMETERS,RESULTS,DATASET,STR)
valoresCaixa = PARAMETERS.upperBounds;
valores_d = PARAMETERS.valores_d;
valores_r = PARAMETERS.valores_r;

qtdR = size(QP,4);
qtdD = size(QP,3);
auxMonta = []; auxMontaError = [];
bestParameters = {}; monta = []; monta_error = [];
auxpara = []; parameters =[];
for grau = 1:qtdD
    for r = 1:qtdR
        [auxpara,auxMonta,auxMontaError] = ...
            auxiliarFunction(grau,r,[],[],...
            K_SVM,K_PG,PG_C,PG_N,SPG_es,SPG_AugLag,SPG_AugLag_rho,F_QP,...
            F_AL,LIB,QP,QP_reg,LIBI,SMO,MY_LIB,PARAMETERS,DATASET,STR);
        
        auxpara = [auxpara(:,1:4) valores_d(grau)*ones(15,1) valores_r(r)*ones(15,1)];
        parameters(:,:,grau,r) = auxpara;
        
        monta = [monta;auxMonta];
        monta_error = [monta_error;auxMontaError];

    end
end

bestParameters = [];
for m = 1:13
    acc_Method = permute(parameters(m,3,:,:),[3 4 1 2]);
    
    [i,j] = find(acc_Method == max(max(acc_Method)));
    
    bestParameters = [bestParameters;...
                        parameters(m,:,i(1),j(1))];
end


RESULTS.tableGridSearch{d,kernel} = full(monta);
RESULTS.tableGridSearchError{d,kernel} = monta_error;
RESULTS.bestParameters{d,kernel} = bestParameters;
end

%% building the matrix of all k-fold problem solved
function [bestParameters,monta,monta_error] = auxiliarFunction(grau,r,monta,monta_error,K_SVM,K_PG,PG_C,PG_N,SPG_es,SPG_AugLag,SPG_AugLag_rho,F_QP,F_AL,LIB,QP,QP_reg,LIBI,SMO,MY_LIB,PARAMETERS,DATASET,STR)
qtdGmKr = size(QP,5);
qtdCaixa= size(QP,2);
p = DATASET.p;
valoresGmKrnl = PARAMETERS.valoresGmKrnl;
valoresCaixa = PARAMETERS.upperBounds;
valores_d = PARAMETERS.valores_d;
valores_r = PARAMETERS.valores_r;

aux = []; aux_error = [];
for g = 1:qtdGmKr %****** GAMMA_KERNEL
    for c = 1:qtdCaixa %****** CAIXA
        for k = 1:PARAMETERS.k_fold %****** k-FOLD

            % accuracy
            acc_K_SVM(c,g,k)       = K_SVM{k,c,grau,r,g}.accuracy;
            acc_K_PG(c,g,k)        = K_PG{k,c,grau,r,g}.accuracy;
            acc_PG_C(c,g,k)        = PG_C{k,c,grau,r,g}.accuracy;
            acc_PG_N(c,g,k)        = PG_N{k,c,grau,r,g}.accuracy;
            acc_SPG_es(c,g,k)      = SPG_es{k,c,grau,r,g}.accuracy;
            acc_SPG_AugLag(c,g,k)  = SPG_AugLag{k,c,grau,r,g}.accuracy;
            acc_SPG_AugLag_rho(c,g,k)  = SPG_AugLag_rho{k,c,grau,r,g}.accuracy;
            acc_F_QP(c,g,k)        = F_QP{k,c,grau,r,g}.accuracy;
            acc_F_AL(c,g,k)        = F_AL{k,c,grau,r,g}.accuracy;
            acc_LIB(c,g,k)         = LIB{k,c,grau,r,g}.accuracy;
            acc_QP(c,g,k)          = QP{k,c,grau,r,g}.accuracy;
            acc_QP_reg(c,g,k)      = QP_reg{k,c,grau,r,g}.accuracy;
            acc_LIBI(c,g,k)        = LIBI{k,c,grau,r,g}.accuracy;
            acc_SMO(c,g,k)         = SMO{k,c,grau,r,g}.accuracy;
            acc_MyLib(c,g,k)       = MY_LIB{k,c,grau,r,g}.accuracy;

            % balanced accuracy
            accBal_K_SVM(c,g,k)       = K_SVM{k,c,grau,r,g}.balancedAcc;
            accBal_K_PG(c,g,k)        = K_PG{k,c,grau,r,g}.balancedAcc;
            accBal_PG_C(c,g,k)        = PG_C{k,c,grau,r,g}.balancedAcc;
            accBal_PG_N(c,g,k)        = PG_N{k,c,grau,r,g}.balancedAcc;
            accBal_SPG_es(c,g,k)      = SPG_es{k,c,grau,r,g}.balancedAcc;
            accBal_SPG_AugLag(c,g,k)  = SPG_AugLag{k,c,grau,r,g}.balancedAcc;
            accBal_SPG_AugLag_rho(c,g,k)  = SPG_AugLag_rho{k,c,grau,r,g}.balancedAcc;
            accBal_F_QP(c,g,k)        = F_QP{k,c,grau,r,g}.balancedAcc;
            accBal_F_AL(c,g,k)        = F_AL{k,c,grau,r,g}.balancedAcc;
            accBal_LIB(c,g,k)         = LIB{k,c,grau,r,g}.balancedAcc;
            accBal_QP(c,g,k)          = QP{k,c,grau,r,g}.balancedAcc;
            accBal_QP_reg(c,g,k)      = QP_reg{k,c,grau,r,g}.balancedAcc;
            accBal_LIBI(c,g,k)        = LIBI{k,c,grau,r,g}.balancedAcc;
            accBal_SMO(c,g,k)        = SMO{k,c,grau,r,g}.balancedAcc;
            accBal_MyLib(c,g,k)        = MY_LIB{k,c,grau,r,g}.balancedAcc;
            

            classProportionPos_train = 100*size(find(DATASET.y_train(DATASET.idx_train{k,1}) == 1),1)/size(DATASET.idx_train{k,1},1);
            classProportionNeg_train = 100*size(find(DATASET.y_train(DATASET.idx_train{k,1}) == -1),1)/size(DATASET.idx_train{k,1},1);
            classProportionPos_val = 100*size(find(DATASET.y_train(DATASET.idx_val{k,1}) == 1),1)/size(DATASET.idx_val{k,1},1);
            classProportionNeg_val = 100*size(find(DATASET.y_train(DATASET.idx_val{k,1}) == -1),1)/size(DATASET.idx_val{k,1},1);
            
            aux = [aux;...
                %fold n_train n_test p gammaKernel C
                k size(DATASET.idx_train{k,1},1) size(DATASET.idx_val{k,1},1) p classProportionPos_train classProportionNeg_train classProportionPos_val classProportionNeg_val valores_d(grau) valores_r(r) valoresGmKrnl(g) valoresCaixa(c) ...
                K_SVM{k,c,grau,r,g}.outputs(10) sum(K_SVM{k,c,grau,r,g}.outputs(10:11)) sum(K_PG{k,c,grau,r,g}.outputs(10:11)) sum(PG_C{k,c,grau,r,g}.outputs(10:11)) ...
                sum(PG_N{k,c,grau,r,g}.outputs(10:11)) sum(SPG_es{k,c,grau,r,g}.outputs(10:11)) sum(SPG_AugLag{k,c,grau,r,g}.outputs(10:11)) sum(SPG_AugLag_rho{k,c,grau,r,g}.outputs(10:11)) sum(F_QP{k,c,grau,r,g}.outputs(10:11)) ...
                sum(F_AL{k,c,grau,r,g}.outputs(10:11)) sum(LIB{k,c,grau,r,g}.outputs(10:11)) sum(QP{k,c,grau,r,g}.outputs(10:11)) sum(QP_reg{k,c,grau,r,g}.outputs(10:11)) sum(LIBI{k,c,grau,r,g}.outputs(10:11)) ...
                sum(SMO{k,c,grau,r,g}.outputs(10:11)) sum(MY_LIB{k,c,grau,r,g}.outputs(10:11)) ...
                K_SVM{k,c,grau,r,g}.outputs(1) K_PG{k,c,grau,r,g}.outputs(1) PG_C{k,c,grau,r,g}.outputs(1) ...
                PG_N{k,c,grau,r,g}.outputs(1) SPG_es{k,c,grau,r,g}.outputs(1) SPG_AugLag{k,c,grau,r,g}.outputs(1) SPG_AugLag_rho{k,c,grau,r,g}.outputs(1) F_QP{k,c,grau,r,g}.outputs(1) ...
                F_AL{k,c,grau,r,g}.outputs(1) LIB{k,c,grau,r,g}.outputs(1) QP{k,c,grau,r,g}.outputs(1) QP_reg{k,c,grau,r,g}.outputs(1) LIBI{k,c,grau,r,g}.outputs(1) ...
                SMO{k,c,grau,r,g}.outputs(1) MY_LIB{k,c,grau,r,g}.outputs(1) ...
                K_SVM{k,c,grau,r,g}.outputs(5) K_PG{k,c,grau,r,g}.outputs(5) PG_C{k,c,grau,r,g}.outputs(5) ...
                PG_N{k,c,grau,r,g}.outputs(5) SPG_es{k,c,grau,r,g}.outputs(5) SPG_AugLag{k,c,grau,r,g}.outputs(5) SPG_AugLag_rho{k,c,grau,r,g}.outputs(5) F_QP{k,c,grau,r,g}.outputs(5) ...
                F_AL{k,c,grau,r,g}.outputs(5) LIB{k,c,grau,r,g}.outputs(5) QP{k,c,grau,r,g}.outputs(5) QP_reg{k,c,grau,r,g}.outputs(5) LIBI{k,c,grau,r,g}.outputs(5) ...
                SMO{k,c,grau,r,g}.outputs(5) MY_LIB{k,c,grau,r,g}.outputs(5) ...
                K_SVM{k,c,grau,r,g}.outputs(6:9) K_PG{k,c,grau,r,g}.outputs(6:9) PG_C{k,c,grau,r,g}.outputs(6:9) ...
                PG_N{k,c,grau,r,g}.outputs(6:9) SPG_es{k,c,grau,r,g}.outputs(6:9) SPG_AugLag{k,c,grau,r,g}.outputs(6:9) SPG_AugLag_rho{k,c,grau,r,g}.outputs(6:9) F_QP{k,c,grau,r,g}.outputs(6:9) ...
                F_AL{k,c,grau,r,g}.outputs(6:9) LIB{k,c,grau,r,g}.outputs(6:9) QP{k,c,grau,r,g}.outputs(6:9) QP_reg{k,c,grau,r,g}.outputs(6:9) LIBI{k,c,grau,r,g}.outputs(6:9) ...
                SMO{k,c,grau,r,g}.outputs(6:9) MY_LIB{k,c,grau,r,g}.outputs(6:9) ...
                K_SVM{k,c,grau,r,g}.outputs(3) K_PG{k,c,grau,r,g}.outputs(3) PG_C{k,c,grau,r,g}.outputs(3) ...
                PG_N{k,c,grau,r,g}.outputs(3) SPG_es{k,c,grau,r,g}.outputs(3) SPG_AugLag{k,c,grau,r,g}.outputs(3) SPG_AugLag_rho{k,c,grau,r,g}.outputs(3) F_QP{k,c,grau,r,g}.outputs(3) ...
                F_AL{k,c,grau,r,g}.outputs(3) LIB{k,c,grau,r,g}.outputs(3) QP{k,c,grau,r,g}.outputs(3) QP_reg{k,c,grau,r,g}.outputs(3) LIBI{k,c,grau,r,g}.outputs(3) ...
                SMO{k,c,grau,r,g}.outputs(3) MY_LIB{k,c,grau,r,g}.outputs(3) ...
                K_SVM{k,c,grau,r,g}.outputs(4) K_PG{k,c,grau,r,g}.outputs(4) PG_C{k,c,grau,r,g}.outputs(4) ...
                PG_N{k,c,grau,r,g}.outputs(4) SPG_es{k,c,grau,r,g}.outputs(4) SPG_AugLag{k,c,grau,r,g}.outputs(4) SPG_AugLag_rho{k,c,grau,r,g}.outputs(4) F_QP{k,c,grau,r,g}.outputs(4) ...
                F_AL{k,c,grau,r,g}.outputs(4) LIB{k,c,grau,r,g}.outputs(4) QP{k,c,grau,r,g}.outputs(4) QP_reg{k,c,grau,r,g}.outputs(4) LIBI{k,c,grau,r,g}.outputs(4) ...
                SMO{k,c,grau,r,g}.outputs(4) MY_LIB{k,c,grau,r,g}.outputs(4) ...
                K_SVM{k,c,grau,r,g}.accuracy K_PG{k,c,grau,r,g}.accuracy PG_C{k,c,grau,r,g}.accuracy ...
                PG_N{k,c,grau,r,g}.accuracy SPG_es{k,c,grau,r,g}.accuracy SPG_AugLag{k,c,grau,r,g}.accuracy SPG_AugLag_rho{k,c,grau,r,g}.accuracy F_QP{k,c,grau,r,g}.accuracy ...
                F_AL{k,c,grau,r,g}.accuracy LIB{k,c,grau,r,g}.accuracy QP{k,c,grau,r,g}.accuracy QP_reg{k,c,grau,r,g}.accuracy LIBI{k,c,grau,r,g}.accuracy ...
                SMO{k,c,grau,r,g}.accuracy MY_LIB{k,c,grau,r,g}.accuracy ...
                K_SVM{k,c,grau,r,g}.balancedAcc K_PG{k,c,grau,r,g}.balancedAcc PG_C{k,c,grau,r,g}.balancedAcc ...
                PG_N{k,c,grau,r,g}.balancedAcc SPG_es{k,c,grau,r,g}.balancedAcc SPG_AugLag{k,c,grau,r,g}.balancedAcc SPG_AugLag_rho{k,c,grau,r,g}.balancedAcc F_QP{k,c,grau,r,g}.balancedAcc ...
                F_AL{k,c,grau,r,g}.balancedAcc LIB{k,c,grau,r,g}.balancedAcc QP{k,c,grau,r,g}.balancedAcc QP_reg{k,c,grau,r,g}.balancedAcc LIBI{k,c,grau,r,g}.balancedAcc ...
                SMO{k,c,grau,r,g}.balancedAcc MY_LIB{k,c,grau,r,g}.balancedAcc ...
                K_SVM{k,c,grau,r,g}.f1_score K_PG{k,c,grau,r,g}.f1_score PG_C{k,c,grau,r,g}.f1_score ...
                PG_N{k,c,grau,r,g}.f1_score SPG_es{k,c,grau,r,g}.f1_score SPG_AugLag{k,c,grau,r,g}.f1_score SPG_AugLag_rho{k,c,grau,r,g}.f1_score F_QP{k,c,grau,r,g}.f1_score ...
                F_AL{k,c,grau,r,g}.f1_score LIB{k,c,grau,r,g}.f1_score QP{k,c,grau,r,g}.f1_score QP_reg{k,c,grau,r,g}.f1_score LIBI{k,c,grau,r,g}.f1_score ...
                SMO{k,c,grau,r,g}.f1_score MY_LIB{k,c,grau,r,g}.f1_score ...
                K_SVM{k,c,grau,r,g}.ConfMatr K_PG{k,c,grau,r,g}.ConfMatr PG_C{k,c,grau,r,g}.ConfMatr ...
                PG_N{k,c,grau,r,g}.ConfMatr SPG_es{k,c,grau,r,g}.ConfMatr SPG_AugLag{k,c,grau,r,g}.ConfMatr SPG_AugLag_rho{k,c,grau,r,g}.ConfMatr F_QP{k,c,grau,r,g}.ConfMatr ...
                F_AL{k,c,grau,r,g}.ConfMatr LIB{k,c,grau,r,g}.ConfMatr QP{k,c,grau,r,g}.ConfMatr QP_reg{k,c,grau,r,g}.ConfMatr LIBI{k,c,grau,r,g}.ConfMatr ...
                SMO{k,c,grau,r,g}.ConfMatr MY_LIB{k,c,grau,r,g}.ConfMatr ...
                K_SVM{k,c,grau,r,g}.outputs(2) K_PG{k,c,grau,r,g}.outputs(2) PG_C{k,c,grau,r,g}.outputs(2) ...
                PG_N{k,c,grau,r,g}.outputs(2) SPG_es{k,c,grau,r,g}.outputs(2) SPG_AugLag{k,c,grau,r,g}.outputs(2) SPG_AugLag_rho{k,c,grau,r,g}.outputs(2) F_QP{k,c,grau,r,g}.outputs(2) ...
                F_AL{k,c,grau,r,g}.outputs(2) LIB{k,c,grau,r,g}.outputs(2) QP{k,c,grau,r,g}.outputs(2) QP_reg{k,c,grau,r,g}.outputs(2) LIBI{k,c,grau,r,g}.outputs(2) ...
                SMO{k,c,grau,r,g}.outputs(2) MY_LIB{k,c,grau,r,g}.outputs(2) ...
                ];
            
            aux_error = [aux_error;...
                K_SVM{k,c,grau,r,g}.Error K_PG{k,c,grau,r,g}.Error PG_C{k,c,grau,r,g}.Error ...
                PG_N{k,c,grau,r,g}.Error SPG_es{k,c,grau,r,g}.Error SPG_AugLag{k,c,grau,r,g}.Error F_QP{k,c,grau,r,g}.Error ...
                F_AL{k,c,grau,r,g}.Error LIB{k,c,grau,r,g}.Error QP{k,c,grau,r,g}.Error LIBI{k,c,grau,r,g}.Error ...
                SMO{k,c,g}.Error MY_LIB{k,c,g}.Error ...
                ];
            
            
        end
    end
    
    monta = [monta;full(aux)];
    aux = [];
    
    monta_error = [monta_error;full(aux_error)];
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

if STR.runMyLib,[result_MyLib] = organizingBestParameters(m_acc_MyLib,valoresCaixa,valoresGmKrnl);
else, result_MyLib=result_LIB; end

bestParameters = [result_K_SVM;result_K_PG;result_PG_C;result_PG_N;...
                  result_SPG_es;result_SPG_AugLag;result_SPG_AugLag_rho;result_F_QP;...
                  result_F_AL;result_LIB;result_QP;result_QP_reg;result_LIBI;...
                  result_SMO;result_MyLib];




end