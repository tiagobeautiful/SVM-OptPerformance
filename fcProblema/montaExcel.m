function montaExcel(caixa,SPG_SVM,LIBSVM,QP_semReg,QP_comReg,FILTRO,STR,DATASET)

for auxc = 1:length(caixa)
    c = caixa(auxc);
    saidaCompleta = [];
    for d = 1:size(STR.dataset,1) %****** DATASETS
        DATASET{d,1}.n_test = size(DATASET{d,1}.idx_teste,1);
                
        [TN_spg,TP_spg,FP_spg,FN_spg] = matrizDeConfusao(SPG_SVM{d,auxc});
        [TN_fil,TP_fil,FP_fil,FN_fil] = matrizDeConfusao(FILTRO{d,auxc});
        [TN_qpSR,TP_qpSR,FP_qpSR,FN_qpSR] = matrizDeConfusao(QP_semReg{d,auxc});
        [TN_qpCR,TP_qpCR,FP_qpCR,FN_qpCR] = matrizDeConfusao(QP_comReg{d,auxc});
        [TN_lib,TP_lib,FP_lib,FN_lib] = matrizDeConfusao(LIBSVM{d,auxc});
        
        saidaCompleta = [saidaCompleta;
            [DATASET{d,1}.n_train DATASET{d,1}.n_test DATASET{d,1}.p SPG_SVM{d,auxc}.saidas   SPG_SVM{d,auxc}.bStar   NaN                       SPG_SVM{d,auxc}.acuracia   NaN                          SPG_SVM{d,auxc}.acuracia_bal   TN_spg TP_spg FP_spg FN_spg];...
            [DATASET{d,1}.n_train DATASET{d,1}.n_test DATASET{d,1}.p FILTRO{d,auxc}.saidas    FILTRO{d,auxc}.bStar    NaN                       FILTRO{d,auxc}.acuracia    NaN                          FILTRO{d,auxc}.acuracia_bal    TN_fil TP_fil FP_fil FN_fil];...
            [DATASET{d,1}.n_train DATASET{d,1}.n_test DATASET{d,1}.p QP_semReg{d,auxc}.saidas QP_semReg{d,auxc}.bStar NaN                       QP_semReg{d,auxc}.acuracia NaN                          QP_semReg{d,auxc}.acuracia_bal TN_qpSR TP_qpSR FP_qpSR FN_qpSR];...
            [DATASET{d,1}.n_train DATASET{d,1}.n_test DATASET{d,1}.p QP_comReg{d,auxc}.saidas QP_comReg{d,auxc}.bStar NaN                       QP_comReg{d,auxc}.acuracia NaN                          QP_comReg{d,auxc}.acuracia_bal TN_qpCR TP_qpCR FP_qpCR FN_qpCR];...
            [DATASET{d,1}.n_train DATASET{d,1}.n_test DATASET{d,1}.p LIBSVM{d,auxc}.saidas    LIBSVM{d,auxc}.bStar    LIBSVM{d,auxc}.bStar_fcLib LIBSVM{d,auxc}.acuracia    LIBSVM{d,auxc}.acuracia_fcLIB LIBSVM{d,auxc}.acuracia_bal    TN_lib TP_lib FP_lib FN_lib];
            ];
        
    end
    
    xlswrite([STR.caminho '\results_datasetWithGradProj.xlsx'],saidaCompleta,['c=' num2str(c)],'C2')
end



end



function [TN,TP,FP,FN] = matrizDeConfusao(GERAL)
pos_neg = find(GERAL.order == -1); pos_pos = find(GERAL.order == 1);
TN = GERAL.MatrizConf(pos_neg,pos_neg);
TP = GERAL.MatrizConf(pos_pos,pos_pos);
FP = GERAL.MatrizConf(pos_neg,pos_pos);
FN = GERAL.MatrizConf(pos_pos,pos_neg);
end