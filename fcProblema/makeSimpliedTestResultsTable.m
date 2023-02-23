
%% simplified results version
original = RESULTS.tableTestPhase;
i0 = 1;
i1 = 15;

novaTable = [];
aux = 1;
while aux
    novaTable=[novaTable;...
        original(i0,1:12),original(i0:i1,15)',original(i0:i1,17)',original(i0:i1,24)',original(i0:i1,29)'];
    
    if i1 >= size(original,1)
        aux = 0;
    else
        i0 = i1+1;
        i1 = i1 + 15;
    end
end

cabecalho = {'dataset','kernel function','n_train','n_test','attributes', 'classProportionPos_train','classProportionNeg_train','classProportionPos_test','classProportionNeg_test',...
        'gammaKernel','upperBound','degree (poly kernel only)','r (poly kernel only)','fval',NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,...
        'exitflag',NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,'totalTrainingTime',NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,...
        'accuracy',NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,};
segundaLinha = {NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,...
    'knapsack_SVM','knapsack_projGrad','projGrad_cauchy','projGrad_newton','SPG_exactSearch','SPG_augmentedLagrangian','SPG_augmentedLagrangian_rhoUpdate','filter_quadprog','filter_augmentedLagrangian',...
    'libsvm','quadprog','quadprog_reg','liblinear','SMO','My_libsvm',...
    'knapsack_SVM','knapsack_projGrad','projGrad_cauchy','projGrad_newton','SPG_exactSearch','SPG_augmentedLagrangian','SPG_augmentedLagrangian_rhoUpdate','filter_quadprog','filter_augmentedLagrangian',...
    'libsvm','quadprog','quadprog_reg','liblinear','SMO','My_libsvm',...
    'knapsack_SVM','knapsack_projGrad','projGrad_cauchy','projGrad_newton','SPG_exactSearch','SPG_augmentedLagrangian','SPG_augmentedLagrangian_rhoUpdate','filter_quadprog','filter_augmentedLagrangian',...
    'libsvm','quadprog','quadprog_reg','liblinear','SMO','My_libsvm',...
    'knapsack_SVM','knapsack_projGrad','projGrad_cauchy','projGrad_newton','SPG_exactSearch','SPG_augmentedLagrangian','SPG_augmentedLagrangian_rhoUpdate','filter_quadprog','filter_augmentedLagrangian',...
    'libsvm','quadprog','quadprog_reg','liblinear','SMO','My_libsvm',};

i1 = 1;
for d = 1:size(STR.dataset,1)
    for k = 1:size(PARAMETERS.type_kernel)  %****** Kernels functions chosen
        datasetAndKernel{i1,1} = STR.dataset{d,1};
        datasetAndKernel{i1,2} = PARAMETERS.type_kernel{k,1};
        i1 = i1+1;
    end
end

xlswrite([STR.caminho '\finalsSimplified.xlsx'], cabecalho,'results','A1') %header
xlswrite([STR.caminho '\finalsSimplified.xlsx'], segundaLinha,'results','A2') %methods
xlswrite([STR.caminho '\finalsSimplified.xlsx'], novaTable,'results','B3') %results
xlswrite([STR.caminho '\finalsSimplified.xlsx'], datasetAndKernel,'results','A3') %methods