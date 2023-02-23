% Objective: do the framework of the numerical experiment

%% *** training phase
PARAMETERS.chosenKernel = PARAMETERS.type_kernel{k,1};
[RESULTS,GRIDSEARCH] = setParametersKernel(d,k,DATASET{d,1},PARAMETERS,RESULTS,STR);


%% *** test phase
%do test phase
[TESTPHASE,RESULTS] = testingPhase2(d,k,RESULTS,TESTPHASE,PARAMETERS,DATASET{d,1},STR);

%% *** outputs
%do outputs
save([STR.caminho '\results.mat'],'RESULTS','GRIDSEARCH','TESTPHASE','PARAMETERS','DATASET','STR','nameDS','columnKernel','columnMethods')
xlswrite([STR.caminho '\finals.xlsx'], RESULTS.tableTestPhase,'results','C2')

organizingOutputHeader

if STR.doGridSearch
    xlswrite([STR.caminho '\gridSearch_' PARAMETERS.type_kernel{k,1} '.xlsx'], RESULTS.tableGridSearch{d,k},STR.datasetAtual,'A3')
end