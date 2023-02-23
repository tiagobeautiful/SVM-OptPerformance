xlswrite([STR.caminho '\finals.xlsx'], header,'results','A1') %header
methodsUtilized = 15;

for m = 1:methodsUtilized
    nameDS{end+1,1} = STR.datasetAtual;
    columnMethods{end+1,1} = methods{m,1};
    columnKernel{end+1,1} = PARAMETERS.chosenKernel;
end

xlswrite([STR.caminho '\finals.xlsx'], nameDS,'results','A2') %dataset names
xlswrite([STR.caminho '\finals.xlsx'], columnMethods,'results','B2') %methods
xlswrite([STR.caminho '\finals.xlsx'], columnKernel,'results','C2') %kernel function