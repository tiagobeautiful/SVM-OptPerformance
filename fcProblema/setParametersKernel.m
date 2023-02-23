function [RESULTS,GRIDSEARCH] = setParametersKernel(d,kernel,DATASET,PARAMETERS,RESULTS,STR)

GRIDSEARCH = [];
RESULTS.tableGridSearch{d,kernel} = [];
RESULTS.tableGridSearchError{d,kernel} = [];

for m = 1:15
    
    %computing gammaKernel when the dataset is scaled
    if contains(STR.datasetAtual,'scale')
        variancia = mean(var(DATASET.X_train)); %mean of the collumns
        gammaKernel = 1/(DATASET.p * variancia);
    else
        gammaKernel = 1/DATASET.p;
    end
    
    
    if strcmp(PARAMETERS.type_kernel{kernel,1},'rbf')          % Gaussian/RBF kernel
        RESULTS.bestParameters{d,kernel}(m,:) = [gammaKernel 1 NaN NaN NaN NaN];
        
    elseif strcmp(PARAMETERS.type_kernel{kernel,1},'linear')   % Linear kernel
        RESULTS.bestParameters{d,kernel}(m,:) = [NaN 1 NaN NaN NaN NaN];

    elseif strcmp(PARAMETERS.type_kernel{kernel,1},'poly')     % Polynomial kernel
        RESULTS.bestParameters{d,kernel}(m,:) = [gammaKernel 1 NaN NaN 3 0];
    end

end