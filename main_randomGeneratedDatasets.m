%##########################################################################
%#
%# Battery test for solving the training SVM problem:
%#                  minimize       f(x) = 1/2 x^T*P*x - a^T*x
%#                  subjected to   b^T*x = c
%#                                lb <= x <= ub
%# where a = ones(n,1), b = label, c = 0, lb = zeros(n,1), ub = C*ones(n,1)
%#
%##########################################################################

clear
clc
warning off
addpath(genpath([pwd '\fcProblema']))

%% Initializing....
seed = 0;
rng(seed) %seed
definingParameters %creating STR e PARAMETERS structs

%% loop
startLoop = tic();
for d = 1:50 %****** DATASETS
    %% *** loading
    [DATASET{d,1},STR] = readDatasetGeneratedRandom(d,PARAMETERS,STR);
        
    for k = 1:size(PARAMETERS.type_kernel)  %****** Kernels functions chosen
        callTrainingScript
    end
    
    % cleaning the struct
    DATASET{d,1} = rmfield(DATASET{d,1},{'X_train', 'y_train', 'X_test', 'y_test'});

end
endLoop = toc(startLoop);

makeSimpliedTestResultsTable