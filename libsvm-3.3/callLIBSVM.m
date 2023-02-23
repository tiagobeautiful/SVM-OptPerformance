function [GERAL,model] = callLIBSVM(PARAMETERS,DATASET)

trainData_X = DATASET.X_train;
trainData_y = DATASET.y_train;


% *** Building options
if strcmp(PARAMETERS.chosenKernel,'linear')
    PARAMETERS.options_libsvm = ['-q -s 0 -c ' num2str(PARAMETERS.C) ' -t 0'];    
elseif strcmp(PARAMETERS.chosenKernel,'poly')
    PARAMETERS.options_libsvm = ['-q -s 0 -c ' num2str(PARAMETERS.C) ...
        ' -t 1 -g ' num2str(PARAMETERS.gammaKernel) ' -r ' num2str(PARAMETERS.r) ' -d ' num2str(PARAMETERS.d)];
elseif strcmp(PARAMETERS.chosenKernel,'rbf')
    PARAMETERS.options_libsvm = ['-q -s 0 -c ' num2str(PARAMETERS.C) ...
        ' -t 2 -g ' num2str(PARAMETERS.gammaKernel)];
end

% *** Training
startTime = tic();
model = svmtrain(trainData_y, trainData_X, PARAMETERS.options_libsvm);
trainingTime = toc(startTime);

% *** Organizing LIBSVM solution
aux = model.sv_coef ./ trainData_y(model.sv_indices);
alpha_libsvm = zeros(size(trainData_X,1),1);
[i,j] = sort(model.sv_indices);
alpha_libsvm(i) =  aux(j);
index = find(alpha_libsvm > PARAMETERS.toleranceSV);

%fval
if isfield(PARAMETERS,'Poriginal')
    fval = PARAMETERS.majorValue*(0.5*alpha_libsvm'*PARAMETERS.Poriginal*alpha_libsvm - sum(alpha_libsvm));
else
    fval = PARAMETERS.majorValue*(0.5*alpha_libsvm'*PARAMETERS.P*alpha_libsvm - sum(alpha_libsvm));
end

% *** Caution: index between [0,STR.C]
aux = find(alpha_libsvm >= 0 & alpha_libsvm <= PARAMETERS.C);
viab = trainData_y'*alpha_libsvm;
if abs(viab) < PARAMETERS.tolerance && size(aux,1) == size(trainData_X,1)
    sucesso = 1;
else
    sucesso = 0;
end


GERAL.alpha = alpha_libsvm;
GERAL.index = index;
GERAL.outputs = [NaN,NaN,fval,size(index,1),sucesso,abs(viab),NaN,NaN,NaN,NaN,trainingTime];
GERAL.norm_Sistemas = [NaN NaN];
fprintf('kernel %s: rodei LIBSVM_original\n', PARAMETERS.chosenKernel)
