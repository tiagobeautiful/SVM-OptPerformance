function [GERAL] = LIBSVM_decomposition(DATASET,PARAMETERS)
%% parametros
kmax = 1000000;
tau = 1e-12;
timeLimit = PARAMETERS.timeLimit;

k = 0;
%% vetores e matrizes do problema decomposto
n = DATASET.n_train;
y_train = DATASET.y_train;
alphak = zeros(DATASET.n_train,1) ;
gradAlpha = -ones(DATASET.n_train,1);
C = PARAMETERS.C;
criterio = 1;

startTime = tic();
Q_diag = [];
for t = 1:n
    Q_diag(t,1) = makingKernelFunction(PARAMETERS,DATASET.X_train(t,:),[]);
end

while criterio
    k = k+1;
    %% Working set selection
    [i,j,Q,a] = workingSet_libsvm(PARAMETERS,tau,n,C,DATASET.X_train,y_train,Q_diag,alphak,gradAlpha);
    
    if i == -1
        exitflag = 1;
        break
    elseif toc(startTime) > timeLimit 
        exitflag = 0;
        break
    elseif k > kmax
        exitflag = 2;
        break
    end
        
    %% preparing....
%     a = Q(i,1) + Q_diag(j) - 2*y_train(i)*y_train(j)*Q(i,2);
    if a <= 0
        a = tau;
    end
    b = -y_train(i)*gradAlpha(i) + y_train(j)*gradAlpha(j);
    
    %% update alpha
    oldAi = alphak(i); oldAj = alphak(j);
    alphak(i) = oldAi + y_train(i)*b/a;
    alphak(j) = oldAj - y_train(j)*b/a;
    
    %% project alpha back to the feasible region
    soma = y_train(i)*oldAi + y_train(j)*oldAj;
    if alphak(i) > C
        alphak(i) = C;
    end
    if alphak(i) < 0
        alphak(i) = 0;
    end 
    alphak(j) = y_train(j)*(soma - y_train(i)*alphak(i));
    
    if alphak(j) > C
        alphak(j) = C;
    end
    if alphak(j) < 0
        alphak(j) = 0;
    end 
    alphak(i) = y_train(i)*(soma - y_train(j)*alphak(j));
    
    %% update gradient
    deltaAlphai = alphak(i) - oldAi;
    deltaAlphaj = alphak(j) - oldAj;
    gradAlpha = gradAlpha + sum([deltaAlphai deltaAlphaj] .* Q,2);
%     for t = 1:n
%         gradAlpha(t) = gradAlpha(t) + Q(t,1)*deltaAlphai + Q(t,2)*deltaAlphaj;
%     end
    
end
trainingTime = toc(startTime);

index = find(alphak > PARAMETERS.toleranceSV);
fval = PARAMETERS.majorValue*(gradAlpha'*alphak);
GERAL.outputs = [k,NaN,fval,size(index,1),exitflag,abs(PARAMETERS.b'*alphak),NaN, NaN, NaN,0,trainingTime];
GERAL.alpha = alphak;
GERAL.index = index;
GERAL.lambda = NaN;
GERAL.norm_Sistemas = [NaN NaN];