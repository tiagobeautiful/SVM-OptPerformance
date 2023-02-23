%Objective: apply SPG framework to solve the quadratic constrained problem

function [GERAL] = SPG(PARAMETERS)

%% Initializing
epsilon = PARAMETERS.tolerance;
timeLimit = PARAMETERS.timeLimit;
kmax = PARAMETERS.kmax;

sucesso = 0;
k = 0;
lambda_min = 1e-30;
lambda_max = 1e30;
lambda = 1;
criterioParada = 1;

%% Problem parameters
a = PARAMETERS.a; y_treino = PARAMETERS.b;
% P = PARAMETERS.P;
n = size(a,1);
xk = zeros(n,1);

%% loop
startTime = tic();
while criterioParada
    
    %% Step 1: compute direction using Gram-Schimidt process
    g = PARAMETERS.P*xk - a;
    aux = (xk-lambda*g);
    p = aux - ((y_treino'*aux)/(n))*y_treino; %y'*y = n_train
    d = p - xk;
    
    %% Step 2: exact search
    t = - (d'*g)/(d'*PARAMETERS.P*d);
    xk1 = xk + t*d;
    
    t1= t; t2=t;
    i = find(xk1 < 0);
    if ~isempty(i)
        t1 = min(-xk(i,1)./d(i,1));
    end
    
    j = find(xk1 > PARAMETERS.C);
    if ~isempty(j)
        t2 = min( (PARAMETERS.C - xk(j,1))./d(j,1)) ;
    end
    
    xk1 = xk + min([t1,t2])*d;

    %% Step 3: spectral step
    s = xk1 - xk;
    y = PARAMETERS.P*(xk1 - xk);
    beta = s'*y;
    
    if beta <= 0
        lambda = lambda_max;
    else
        lambda = max(lambda_min, min(lambda_max, ((s'*s)/beta) ));
    end

    
    %% Step 4: stopping criteria
    criterio1 = norm(p - xk);
    criterio3 = abs(y_treino'*xk1);

    if (criterio1 < 1e-3  && criterio3 < epsilon )
        criterioParada = 0;
        sucesso = 1;
    elseif k >= kmax
        criterioParada = 0;
        sucesso = 2;
    elseif toc(startTime) > timeLimit
        criterioParada = 0;
        sucesso = 0;
    else
        xk = xk1;
        k=k+1;
    end
    
end
trainingTime = toc(startTime);

%forcando ser zero quando a componente for menor que 1e-8
aux = find(xk1 <= PARAMETERS.toleranceSV);
xk1(aux) = 0;

%% outputs
fval = PARAMETERS.majorValue*(0.5*xk1'*PARAMETERS.P*xk1-sum(xk1));
index = find(xk1 > PARAMETERS.toleranceSV);
GERAL.outputs = [k,PARAMETERS.constant,fval,size(index,1),sucesso,criterio3,criterio1,NaN,NaN,PARAMETERS.buildStructTime,trainingTime];

GERAL.alpha = xk1;
GERAL.index = index;
GERAL.norm_Sistemas = [NaN NaN];
