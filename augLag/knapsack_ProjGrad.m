%Objective: funcao implementada considerando a direcao oposta do gradiente,
%determinando um x tentativo e aplica o algoritmo AL-mochila para
% min  0.5*||x-xtentativo||^2
% s.a  b'x = 0
%      0 <= x <= ub 

function [GERAL] = knapsack_ProjGrad(PARAMETERS)

%% initializing
kmax = PARAMETERS.kmax;
timeLimit = PARAMETERS.timeLimit;           % *** time limit tolerance

sucesso = 0;
k = 0;
criterioParada = 1;

%% Problem parameters
a = PARAMETERS.a; b = PARAMETERS.b; %P = PARAMETERS.P;
lb = PARAMETERS.lb; ub = PARAMETERS.ub;
xk = zeros(size(a,1),1);
e = ones(size(a,1),1);


%% loop
startTime = tic();
while criterioParada
    
    %% Step 1: compute the direction
    g = PARAMETERS.P*xk - a;
    d = -g;
    
    %% Step 2: exact search
    t = - (d'*g)/(d'*PARAMETERS.P*d); %passo exato da quadratica, quando P definida positiva
    xten = xk  + t*d;
    
    %% Step 3: Knapsack algorithm (Torrealba et al 2021)
    [w,~,~,~,lamb] = knapsack_Torrealba(e,[],b,-b'*xten,lb-xten,ub-xten);
    xk1 = xten + w;
    
    %% Step 4: Stoppage criteria
    criterio1 = abs(b'*xk1);
    criterio2 = norm(xk1 - xk);

    if (criterio1 < 1e-5 && criterio2 < 1e-3 )
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

% forcing to zero when the component is less than tolerance
aux = find(xk1 <= PARAMETERS.toleranceSV);
xk1(aux) = 0;

%% outputs
fval = PARAMETERS.majorValue*(0.5*xk1'*PARAMETERS.P*xk1 - sum(xk1));
index = find(xk1 > PARAMETERS.toleranceSV);
GERAL.outputs = [k,PARAMETERS.constant,fval,size(index,1),sucesso,criterio1,criterio2,NaN,NaN,PARAMETERS.buildStructTime,trainingTime];

GERAL.alpha = xk1;
GERAL.index = index;
GERAL.lambda = lamb;
GERAL.norm_Sistemas = [NaN NaN];
