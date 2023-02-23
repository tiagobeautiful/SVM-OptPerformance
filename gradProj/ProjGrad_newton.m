% Objective: apply Projected Gradient framework with Newton direction. To
% compute the feasible direction, we use the Gram-Schimidt process. The
% optimization problem is
% minimize   0.5 x'*A*x + b'*x
% subject to c'*x = 0
%            0 <= x <= C

function [GERAL] = ProjGrad_newton(PARAMETERS)

%% initializing
timeLimit = PARAMETERS.timeLimit;
kmax = PARAMETERS.kmax;

k = 0;
tol = 1e-3;
tol2 = 1e-8;
sair = 0;

%% Problem parameters
C = PARAMETERS.C;
% A = PARAMETERS.P;
b = -PARAMETERS.a;
c = PARAMETERS.b;
n=size(b,1);

x = zeros(n,1);
ativas0 = find(x == 0);
ativasC = find(x == C);

%% loop
startTime = tic();
xstar = -PARAMETERS.P\b; %Newton system

while sair == 0
    %% Step 1: compute direction
    g = PARAMETERS.P*x+b;
    d = xstar-x;
    gaux = -g;
    d(ativas0) = 0;
    d(ativasC) = 0;
    gaux(ativas0) = 0;
    gaux(ativasC) = 0;
    caux = c;
    caux(ativas0) = 0;
    caux(ativasC) = 0;
    aux = caux'*caux;
    if aux > 0
        d = d - (caux'*d)/(caux'*caux)*caux;
        gaux = gaux - (caux'*gaux)/(caux'*caux)*caux;
    end
    if g'*d >= -tol2 %precisa olhar condição do ângulo etc!!!!!!!!!!!!
        d = gaux;
    end
    d(abs(d) <= tol2) = 0; %safeguard
    
    if norm(gaux) >= tol
        %% Step 2.1: exact search and compute new point
        alpha = -(d'*g)/(d'*PARAMETERS.P*d); %exact search
        for i = 1:n %find the maximum where we can go
            if d(i) > 0
                alpha = min(alpha,(C-x(i))/d(i));
            elseif d(i) < 0
                alpha = min(alpha,-x(i)/d(i));
            end
        end
        x = x + alpha*d;
        x(abs(x-C) <= tol2) = C; %safeguard
        x(abs(x) <= tol2) = 0;
        ativas0 = find(x == 0);
        ativasC = find(x == C);
    else
        %% Step 2.1: compute the generated KKT linear system
        gaux = g;
        gaux(ativas0) = 0;
        gaux(ativasC) = 0;
        if aux > 0
            lambda1 = (-caux'*gaux)/(caux'*caux);
        else
            lambda1 = 0;
        end
        if caux'*caux == 0
            lambda1 = 0;
        end
        lambda0 = g(ativas0)+lambda1*c(ativas0);
        lambdaC = -g(ativasC)-lambda1*c(ativasC);
        [min0,ind0] = min(lambda0);
        [minC,indC] = min(lambdaC);
        if isempty(min0)
            min0 = 0;
        end
        if isempty(minC)
            minC = 0;
        end
        
        %% Step 2.2: stopping criteria
        if min(min0,minC) >= 0
            sair = 1;
            sucesso = 1;
        elseif k > kmax
            sair = 1;
            sucesso = 2;
        elseif toc(startTime) > timeLimit
            sair = 1;
            sucesso = 0;
        elseif min0 < minC
            ativas0(ind0) = [];
        else
            ativasC(indC) = [];
        end
    end    
    
    if toc(startTime) > timeLimit
        sair = 1;
        sucesso = 0;
    end
    k = k +1;

end
trainingTime = toc(startTime);

% forcing to zero when the component is less than tolerance
aux = find(x <= PARAMETERS.toleranceSV);
x(aux) = 0;

%% outputs
fval = PARAMETERS.majorValue*(0.5*x'*PARAMETERS.P*x+b'*x);
index = find(x > PARAMETERS.toleranceSV);
GERAL.outputs = [k,PARAMETERS.constant,fval,size(index,1),sucesso,abs(c'*x),NaN,NaN,NaN,PARAMETERS.buildStructTime,trainingTime];

GERAL.alpha = x;
GERAL.index = index;
GERAL.norm_Sistemas = [norm(PARAMETERS.P*xstar+b) NaN];
