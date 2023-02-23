% Objective: apply Projected Gradient framework with Cauchy direction. To
% compute the feasible direction, we use the Gram-Schimidt process. The
% optimization problem is
% minimize   0.5 x'*A*x + b'*x
% subject to c'*x = 0
%            0 <= x <= C

function [GERAL] = ProjGrad_cauchy(PARAMETERS)

%% initializing
kmax = PARAMETERS.kmax;
timeLimit = PARAMETERS.timeLimit;
k=0;
sucesso = 0;
tol = 1e-3;
tol2 = 1e-10;
sair = 0;


%% Problem parameters
C = PARAMETERS.C;
% A = PARAMETERS.P;
b = -PARAMETERS.a;
c = PARAMETERS.b;
n = size(b,1);

x = rand(n,1);
[~,ind] = min(c);
x(ind) = 0;
x(ind) = -c'*x/c(ind); %obrigando c'*x = 0

x(abs(x-C) <= tol^2) = C; %salvaguarda: quando x é quase C, joguei em C
x(abs(x) <= tol^2) = 0;

ativas0 = find(x == 0);
ativasC = find(x == C);

%% loop
startTime = tic();
while sair == 0
    %% Step 1: compute direction
    g = PARAMETERS.P*x+b;
    daux = -g + (c'*g)/(c'*c)*c;
    d = -g;
    d(ativas0) = 0;
    d(ativasC) = 0;
    caux = c;
    caux(ativas0) = 0;
    caux(ativasC) = 0;
    d = d - (caux'*d)/(caux'*caux)*caux;
    d(abs(d) <= tol2) = 0; %safeguard
    
    
    if norm(d) >= tol
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
        lambda1 = (-caux'*gaux)/(caux'*caux);
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
        elseif k >= kmax
            sair = 1;
            sucesso = 2;
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
    
    k=k+1;
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
GERAL.norm_Sistemas = [NaN NaN];
