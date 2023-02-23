%Objective: apply SPG framework to solve the following problem
%       minimize   L(x,lambda,rho) = 1/2 x^T*P*x - a^T*x + lambda*(b^T*x - c) + rho/2*(b^T*x - c)^2
%       subject to   0 < x < C*e

function [GERAL] = SPG_AL(PARAMETERS)

%% Initializing
kmax = PARAMETERS.kmax;
timeLimit = PARAMETERS.timeLimit;
epsilon = PARAMETERS.tolerance;             % *** stopping criteria tolerance

sucesso = 0;
k = 0;
lambda_min = 1e-30;
lambda_max = 1e30;
lambda = 1;
criterioParada = 1;
rho = 1;
mltLk = 1;

%% Problem parameters
a = PARAMETERS.a; b = PARAMETERS.b;
Pmod = PARAMETERS.P + rho*b*b';
n=size(a,1);

xk = zeros(n,1);

%% loop
startTime = tic();
while criterioParada
   
    %% Step 1: Compute the direction
    g = Pmod*xk -a + mltLk*b;
    proj = min(max(PARAMETERS.lb,xk-lambda*g),PARAMETERS.ub);
    d = proj - xk;

    %% Step 2: Exact search
    t = - (xk'*Pmod*d +(mltLk*b-a)'*d)/(d'*Pmod*d);
    
    t1= t; t2=t;
    barx = xk+t*d;
    
    i = find(barx < 0);
    if ~isempty(i)
        t1 = min(-xk(i,1)./d(i,1));
    end
    
    j = find(barx > PARAMETERS.C);
    if ~isempty(j)
        t2 = min( (PARAMETERS.C - xk(j,1))./d(j,1)) ;
    end
    xk1 = xk + min([t1,t2])*d;
      
    %% Step 3: refresh Lagrange multiplier and the spectral step
    cond1 = b'*xk1;
    mltLk1 = mltLk + rho*cond1;
    
    % spectral step
    s = xk1 - xk;
    y = Pmod*(xk1 - xk) + (mltLk1 - mltLk)*b;
    beta = s'*y;
    if beta <= 0
        lambda = lambda_max;
    else
        lambda = max(lambda_min, min(lambda_max, ((s'*s)/beta) ));
    end
        
    %% Step 4: stopping criteria
    criterio1 = abs(cond1);
    criterio2 = norm(proj - xk);

    if (criterio2 < 1e-2 && criterio1 < epsilon )
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
        mltLk = mltLk1;
        k=k+1;
    end
    
end
trainingTime = toc(startTime);

% forcing to zero when the component is less than tolerance
aux = find(xk1 <= PARAMETERS.toleranceSV);
xk1(aux) = 0;

%% outputs
fval = PARAMETERS.majorValue*(0.5*xk1'*PARAMETERS.P*xk1-sum(xk1));
index = find(xk1 > PARAMETERS.toleranceSV);
GERAL.outputs = [k,PARAMETERS.constant,fval,size(index,1),sucesso,criterio1,criterio2,NaN,NaN,PARAMETERS.buildStructTime,trainingTime];

GERAL.alpha = xk1;
GERAL.index = index;
GERAL.lambda = mltLk1;
GERAL.norm_Sistemas = [NaN NaN];

