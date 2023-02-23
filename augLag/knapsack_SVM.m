%Objective: apply Torrealba et al (2021) framework to solve the following problem
%       minimize   L(x,lambda,rho) = 1/2 x^T*P*x - a^T*x + lambda*(b^T*x - c) + rho/2*(b^T*x - c)^2
%       subject to   0 < x < C*e

function [GERAL] = knapsack_SVM(PARAMETERS)
%% initializing
epsilon = PARAMETERS.tolerance;             % *** stopping criteria tolerance
timeLimit = PARAMETERS.timeLimit;           % *** time limit tolerance
kmax = PARAMETERS.kmax;                     % max iteration

rho = 1;                                    % *** penalty parameter AL
lambda = 1;                                 % *** Lagrange multiplier LA
beta = 1;
k = 1;                                      % starting iterator
continuar = 1;                              % flag to end main loop
xk = zeros(PARAMETERS.n_train,1);% viable initial solution

%% Matrix and arrays of the problem
a = PARAMETERS.a; b = PARAMETERS.b;
lb = PARAMETERS.lb; ub = PARAMETERS.ub;
P = PARAMETERS.P;

%% Solving linear systems

startTime = tic();
R = chol(P);
q = R\(R'\a);
z = R\(R'\b);
clear R

% *** residual of the linear systems
GERAL.norm_Sistemas = [norm(P*q-a) norm(P*z-b)];

%% 
gamma = rho/(1+rho*(b'*z));
p1 = (beta*gamma*(b'*z) - 1)*z;
p2 = q + (- gamma*beta*b'*q)*z; % q + (rho*c - gamma*beta*b'*(q + rho*c*z))*z; %<- forma original, considerando o termo 'c'

%% loop
while continuar
    %% Step 1: define a solution for the uncontrained AL function
    bar_xk = p2 + lambda*p1;    

    %% Step 2: define a solution for the box constrained AL
    xk1 = max(lb, min(bar_xk ,ub) );

    %% Step 3: Refreshing Lagrange mult.
    cond1 = b'*xk1;
    lambda = lambda + rho*cond1;
    abs_cond1 = abs(cond1);
    
    %% Step 4: Stopping criteria
    cond2 = norm(xk1 - xk);
        
    if (abs_cond1 < epsilon && cond2 < epsilon)
        sucesso = 1;
        continuar = 0;
    elseif k > kmax 
        continuar = 0;
        sucesso = 2;
    elseif toc(startTime) > timeLimit
        continuar = 0;
        sucesso = 0;
    else
        k = k+1;
        xk = xk1;
    end
end
trainingTime = toc(startTime);

% forcing to zero when the component is less than tolerance
aux = find(xk1 <= PARAMETERS.toleranceSV);
xk1(aux) = 0;

%% Outputs
fval = PARAMETERS.majorValue*(0.5*xk1'*P*xk1 - sum(xk1));                                  %valor da funcao objetivo
index = find(xk1 > PARAMETERS.toleranceSV);                               %indices das amostra que serao vetores suporte

GERAL.outputs = [k-1,PARAMETERS.constant,fval,size(index,1),sucesso,cond1,cond2,NaN,NaN,PARAMETERS.buildStructTime,trainingTime];

GERAL.alpha = xk1;                                                       %solucao do problema dual
GERAL.index = index;
GERAL.lambda = lambda - rho*cond1;                                  %lambda da iteracao
GERAL.rho = rho;                                                    %parametro de penalidade da ultima iteracao
GERAL.q = q;
GERAL.z = z;

end