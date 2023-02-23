%Objective: Creates matrix and arrays for the optimization problem
%                  minimize       f(x) = 1/2 x^T*P*x - a^T*x
%                  subjected to   b^T*x = c
%                                lb <= x <= ub
% where a = ones(n,1), b = label, c = 0, lb = zeros(n,1), ub = C*ones(n,1)

function [PARAMETERS] = makingStructs(PARAMETERS,DATASET)

% tic
%%
startTime = tic();
[PARAMETERS.n_train,PARAMETERS.p_train] = size(DATASET.X_train);

%% Arrays of the quadratic problem
PARAMETERS.a = ones(PARAMETERS.n_train,1);                           %array of ones
PARAMETERS.b = DATASET.y_train;                     %array of labels
PARAMETERS.c = 1e-8;                                %value of the intercept equality constraint
PARAMETERS.lb = zeros(PARAMETERS.n_train,1);                         %lower bound

%% Creating the Hessian
PARAMETERS.kernelX = makingKernelFunction(PARAMETERS,DATASET.X_train,[]);

Y = spdiags(DATASET.y_train,0,PARAMETERS.n_train,PARAMETERS.n_train);
P = Y*PARAMETERS.kernelX*Y;
PARAMETERS.P = sparse(P);

PARAMETERS.buildStructTime = toc(startTime);
