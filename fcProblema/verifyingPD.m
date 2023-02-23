%Objective: verify with Hessian P is positive definite by a Cholesky
%decomposition. Adds "constant" at the principal diagonal to P become
%diagonal dominant

function [P,constant] = verifyingPD(P,n_train,timeLimit)

aux = 1; constant = 0;
tempo = tic();
while aux
    try
        [R] = chol(P);
        aux = 0;
        clear R
    catch
        if aux == 1
            constant = constant +1;
            P = P + spdiags(constant*ones(n_train,1),0,n_train,n_train);
        end
    end    
    
    if toc(tempo) > timeLimit*2
        fprintf('verifyingPD.m reached time limit. P matrix adds %d at principal \n',constant)
        aux = 0;
    end
end