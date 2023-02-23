% Objective: computes the chosen kernel function

function K = makingKernelFunction(PARAMETERS,X,X2)
ker = PARAMETERS.chosenKernel;
gamma = PARAMETERS.gammaKernel;

switch ker
    case 'linear'
        %K(x_i,x_j) = <x_i,x_j>
        if ~isempty(X2)
            K = X * X2';
        else
            K = X * X'; %[n x p]
        end

    case 'poly'
%         K(x_i,x_j) = (gamma*<x_i,x_j> + r)^d
        if ~isempty(X2)
            K = (gamma*X*X2' + PARAMETERS.r).^PARAMETERS.d;
%             K = (X*X2').^gamma;
        else
            K = (gamma*X*X' + PARAMETERS.r).^PARAMETERS.d;
%             K = (X*X').^gamma;
        end

    case 'rbf'
        %K(x_i,x_j) = exp( - gamma * ||x_i-x_j||^2)  
        
        n1sq = sum(X'.^2);D = [];
        if isempty(X2)            %training phase
              a1 = n1sq';
              a2 = n1sq;
              a3 = 2*X*X';
              D = a1+a2-a3;
        else                      %test phase
            n2sq = sum(X2'.^2);
            a1 = n1sq';
            a2 = n2sq;
            a3 = 2*X*X2';
            D = a1+a2-a3;
        end
        K = exp(-gamma*D);
        
    otherwise
        error([ker ' kernel funtion not implemented!'])
end


% if isempty(X2)
%     PARAMETERS.kernelX = K;
% else
%     PARAMETERS.kernelTest = K;
% end