function [i,j,Q,conc] = workingSet_libsvm(PARAMETERS,tau,n,C,X_train,y_train,Q_diag,alphak,gradAlpha)
i = -1;
j = -1;
Q = [];
conc = 0;
gradAlpha_max = -Inf; gradAlpha_min = Inf;

%% select i
t1 = find(y_train == 1 & alphak < C);
t2 = find(y_train == -1 & alphak > 0);
t = union(t1,t2);
[~,idx] = max(-y_train(t).*gradAlpha(t));
i = t(idx);
gradAlpha_max = -y_train(i)*gradAlpha(i);

% for t = 1:n
%     if (y_train(t) == 1 && alphak(t) < C) || (y_train(t) == -1 && alphak(t) > 0)
%         if -y_train(t)*gradAlpha(t) >= gradAlpha_max
%             i = t;
%             gradAlpha_max = -y_train(t)*gradAlpha(t);
%         end
%     end
% end

%% select j
if i ~= -1
    K = makingKernelFunction(PARAMETERS,X_train,X_train(i,:));
    Qi = y_train(i)*K.*y_train;
    clear K
else
    return;
end

obj_min = Inf;
t1 = find(y_train == 1 & alphak > 0);
t2 = find(y_train == -1 & alphak < C);
t = union(t1,t2);
b = gradAlpha_max + y_train(t).*gradAlpha(t);
b(b<=0) = NaN;
a = Qi(i,1) + Q_diag(t,1) - 2*y_train(i)*y_train(t).*Qi(t,1);
a(a<=0) = tau;
conta = -(b.^2)./a;
[~,idx1] = min(conta);

j = t(idx1);
conc = a(idx1);
obj_min = conta(idx1);
gradAlpha_min=-y_train(j)*gradAlpha(j);

% obj_min = Inf;
% for t = 1:n
%     if (y_train(t) == 1 && alphak(t) > 0) || (y_train(t) == -1 && alphak(t) < C)
%         b = gradAlpha_max + y_train(t)*gradAlpha(t);
%         
%         if (-y_train(t)*gradAlpha(t) <= gradAlpha_min)
%             gradAlpha_min = -y_train(t)*gradAlpha(t);
%         end
%         
%         if b > 0
%             %ver isso aqui
%             a = Q_diag(i,1) + Q_diag(t,1) - 2*y_train(i)*y_train(t)*Qi(t,1);
%             
%             if a <= 0
%                 a = tau;
%             end
%             
%             conta = -b^2/a;
%             if (conta <= obj_min)
%                 j = t;
%                 conc = a;
%                 obj_min = conta;
%             end
%         end
%     end
% end

if (gradAlpha_max-gradAlpha_min) < 1e-3
    i = -1; j =-1;
    return
end

K = makingKernelFunction(PARAMETERS,X_train,X_train(j,:));
Qj = y_train(j)*K.*y_train;
Q = [Qi Qj];


