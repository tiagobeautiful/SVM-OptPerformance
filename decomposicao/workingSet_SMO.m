function [i,j,Q,a,b] = workingSet_SMO(PARAMETERS,n,C,X_train,y_train,Q_diag,alphak,gradAlpha)

i = -1; j = -1;
Q = []; a = 0; b = 0;
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
t1 = find(y_train == 1 & alphak > 0);
t2 = find(y_train == -1 & alphak < C);
t = union(t1,t2);
[~,idx] = min(-y_train(t).*gradAlpha(t));
j = t(idx);
gradAlpha_min = -y_train(j)*gradAlpha(j);

% for t = 1:n
%     if (y_train(t) == -1 && alphak(t) < C) || (y_train(t) == 1 && alphak(t) > 0)
%         if -y_train(t)*gradAlpha(t) <= gradAlpha_min
%             j = t;
%             gradAlpha_min = -y_train(t)*gradAlpha(t);
%         end
%     end
% end

%%
if (gradAlpha_max-gradAlpha_min) < 1e-3
    i = -1; j =-1;
    return
end

K = makingKernelFunction(PARAMETERS,X_train,X_train([i,j],:));
Q = y_train([i;j])'.* (K.*y_train);

a = Q(i,1) + Q_diag(j,1) - 2*y_train(i)*y_train(j)*Q(i,2);
b = -y_train(i)*gradAlpha(i) + y_train(j)*gradAlpha(j);

