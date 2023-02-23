function [GERAL] = callQuadprog(PARAMETERS,flag_reg)

options=optimset('Algorithm','interior-point-convex','TolX',PARAMETERS.tolerance,'Tolfun',PARAMETERS.tolerance,'TolCon',PARAMETERS.tolerance,'Display','off');

if flag_reg == 0
    startTime = tic();
    [xk1,fval,sucesso,output,lambda] = quadprog(PARAMETERS.Poriginal,-PARAMETERS.a,[],[],...
        PARAMETERS.b',PARAMETERS.c,PARAMETERS.lb,PARAMETERS.ub,[],options);
    trainingTime = toc(startTime);
    
else
    startTime = tic();
    [xk1,fval,sucesso,output,lambda] = quadprog(PARAMETERS.P,-PARAMETERS.a,[],[],...
        PARAMETERS.b',PARAMETERS.c,PARAMETERS.lb,PARAMETERS.ub,[],options);
    trainingTime = toc(startTime);
end

GERAL.alpha = xk1;
GERAL.index = find(xk1 > PARAMETERS.toleranceSV);
GERAL.outputs = [output.iterations,NaN,PARAMETERS.majorValue*fval,size(GERAL.index,1),sucesso,abs(PARAMETERS.b'*xk1),NaN,NaN,NaN,PARAMETERS.buildStructTime,trainingTime];
GERAL.lambda = lambda.eqlin;
GERAL.norm_Sistemas = [NaN NaN];

clear xk1 fval sucesso output lambda trainingTime startTime