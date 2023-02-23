function [GERAL]=filtro(STR,x0,rodaQuadprog)
%-----------------------------------------------------
% General SQP original filter algorithm
% minimize an objective function subject to equality and inequality
% constraints
% Step is computed by SQP algorithm
% Subrotines:
% internoreto.m -> compute the step by SQP algorithm
% testereto.m: test the trial point by the filter criterion
% restauracaoPQS.m: restoration procedure
%steihaug.m: minimize a quadratic function subject to a ball
% atualizareto.m -> filter update




%% Parametros
timeLimit = STR.timeLimit;
STR.tol_otim=STR.tolerance;
STR.tol_viab=STR.tolerance;
nIter_int = 0;
F  = [];

alfa = 0.1; %0.01
xk = x0;
n = length(xk);
k = 1;
options=optimset('Algorithm','interior-point-convex','TolX',1e-8,'Tolfun',1e-8,'TolCon',1e-8,'Display','off');
fig = 0;
Hsp = speye(n);
k_max = 50;% STR.KmaxFiltro;
FF = [];


%% Inicializacao
l = STR.lb;
u = STR.ub;
% para eliminar essas variaveis do calculo da complementaridade
ufin=find(u < 1e10); lfin = find(l > -1e10);  indfin=intersect(ufin,lfin);

x = xk;
[f,g,h,~,cineq,ceq,Jineq,Jeq,Hf,STR]=calculos(x,STR);
fant = f;
H = Hf;

delta = 1e6*min(norm(g),h);
meq = length(ceq);
mineq = length(cineq);
lambda_eq = sparse(meq,1);  % multiplicadores de lagrange iniciais das restricoes de igualdade
lambda_ineq = sparse(mineq,1); % multiplicadores de lagrange iniciais das restricoes de caixa
ndc=10;

pare=0;

%% Loop
startTime = tic();
while  pare==0    
    corrente = [f-alfa*h (1-alfa)*h];
    Ftemp = F; % Temporary filter
    
    %======================================================================
    % Calcula xk+1 (chama o algoritmo interno)
    %======================================================================
    [x,f,g,h,hteste,ceq,Jeq,cineq,Jineq,sucesso,~,lambda_eq,lambda_ineq,~,STR,nIter_int] = internoreto(xk,H,corrente,Ftemp,f,g,Jeq,Jineq,ceq,cineq,...
        lambda_eq,lambda_ineq,delta,STR,rodaQuadprog,nIter_int);
    
    %======================================================================
    % Atualiza a hessiana do modelo quadratico
    %======================================================================
    H = Hf;
    
    %======================================================================
    % Atualizacao do filtro
    %======================================================================
    if f>=fant
        F = atualizareto(F,corrente,fig,FF);
    end
    fant = f;
    passo = norm(x-xk,inf);
    xk = x;
    
    if hteste < STR.tol_viab
        %==================================================================
        % Medida de estacionaridade
        %==================================================================
        beq = Jeq*g;
        if rodaQuadprog == 1
            [pz,~,ext1,output] = quadprog(Hsp,[],[],[],Jeq,beq,l-x+g,u-x+g,[],options);
            nIter_int = nIter_int + output.iterations;
        else
            [pz,ext1,ik,~] = LA_mochila(Hsp,[],Jeq',beq,l-x+g,u-x+g);
            nIter_int = nIter_int + ik;
        end
        
        if ext1==1
            dc=pz-g;
            ndc=norm(dc,inf);
        end
        
    end
    
    %======================================================================
    % KKT conditions
    %======================================================================
    Jac = [Jeq;Jineq];
    lambdak = [lambda_eq; lambda_ineq];
    normL = norm(g+Jac'*lambdak);
    complemen = norm(lambda_ineq(indfin).*(lambda_ineq(indfin)>1e-5).*cineq(indfin),inf);
    
    
    
    if (ndc < STR.tol_otim && hteste < STR.tol_viab) || (normL < STR.tol_otim && complemen < STR.tol_otim && hteste < STR.tol_viab)
        pare = 1;
    elseif toc(startTime) > timeLimit
        pare = 1;
    end
    %======================================================================
    
    
    delta = max(1,sqrt(2)*passo);
 
    if k>=k_max
        exitflag = 0;
        pare = 1;
    elseif sucesso~=0
        exitflag= 3;
%         saida =[k, f, h, hteste, ndc, normL, complemen, passo, delta, nIter_int,full(lambda_eq)];
        pare = 1;
    else
        k = k+1;
        exitflag = 1;    
    end
    
    
%     saida =[k, f, h, hteste, ndc, normL, complemen, passo, delta, nIter_int,full(lambda_eq)];
end
trainingTime = toc(startTime);

% index = find(xk1 > 1e-8);
% saidas = [saida(1),saida(2),sucesso,PARAMETROS.b'*xk1,NaN,size(FILTRO.index,1),tempo];
index = find(xk > STR.toleranceSV);
GERAL.outputs = [k,STR.constant,STR.majorValue*f,size(index,1),exitflag,abs(STR.b'*xk),hteste, ndc, normL,STR.buildStructTime,trainingTime];

GERAL.alpha = xk;
GERAL.index = index;
GERAL.lambda = full(lambda_eq);
GERAL.norm_Sistemas = [NaN NaN];
