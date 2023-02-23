function [GERAL] = classificationTask(GERAL,DATASET,PARAMETERS)

%% Train and test data
X_train = DATASET.X_train;
y_train = DATASET.y_train;
X_test = DATASET.X_test;
y_test = DATASET.y_test;


% Initializing
nSV = GERAL.index; %support vector index
alpha = GERAL.alpha; %dual quadratic solution

%% Lembrando....
%Lembrando que a funcao de decisao de SVM para uma variavel
%desconhecida/teste x eh dada por
%       f(x) = sign( (w^*)^T * phi(x)  +  b^* )                         (eq.1)
%onde:
%    w^* = X^T*(y .* alpha^*) = \sum_{s \in SV} y_s * alpha_s * phi(x_s)
%    b^* = y_j - (w^*)^T*x_j, para algum j \in {1,...,n} com alpha_j \in ]0,STR.C[
%
%Note que a relacao de w^* envolve uma funcao phi(.), que nem sempre eh
%conhecida. Entretanto, temos que (w^*)^T*phi(x) em (eq.1) forma a seguinte relacao
%    w^* = \sum_{s \in SV} y_s * alpha_s * K(x_s,x)                 (eq.2)
%sendo K(.,.) o kernel (implementado ate o momento: linear, rbf, poly)
%Assim, eh com a (eq. 2) que trabalharemos!

%Building the kernel matrix between train and test data
K = makingKernelFunction(PARAMETERS,X_train(nSV,:),X_test);

%Calculando (w^*)^T*X matricialmente, gerando um vetor com a qtd de amostras
%do conjunto de teste
wStar = ( (y_train(nSV,:) .* alpha(nSV,:))' * K )' ;


% Computing bStar by LIBSVM way
aux1 = find(GERAL.alpha(:,1) > 0);
aux2 = find(GERAL.alpha(:,1) < PARAMETERS.C);
i = intersect(aux1,aux2); %indices que estão entre ]0,STR.C[
if isempty(i)
    %encontrando M(alpha)
    aux0 = find(GERAL.alpha == 0 & y_train == -1);
    aux1 = find(GERAL.alpha == PARAMETERS.C & y_train == 1);
    uniao = union(aux0,aux1);
    
    M_alpha = max(y_train(uniao,1) .* (PARAMETERS.P(uniao,:)*GERAL.alpha - PARAMETERS.a(uniao,1) ) );
    if isempty(M_alpha), M_alpha = 0;end
    clear aux0 aux1 uniao
    
    %encontrando m(alpha)
    aux0 = find(GERAL.alpha == 0 & y_train == 1);
    aux1 = find(GERAL.alpha == PARAMETERS.C & y_train == -1);
    uniao = union(aux0,aux1);
    
    m_alpha = min(y_train(uniao,1) .* (PARAMETERS.P(uniao,:)*GERAL.alpha - PARAMETERS.a(uniao,1) ) );
    if isempty(m_alpha), m_alpha = 0;end
    %encontrando o ponto medio entre M_alpha e m_alpha
    rho = (M_alpha + m_alpha)/2;
else   
    numerador = sum( y_train(i,1) .* (PARAMETERS.P(i,:)*GERAL.alpha - 1) );
    rho = numerador/size(i,1);
end


%Decision function
fitted = (wStar - rho) ;
predicao = sign( fitted );
predicao(predicao == 0) = 1; %atribuindo um valor para quando acontecer do valor ser zero

%Confusion matrix
try
    [MatrizConf,order] = confusionmat(y_test,predicao);
    
    pos_neg = find(order == -1);
    pos_pos = find(order == 1);
    
    TN = MatrizConf(pos_neg,pos_neg);
    TP = MatrizConf(pos_pos,pos_pos);
    FP = MatrizConf(pos_neg,pos_pos);
    FN = MatrizConf(pos_pos,pos_neg);
    ConfMatr = [TP TN FP FN];
    
    acuracia = 100*(TN + TP)/(TN+TP+FP+FN);
    acuracia_bal = 100 * 0.5 * ( TP/(TP + FN) + TN/(TN + FP) );
    precisao = 100*(TP)/(TP+TN);
    recall = 100*(TP)/(TP+FN);
    f1_score = 2/(1/recall + 1/precisao);
catch
    predicao = NaN;
    acuracia = NaN;
    precisao = NaN;
    f1_score = NaN;
    acuracia_bal = NaN;
    recall = NaN;
    ConfMatr = [NaN NaN NaN NaN];
end

%% Organizando...
GERAL.accuracy = acuracia;
GERAL.balancedAcc = acuracia_bal;
GERAL.precision = precisao;
GERAL.recall = recall;
GERAL.f1_score = f1_score;
GERAL.ConfMatr = ConfMatr;
GERAL.predicao = predicao;
GERAL.valores_decisao = fitted;
GERAL.bStar = -rho;

