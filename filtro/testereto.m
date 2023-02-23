function proib=testereto(F,tentativo,corrente,fig)
% proib=testereto(F,tentativo,corrente,fig)
% Esta função testa se um ponto tentativo é ou não proibido pelo filtro
% corrente (filtro original).
% Entrada:
%   F= filtro temporário
%       Os pares do filtro são da forma (f-alfa*h,(1-alfa)*h)
%   tentativo= ponto tentativo (f(x+), h(x+))
%   corrente= ponto corrente (f(xk)-alfa*h(xk), (1-alfa)*h(xk))
%   alfa= fator que define a margem
%   fig= mostra o gráfico se fig=1
% Saída:
%   proib=0 se o ponto for aceito pelo filtro
%   proib=1 se o ponto for proibido
%
% Gislaine 11/02/2011
ep=0;
if isempty(F)
    nf=0;
else
    nf=length(F(:,1));
end
%F=Ordena(F);
f0=corrente(1);
h0=corrente(2);
f=tentativo(1);
h=tentativo(2);
%=========================================================================
%Gráfico
%=========================================================================
if nf~=0 && fig==1
    oti=F(:,1);
    inf=F(:,2);
    clf(figure(1))
    hold on
    grid on
    abmin = min([oti;f]); abmax = max([oti;f]);
    ormin = 0; ormax = max([inf;h;3]);
    axis1 = [abmin-0.1*(abmax+1-abmin) abmax + 0.1*(abmax+1-abmin)] ;
    axis2 = [-0.1*ormax ormax + 0.1*ormax];
    abmin = axis1(1); abmax = axis1(2);
    ormin = axis2(1); ormax = axis2(2);
    axis([axis1 axis2])
    title('Filtro')
    xlabel('objetivo')
    ylabel('inviabilidade')
    plot(oti,inf,'or');
    
    for j=1:nf
        plot([oti(j) abmax],[inf(j) inf(j)],'--r')
        plot([oti(j) oti(j)],[inf(j) ormax],'--r')
    end
    pause
    plot(f0,h0,'om');
    plot(f,h,'ok');
    plot([f0 abmax],[h0 h0],'--m')
    plot([f0 f0],[h0 ormax],'--m')
    pause
    
    %=========================================================================
    %Teste
    %=========================================================================
    
    if f-f0>=ep && h>=h0
        proib=1;
    else
        j=nf;
        while j>0 && f-F(j,1)<ep
            j=j-1;
        end
        if j>0 && h-F(j,2)>=ep
            proib=1;
        else
            proib=0;
        end
    end
else
    if f-f0>=ep && h-h0>=ep
        proib=1;
    else
        j=nf;
        while j>0 && f-F(j,1)<ep
            j=j-1;
        end
        if j>0 && h-F(j,2)>=ep
            proib=1;
        else
            proib=0;
        end
    end
end
end
