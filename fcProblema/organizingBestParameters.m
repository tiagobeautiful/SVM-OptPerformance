function [result] = organizingBestParameters(m_acc_GERAL,valoresCaixa,valoresGammaKernel)
    if isempty(valoresGammaKernel)
        valoresGammaKernel = ones(size(valoresCaixa));
    end
    
    [j ,i]  = find(m_acc_GERAL == max(max(m_acc_GERAL)) );
    
    if ~isempty(j) && ~isempty(i)
        result = [valoresGammaKernel(i(1)) valoresCaixa(j(1)) max(max(m_acc_GERAL)) m_acc_GERAL(j(1),i(1)) NaN NaN];
    else
        result = [NaN NaN NaN NaN NaN NaN];
    end

end
