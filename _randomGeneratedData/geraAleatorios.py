import numpy as np
import pandas as pd
import os
from sklearn import datasets
import random
from random import randint


# funcao usando make_classification
def gerandoAleatorios(contProblemas,seed,class_sep,qtdAtributos,qtdAmostras):

    #writers para gerar o excel
    writer1 = pd.ExcelWriter(pasta + '\\Problem_'+ str(contProblemas) +'.xlsx',engine='xlsxwriter')


    # usando make_classification
    x,y = datasets.make_classification(n_samples=qtdAmostras,
                                n_features=qtdAtributos,
                                n_repeated=0,
                                class_sep=class_sep,
                                n_redundant=0,
                                random_state=seed)
    
    idx = np.asarray( np.where(y == 0))
    y[idx] = -1

    df = pd.DataFrame(x)
    df['label'] = y

    df.to_excel(writer1, index=False,header=False)
    df = []; x = []; y = []

    writer1.save()


# ----------------------------------------------------------------------------------------------------

pasta = os.path + '\\_aleatorios' # Nome arquivo
try:
    os.mkdir(pasta)
except:
    print('Folder already exists!')
#implementar baseado na quantidade de problemas que desejo, sorteando a semente, listaAmostras e listaAtributos

qtdproblemas = 50
listaAmostras = range(100,5000,50)                              # qtd amostras
listaAtributos = range(2,50)                                    # qtd atributos
classSep = [0.125,0.25,0.5,1,2,4,10]                            # separacao de classes para o make_classification

listSeed = [];listN = []
listP = [];listClassSep = []

# chamando make_classification
for contProblemas in range(1,qtdproblemas+1):
    
    seed         = randint(0, 100) #semente aleatoria
    class_sep    = random.choice(classSep)
    qtdAtributos = random.choice(listaAtributos)
    qtdAmostras  = random.choice(listaAmostras)

    gerandoAleatorios(contProblemas,seed,class_sep,qtdAtributos,qtdAmostras)

    listSeed.append(seed)
    listN.append(qtdAmostras)
    listP.append(qtdAtributos)
    listClassSep.append(class_sep)

d = {'Problem': range(1,qtdproblemas+1),'seed':listSeed, 'amostras': listN, 'atributos': listP, 'classSeparation': listClassSep}
dados = pd.DataFrame(data=d)

#writers para gerar o excel
# writer1 = pd.ExcelWriter(pasta + '\\CaracteristicasProblemas.xlsx',engine='xlsxwriter')
# dados.to_excel(writer1, index=False)
# writer1.save()