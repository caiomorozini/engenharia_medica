import numpy as np
import pandas as pd


def ROC(valores, rotulos):

    '''
    :param valores: Todos os valores de uma dada característica
    :param rotulos: Os respectivos rótulos para cada valor
    :return: um vetor de verdadiros positivos e outro de falsos positivos
    '''

    #ATENÇÃO: A classe considerada negativa deve ser aquela que possui o menor
    # de todos os valores, caso essa condição n seja satisfeita irá ocorrer um espelhamento
    #no grafico e consequentemente em sua área. Digamos que a área era para ser 0.86, mas deu
    #0.14 note que é exatemente 1 - 0.86, você saberá se houve um espelhamento caso a sua
    #área tenha dado menor que 0.5 que seria o pior caso (com todas as caracteristicas sobrepostas)

    df_aux = pd.DataFrame(np.array([valores, rotulos]).T, columns=['valor', 'rotulo'])
    df_aux.sort_values('valor', inplace=True)
    df_aux.reset_index(inplace=True)

    VP_percent = []
    FP_percent = []
    thresholds = list(df_aux['valor'])
    thresholds = [min(thresholds) - 1] + thresholds + [max(thresholds) + 1]
    for valor in thresholds:
        VP = df_aux[(df_aux['valor'] >= valor) & (df_aux['rotulo'] == 1)].shape[0]
        FP = df_aux[(df_aux['valor'] >= valor) & (df_aux['rotulo'] == 0)].shape[0]
        VN = df_aux[(df_aux['valor'] < valor) & (df_aux['rotulo'] == 0)].shape[0]
        FN = df_aux[(df_aux['valor'] < valor) & (df_aux['rotulo'] == 1)].shape[0]

        FP_percent.append(FP / (FP + VN))
        VP_percent.append(VP / (VP + FN))



    return (FP_percent, VP_percent)


def AUC(VP, FP):

    '''
    :param VP: Vetor de valores verdadeiro positivos (eixo y, bases maior e menor trapézio)
    :param FP: Vetor de valores falso positivos (eixo x, altura do trapézio)
    :return: área abaixo da curva
    '''

    area = 0
    for i in range(len(VP) - 1):
        area += (VP[i] + VP[i + 1]) * (abs(FP[i + 1] - FP[i])) / 2

    return area
