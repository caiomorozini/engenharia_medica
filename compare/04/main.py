import numpy as np
import scipy.io
import pandas as pd
import scipy.stats
from ROC import ROC, AUC
import matplotlib.pyplot as plt

#sobre feature selection
#https://www.youtube.com/watch?v=0bPBxDhvqlI

def FDR(vet1, vet2):
    '''

    :param vet1: todos os valores do rótulo A
    :param vet2: todos os valores do rótulo B
    :return: indice de FDR entre os valores
    '''

    return ((np.mean(vet1) - np.mean(vet2)) ** 2) / (np.var(vet1) + np.var(vet2))


def selecao_escalar(func, features, rotulos, k, alpha=1, beta=1):
    '''

    :param func: critério usado no score auc ou fdr
    :param features: matriz onde as colunas representam as caracteristicas
    e as linhas os seus respectivos valores
    :param rotulos: os repectivos rótulos para cada linha da matriz de features
    :param k: quantidade de n melhores features que devem ser selecionadas
    :param alpha: valor de 0 a 1 que indica o peso do critério func na seleção da feature
    :param beta: valor de 0 a 1 que indica o peso da correlação na seleção da feature
    :return: vetor com os indices das features selecionadas como as k melhores
    '''

    
    qtd_caracteristicas = features.shape[1]

    features_A = features[np.where(rotulos == 0)]
    features_B = features[np.where(rotulos == 1)]

    #Calculando o score de todas as features com o critério puro (sem desconto de correlação)
    pre_score = []
    #vetor que irá armazenar os indices das features selecionadas
    #na ordem em que foram escolhidas
    features_escolhidas_index = []
    for i in range(qtd_caracteristicas):
        if func == 'fdr':
            score = FDR(features_A[:, i], features_B[:, i])
        elif func == 'auc':
            valores = list(features[:, i])
            FP, VP = ROC(valores, list(rotulos))
            score = AUC(VP, FP)
            
            if score < 0.5:
                score = 1 - score 

        pre_score.append(score)


    index_select = pre_score.index(max(pre_score))
    features_escolhidas_index.append(index_select)

    #caso k seja diferente de 1 iremos calcular para achar as n melhores features restantes
    for kn in range(1, k):
        #receberá o score de cada Mk para seleção da nova kn
        score_selecao = []
        for i in range(qtd_caracteristicas):
            #calcula apenas para as features que não foram escolhidas
            if i not in features_escolhidas_index:
                #levando em consideração o quão correlacionada/redundate a features testada no momento
                #está em relação as já selecionadas
                correlacao = 0
                for feature_escolhida_index in features_escolhidas_index:
                    correlacao += abs(scipy.stats.pearsonr(features[:, feature_escolhida_index], features[:, i])[0])


                score_ajustado = alpha * pre_score[i] - beta * correlacao / len(features_escolhidas_index)
                score_selecao.append(score_ajustado)
            else:
                #caso já tenhaos escolhido a feature colocamos um valor super negativo apenas
                #para garantir que essa n iremos selecionar novamente e que não precisemos
                #lidar com o shift do veto.
                #ex: começa com 4: 0 1 2 3
                #seleciona 1 ficaremos com: 0 1 2 (teriamos que mapear cada um
                # desses inices aos seus originais)
                score_selecao.append(-999)


        index_select = score_selecao.index(max(score_selecao))
        features_escolhidas_index.append(index_select)


    return features_escolhidas_index




mat = scipy.io.loadmat('Dados 1.mat')
dados_figadoadiposo = pd.DataFrame(mat['figadoadiposo'])
dados_figadoadiposo.columns = ['média', 'std', 'ske', 'curtose']
dados_figadoadiposo['Rotulo'] = [1] * dados_figadoadiposo.shape[0]
dados_figadocirrotico = pd.DataFrame(mat['figadocirrotico'])
dados_figadocirrotico.columns = ['média', 'std', 'ske', 'curtose']
dados_figadocirrotico['Rotulo'] = [0] * dados_figadocirrotico.shape[0]

todos_dados = pd.concat([dados_figadoadiposo, dados_figadocirrotico])
todos_dados.reset_index(inplace=True)

k = 2
func = 'fdr'
alpha = 0.2
beta = 0.8
caracteristicas_escohidas = selecao_escalar(func,
                np.array(todos_dados.iloc[:, 1:-1]),
                np.array(todos_dados.iloc[:, todos_dados.shape[1] - 1]),
                k,
                alpha=alpha,
                beta=beta)

caracteristicas = ['média', 'std', 'ske', 'curtose']

print(f"As {k} melhores caracteristicas com alpha={alpha} e beta={beta} usando {func} como criterio são:")
for caracteristica in caracteristicas_escohidas:
    print(caracteristicas[caracteristica])


