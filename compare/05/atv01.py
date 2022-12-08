import itertools
import numpy as np
import scipy.io
import pandas as pd
import preprocess as pre
from collections import Counter


mat = scipy.io.loadmat('Dados 1.mat')
dados_figadoadiposo = pd.DataFrame(mat['figadoadiposo'])
dados_figadoadiposo.columns = ['média', 'std', 'ske', 'curtose']
dados_figadoadiposo['Rotulo'] = [1] * dados_figadoadiposo.shape[0]
dados_figadocirrotico = pd.DataFrame(mat['figadocirrotico'])
dados_figadocirrotico.columns = ['média', 'std', 'ske', 'curtose']
dados_figadocirrotico['Rotulo'] = [0] * dados_figadocirrotico.shape[0]

todos_dados = pd.concat([dados_figadoadiposo, dados_figadocirrotico])
todos_dados.reset_index(inplace=True, drop=True)

qtd_total_feaures = todos_dados.shape[1] - 1
nomes_features = list(todos_dados.columns)[0:-1]

matriz_dados_normalizados = []
for i in range(qtd_total_feaures):
    matriz_dados_normalizados.append(
        list(pre.normaliza(list(todos_dados.iloc[:, i]))
                      )
    )


matriz_dados_normalizados = np.array(matriz_dados_normalizados)



def selecao_vetorial(dados, qtd_feaures_escolher, rotulos, nomes_caracteristicas):

    '''
    :param dados: matriz caracteristicas X padrões normalizada

    :param qtd_feaures_escolher: inteiro que representa a qtd de features
    que queremos no espaço

    :param rotulos: vetor com rótulos para cada padrão (faz math com as colunas
    da matriz de dados). É importante que os rótulos sejam 0, 1, 2 e assim por diante,
    não precisa estar ordenado.

    :param nomes_caracteristicas: lista com os nomes das caracteristicas (math linha a
    linha com a matriz dados)

    :return: Dicionario onde a primeira chave representa a combinação de features a segunda o jn
    e por fim o valor desse jn para essa combinação de features, algo como:

    "std, curtose": {
        "j1": 1.302417291214537,
        "j2": 2.098029262305948,
        "j3": 1.549014631152974
    }
    '''

    # (numero total de caracteristicas, combinacao (pares, trincas etc))
    # https://docs.python.org/3/library/itertools.html#itertools.combinations
    #gerando as combinações possiveis entre as features que possuímos
    combinacoes_possiveis = list(
        itertools.combinations(
            range(dados.shape[0]),
            qtd_feaures_escolher
        )
    )

    #Contabilizadno a qtd total de padrões e qts existem por classe
    contagem_rotulos = Counter(rotulos)
    qtd_total_dados = sum(contagem_rotulos.values())
    qtd_rotulos = len(contagem_rotulos.keys())

    resultados = {}
    for combinacao in combinacoes_possiveis:
        #seleciona as features que serão analisada atualemente em conjunto
        aux = []
        #Armazena os nomes das features analizadas atualmente
        nomes_carac_atuais = []
        for pos_feature in combinacao:
            aux.append(list(dados[pos_feature, :]))
            nomes_carac_atuais.append(nomes_caracteristicas[pos_feature])

        aux = np.array(aux)
        sm = np.cov(aux, ddof=0)

        #vare classe por classe para calcular a covariancia de suas caracteristicas
        sw = np.zeros((qtd_feaures_escolher, qtd_feaures_escolher), dtype='float')
        for i in range(qtd_rotulos):
            #lista com as colunas que possuem o rotulo i: list(np.where(rotulos == i)[0])
            cov_classe_i = np.cov(aux[:, list(np.where(rotulos == i)[0])], ddof=0)
            sw += (contagem_rotulos[i] / qtd_total_dados) * cov_classe_i


        j1 = np.trace(sm) / np.trace(sw)
        j2 = np.linalg.det(np.linalg.inv(sw) @ sm)
        j3 = np.trace(np.linalg.inv(sw) @ sm) / qtd_feaures_escolher


        chave = ", ".join(nomes_carac_atuais)
        valores = {'j1': j1, 'j2': j2, 'j3': j3}
        resultados[chave] = valores

    return resultados


qtd_feaures_escolher = 2
rotulos = np.array([1] * 10 + [0] * 10)
nomes_caracteristicas = ['média', 'std', 'ske', 'curtose']
res = selecao_vetorial(
    matriz_dados_normalizados,
    qtd_feaures_escolher,
    rotulos,
    nomes_caracteristicas
)

import json
print(json.dumps(res, indent=2))
















'''
Esse trecho foi só para testar as funçoes de calcular sw sm e sb para ver se 
estava td em ordem
sm = np.cov(matriz_dados_normalizados, ddof=0)

#essa parte de baixo ainda não me agrada muito pois está fixo em duas classes
#pretendo generalizar para n classes
#acredito que posso marter o padrão de receber em ondem, ou seja, primeira parte é td da classe 1
#logo após vem a segunda classe e assim por diante, manter os dados de cada classe
#juntos.
qtd_dados_c0 = todos_dados[todos_dados['Rotulo'] == 0].shape[0]
qtd_dados_c1 = todos_dados[todos_dados['Rotulo'] == 1].shape[0]
qtd_total_dados = qtd_dados_c0 + qtd_dados_c1


cov_c1 = np.cov(matriz_dados_normalizados[:, 0:qtd_dados_c1], ddof=0)
cov_c0 = np.cov(matriz_dados_normalizados[:, qtd_dados_c1:qtd_dados_c1 + qtd_dados_c0], ddof=0)
sw = (qtd_dados_c0 / qtd_total_dados) * cov_c0 + (qtd_dados_c1 / qtd_total_dados) * cov_c1

sb = sm - sw

print(sw, sm, sb)'''




#A baixo está o código testando para o probelam q ele pediu mas sem fazer uma função
#generica, funciona apenas para duas classes
'''pos_feature1 = 0
pos_feature2 = 1
qtd_feaures = 2
matriz_dados_normalizados_duas_features = np.array([list(matriz_dados_normalizados[pos_feature1, :]),  \
                                          list(matriz_dados_normalizados[pos_feature2, :])])
print(matriz_dados_normalizados_duas_features)

sm = np.cov(matriz_dados_normalizados_duas_features, ddof=0)

qtd_dados_c0 = todos_dados[todos_dados['Rotulo'] == 0].shape[0]
qtd_dados_c1 = todos_dados[todos_dados['Rotulo'] == 1].shape[0]
qtd_total_dados = qtd_dados_c0 + qtd_dados_c1


cov_c1 = np.cov(matriz_dados_normalizados_duas_features[:, 0:qtd_dados_c1], ddof=0)
cov_c0 = np.cov(matriz_dados_normalizados_duas_features[:, qtd_dados_c1:qtd_dados_c1 + qtd_dados_c0], ddof=0)
sw = (qtd_dados_c0 / qtd_total_dados) * cov_c0 + (qtd_dados_c1 / qtd_total_dados) * cov_c1

sb = sm - sw

j1 = np.trace(sm) / np.trace(sw)
j2 = np.linalg.det(np.linalg.inv(sw) @ sm)
j3 = np.trace(np.linalg.inv(sw) @ sm) / qtd_feaures
print(j1, j2, j3)'''




