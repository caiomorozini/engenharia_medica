import itertools
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import preprocess as pre
from collections import Counter
from mpl_toolkits import mplot3d


mat = scipy.io.loadmat('Dados_ex2.mat')
#matrizes 25 x 20, padrões x caracteristicas
classe1 = mat['classe1']
classe2 = mat['classe2']

todos_dados = np.concatenate((classe1, classe2))
qtd_total_feaures = todos_dados.shape[1]

matriz_dados_normalizados = []
for i in range(qtd_total_feaures):
    matriz_dados_normalizados.append(
        list(pre.normaliza(list(todos_dados[:, i]))
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


qtd_feaures_escolher = 3
rotulos = np.array([0] * 25 + [1] * 25)
nomes_caracteristicas = [f"Caracteristica {i}" for i in range(qtd_total_feaures)]
res = selecao_vetorial(
    matriz_dados_normalizados,
    qtd_feaures_escolher,
    rotulos,
    nomes_caracteristicas
)

maior_j3 = 0
combinacao_com_maior_j3 = list(res.keys())[0]
for chave in res:
    j3_atual = res[chave]['j3']
    if j3_atual > maior_j3:
        maior_j3 = j3_atual
        combinacao_com_maior_j3 = chave


print(combinacao_com_maior_j3)

plt.figure()
ax = plt.axes(projection="3d")
#nesse caso fiz na mão mesmo pq lá em cima printei o valor das posições das caracteristas
#1 0 10 e sei que as classes estão juntas sendo as primeiras 25 de uma e as demais 25 da outra
#fiz isso por preguiça, mas se na hora da prova n estiver tão bonitinho das duas uma,
#ou terei q ordenar as classes p ficar assim ou terei q usar o metodo npwhere para achar a lista de
#colunas que tem a classe em questão, como feito na cinstução da função de seleção escalar.
x = matriz_dados_normalizados[1, 0:25]
y = matriz_dados_normalizados[0, 0:25]
z = matriz_dados_normalizados[10, 0:25]
ax.scatter3D(x, y, z, color='blue', label='classe1')
x = matriz_dados_normalizados[1, 25:50]
y = matriz_dados_normalizados[0, 25:50]
z = matriz_dados_normalizados[10, 25:50]
ax.scatter3D(x, y, z, color='red', label='classe2')
plt.legend()
plt.show()



#import json
#print(json.dumps(res, indent=2))





















