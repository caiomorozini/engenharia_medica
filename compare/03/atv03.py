import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from extraicarac import extraicarac
import scipy.io
import pandas as pd
import preprocess as pre
import numpy as np
from sklearn import metrics


#O arquivo contém 1179 trechos, de 30 segundos cada, armazenados na
#variável “SINAL” (trechos x tempo). Os trechos são amostrados a 100Hz.

#encontrará também um vetor “ESTAGIOS” (trechos x 1) contendo a
#classificação de cada trecho em estágios de sono
#(0 = vigília, 1 = estágio 1, 2 = estágio 2, 3
#= estágio 3, 4 = estágio 4, 5 = REM).

mat = scipy.io.loadmat('Dados.mat')
dados = mat['SINAL']
estagios = mat['ESTAGIOS']


bandas = {
    'delta 1': [0.5, 2.5],
    'delta 2': [2.5, 4],
    'teta 1': [4, 6],
    'teta 2': [6, 8],
    'alfa': [8, 12],
    'beta': [12, 20],
    'gama': [20, 45]
}


res = extraicarac(sin=dados, freq=100, bandas=bandas)

df = pd.DataFrame(res[0], columns=res[1])

estagio_renomeado = []
for estagio in estagios:
    if estagio == 0:
        estagio_renomeado.append('vigília')
    elif estagio == 1:
        estagio_renomeado.append('estágio 1')
    elif estagio == 2:
        estagio_renomeado.append('estágio 2')
    elif estagio == 3:
        estagio_renomeado.append('estágio 3')
    elif estagio == 4:
        estagio_renomeado.append('estágio 4')
    else:
        estagio_renomeado.append('REM')

df['estágios'] = estagio_renomeado


dados_vigilia = df[df['estágios'] == 'vigília']
dados_estagio_1 = df[df['estágios'] == 'estágio 1']
dados_estagio_2 = df[df['estágios'] == 'estágio 2']
dados_estagio_3 = df[df['estágios'] == 'estágio 3']
dados_estagio_4 = df[df['estágios'] == 'estágio 4']
dados_REM = df[df['estágios'] == 'REM']


###### Encontrando outliers #####
count_outliers_vigilia = 0
for feature in df.columns[0:-1]:
    posicoes_std_outliers_vigilia = pre.remoutliers(dados_vigilia[feature], 3, method='desvio')
    count_outliers_vigilia += len(posicoes_std_outliers_vigilia)

print(count_outliers_vigilia)


count_outliers_estagio_1 = 0
for feature in df.columns[0:-1]:
    posicoes_std_outdados_estagio_1 = pre.remoutliers(dados_estagio_1[feature], 3, method='desvio')
    count_outliers_estagio_1 += len(posicoes_std_outdados_estagio_1)

print(count_outliers_estagio_1)


count_outliers_estagio_2 = 0
for feature in df.columns[0:-1]:
    posicoes_std_outdados_estagio_2 = pre.remoutliers(dados_estagio_2[feature], 3, method='desvio')
    count_outliers_estagio_2 += len(posicoes_std_outdados_estagio_2)

print(count_outliers_estagio_2)

count_outliers_estagio_3 = 0
for feature in df.columns[0:-1]:
    posicoes_std_outdados_estagio_3 = pre.remoutliers(dados_estagio_3[feature], 3, method='desvio')
    count_outliers_estagio_3 += len(posicoes_std_outdados_estagio_3)

print(count_outliers_estagio_3)

count_outliers_estagio_4 = 0
for feature in df.columns[0:-1]:
    posicoes_std_outdados_estagio_4 = pre.remoutliers(dados_estagio_4[feature], 3, method='desvio')
    count_outliers_estagio_4 += len(posicoes_std_outdados_estagio_4)

print(count_outliers_estagio_4)

count_outliers_REM = 0
for feature in df.columns[0:-1]:
    posicoes_std_outdados_REM = pre.remoutliers(dados_REM[feature], 3, method='desvio')
    count_outliers_REM += len(posicoes_std_outdados_REM)

print(count_outliers_REM)


##### Extraindo caracteristicas irrelevantes vigilia x estagio 4 ######
caracteristicas = df.columns[0:-1]

valores_caracteristicas_estagio4 = []
valores_caracteristicas_vigilia = []
for caracteristica in caracteristicas:
    valores_caracteristicas_estagio4.append(dados_estagio_4[caracteristica])
    valores_caracteristicas_vigilia.append(dados_vigilia[caracteristica])


dados_vigilia_teste = np.array(valores_caracteristicas_vigilia)
dados_estagio_4_teste = np.array(valores_caracteristicas_estagio4)
res_relevancia = pre.preselec(dados_vigilia_teste, dados_estagio_4_teste, 0.05)
print(res_relevancia)

irrelevantes_index = set(range(0, len(caracteristicas))) - set(res_relevancia[0])

print("Irrelevantes: ")
for irrelevantes in irrelevantes_index:
    print(caracteristicas[irrelevantes])



##### Extraindo caracteristicas irrelevantes vigilia x rem ######
caracteristicas = df.columns[0:-1]

valores_caracteristicas_rem = []
valores_caracteristicas_vigilia = []
for caracteristica in caracteristicas:
    valores_caracteristicas_rem.append(dados_REM[caracteristica])
    valores_caracteristicas_vigilia.append(dados_vigilia[caracteristica])


dados_vigilia_teste = np.array(valores_caracteristicas_vigilia)
dados_estagio_rem_teste = np.array(valores_caracteristicas_rem)
res_relevancia = pre.preselec(dados_vigilia_teste, dados_estagio_rem_teste, 0.05)
print(res_relevancia)

irrelevantes_index = set(range(0, len(caracteristicas))) - set(res_relevancia[0])

print("Irrelevantes: ")
for irrelevantes in irrelevantes_index:
    print(caracteristicas[irrelevantes])



######### Aplicando a roc para vigilia e estado 4 em tds as caracteristicas ##########
for feature in df.columns[0:-1]:

    dados_vigilia_anorm = list(dados_vigilia[feature])
    dados_estagio_4_anorm = list(dados_estagio_4[feature])

    posicoes_std_outliers = pre.remoutliers(dados_vigilia_anorm, 3, method='desvio')
    dados_sem_outliers_vigilia_anorm = []
    for i in range(len(dados_vigilia_anorm)):
        if i not in posicoes_std_outliers:
            dados_sem_outliers_vigilia_anorm.append(dados_vigilia_anorm[i])

    posicoes_std_outliers = pre.remoutliers(dados_estagio_4_anorm, 3, method='desvio')
    dados_sem_outliers_estagio_4_anorm = []
    for i in range(len(dados_estagio_4_anorm)):
        if i not in posicoes_std_outliers:
            dados_sem_outliers_estagio_4_anorm.append(dados_estagio_4_anorm[i])

    scores = pre.normaliza(list(dados_sem_outliers_vigilia_anorm) + list(dados_sem_outliers_estagio_4_anorm), metodo='linear', r=1)

    
    labels = [0] * len(dados_sem_outliers_vigilia_anorm) + [1] * len(dados_sem_outliers_estagio_4_anorm)
    auc = metrics.roc_auc_score(labels, scores)
    if auc >= 0.5:
        print(f"{feature} - AUC: {auc}")
    else:
        print(f"{feature} - AUC: {1 - auc}")

