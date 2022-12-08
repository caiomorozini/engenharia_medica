import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import preprocess as pre


########## ITEM A e B - (INICIO) ###########
mat = scipy.io.loadmat('Dados1.mat')
dados = mat['sinal']
print(dados.shape)


posicoes = np.linspace(1, dados.shape[0], dados.shape[0])

plt.figure()
plt.scatter(posicoes, dados, marker="*")
plt.savefig('dados_puros.jpg')

#vetor com indices do vetor dados q continham outliers
posicoes_std_outliers = pre.remoutliers(dados, 3, method='desvio')
dados_outliers = []
for i in range(len(posicoes_std_outliers)):
    #Vetor com o repectivo dado que é um outlier
    dados_outliers.append(dados[posicoes_std_outliers[i]])
    #Somando mais um pois no gráfico a posição começa de 1
    #mas no indice dos vetors começa de 0, ou seja, precisa ser feita uma
    #correção pois tem shift a esquerda.
    posicoes_std_outliers[i] += 1


plt.figure()
plt.scatter(posicoes, dados, marker="*")
plt.scatter(posicoes_std_outliers, dados_outliers, marker="*", color='red')
plt.title('Outliers Metodo Desvio')
plt.savefig('outliers_metodo_desvio.jpg')


posicoes_quartis_outliers = pre.remoutliers(dados, 3, method='quartis')
dados_outliers = []
for i in range(len(posicoes_quartis_outliers)):
    dados_outliers.append(dados[posicoes_quartis_outliers[i]])
    posicoes_quartis_outliers[i] += 1


plt.figure()
plt.scatter(posicoes, dados, marker="*")
plt.scatter(posicoes_quartis_outliers, dados_outliers, marker="*", color='red')
plt.title('Outliers Metodo Quartis')
plt.savefig('outliers_metodo_quartis.jpg')
########## ITEM A e B - (FIM) ###########


########## ITEM C - (INICIO) ###########
'''
Sobre os dados

medidas histopatológicas de duas classes distintas de astrocitomas (alto grau e baixo grau).

São basicamente duas matrizes: uma para ske e outra para med
Sendo que as linhas são os valores e cada uma das duas colunas são as classes
'''
mat = scipy.io.loadmat('Dados2.mat')
dados_ske_c1 = mat['ske'][:, 0]
dados_ske_c2 = mat['ske'][:, 1]
dados_med_c1 = mat['med'][:, 0]
dados_med_c2 = mat['med'][:, 1]



plt.figure()
plt.scatter(dados_med_c1, dados_ske_c1, color='red', label='C1')
plt.scatter(dados_med_c2, dados_ske_c2, label='C2')
plt.xlabel('Média')
plt.ylabel('Obliquidade')
plt.legend()
plt.savefig('dados_nao_normalizados.jpg')

#Note que para outliers usamos cada classe individualmente, isso é
#devido ao fato de cada classe poder possuir ordem de grandeza diferente
#ex: uma vai de 0 a 5 e outra de 6 a 50 então 15 é um outlier para a prieira
#mas não é para a segunda por isso precisam ser availada separadamente.
#porém para a normalização, como estaremos mexendo na escala do grafico precisamos fazer
#em todos os dados juntos
dados_ske_concat = list(dados_ske_c1) + list(dados_ske_c2)
dados_med_concat = list(dados_med_c1) + list(dados_med_c2)

dados_ske_norm = pre.normaliza(dados_ske_concat, metodo='linear', r=1)
dados_med_norm = pre.normaliza(dados_med_concat, metodo='linear', r=1)


plt.figure()
plt.scatter(dados_med_norm[0:5], dados_ske_norm[0:5], color='red', label='C1')
plt.scatter(dados_med_norm[5:10], dados_ske_norm[5:10], label='C2')
plt.xlabel('Média')
plt.ylabel('Obliquidade')
plt.legend()
plt.savefig('dados_normalizados.jpg')
########## ITEM C - (FIM) ###########


########## ITEM D - (INICIO) ###########

#A função retorna os indices das características que apresentaram
#significatividade ("rel") e os p-values das características na
#distinção entre classes ("p")
#Inputs:
# - dados1 = array características x padrões da primeira classe
# - dados2 = array características x padrões da segunda classe
# - alfa = taxa de erro tipo I do teste (por exemplo, alfa=0.05)
dados_c1_concat = [dados_ske_c1, dados_med_c1]
dados_c2_concat = [dados_ske_c2, dados_med_c2]
dados1 = np.array(dados_c1_concat)
dados2 = np.array(dados_c2_concat)


res_relevancia = pre.preselec(dados1, dados2, 0.05)
print(res_relevancia)
########## ITEM D - (FIM) ###########


