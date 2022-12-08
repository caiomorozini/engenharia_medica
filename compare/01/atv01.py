import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

'''
Sobre os dados
Linha 1: medida de condução motora (CMAP) do nervo mediano (mCMAP); 
Linha 2: medida de condução sensorial (SNAP) do nervo mediano (mSNAP);
Linha 3: CMAP do nervo ulnar (uCMAP), 
Linha 4: SNAP do nervo ulnar (uSNAP)
'''

######## ITEM A - (INICIO) #########
mat = scipy.io.loadmat('Atividade 1.mat')
dados = mat['dados']
media_mCMAP = sum(dados[0])/len(dados[0])
media_mSNAP = sum(dados[1])/len(dados[1])
media_uCMAP = sum(dados[2])/len(dados[2])
media_uSNAP = sum(dados[3])/len(dados[3])
var_mCMAP = np.var(dados[0])
var_mSNAP = np.var(dados[1])
var_uCMAP = np.var(dados[2])
var_uSNAP = np.var(dados[3])

#U - nerbo unular
#M - nervo mediano
print(f"média mCMAP: {media_mCMAP}")
print(f"média mSNAP: {media_mSNAP}")
print(f"média uCMAP: {media_uCMAP}")
print(f"média uSNAP: {media_uSNAP}")

print(f"variância mCMAP: {var_mCMAP}")
print(f"variância mCMAP: {var_mSNAP}")
print(f"variância mCMAP: {var_uCMAP}")
print(f"variância mCMAP: {var_uSNAP}")
######## ITEM A - (FIM) #########


######## ITEM B - (INICIO) #########
plt.figure()

plt.subplot(2, 2, 1)
plt.hist(dados[0])
plt.xlabel('Velocidade (m/s)')
plt.ylabel('Qtd')

plt.subplot(2, 2, 2)
plt.hist(dados[1])
plt.xlabel('Velocidade (m/s)')

plt.subplot(2, 2, 3)
plt.hist(dados[2])
plt.xlabel('Velocidade (m/s)')

plt.subplot(2, 2, 4)
plt.hist(dados[3])
plt.xlabel('Velocidade (m/s)')

plt.savefig('distribuicoes_medias.jpg')


#O teste de Shapiro-Wilk testa a hipótese nula
# de que os dados foram extraídos de uma distribuição normal.

#A chance de rejeitar a hipótese nula quando verdadeira
# é próxima a 5% independente do tamanho da amostra.
print(f"teste de normalidade para mCMAP: {stats.shapiro(dados[0])}")
print(f"teste de normalidade para mSNAP: {stats.shapiro(dados[1])}")
print(f"teste de normalidade para uCMAP: {stats.shapiro(dados[2])}")
print(f"teste de normalidade para uSNAP: {stats.shapiro(dados[3])}")


#não posso rejeitar a normalidade pois tds os valores foram maiores que 5%
#portanto as distribuições possuem alta tendencia de ser normais
#https://www.youtube.com/watch?v=7jvDSvByQoM
#basicamente oq essa função cospe é um p valor e quanto maior esse p valor for
#mais confiantes estamos em aceitar h0 e quanto menor menos confientes sendo o limiar
#de decisão usado geralmente o alpha q nesse caso é 5%
#https://www.youtube.com/watch?v=8t9PlD7S5zk&list=PL7xT0Gz6G0-TfV-S6WiGDvIsZds6Pv_g8&index=3
# o segundo valor da tupla é o p-valor
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html


#erro tipo 1: rejeitar h0 quando ele é verdadeiro, probabilidade
# disso acontecer é dado por alpha, nesse caso 5%

#erro tipo 2: aceitar h0 quando o mesmo é falso, dado pela probabilidade beta
#https://www.youtube.com/watch?v=cpH2MSRuVp8&list=PL7xT0Gz6G0-TfV-S6WiGDvIsZds6Pv_g8&index=2
######## ITEM B - (FIM) #########


######## ITEM C - (INICIO) #########
mCMAP = 56
mSNAP = 52
uCMAP = 54
uSNAP = 61


plt.figure()

plt.subplot(2, 2, 1)
plt.hist(dados[0])
plt.axvline(x=mCMAP, color='red')
plt.xlabel('Velocidade (m/s)')
plt.ylabel('Qtd')

plt.subplot(2, 2, 2)
plt.hist(dados[1])
plt.axvline(x=mSNAP, color='red')
plt.xlabel('Velocidade (m/s)')

plt.subplot(2, 2, 3)
plt.hist(dados[2])
plt.axvline(x=uCMAP, color='red')
plt.xlabel('Velocidade (m/s)')

plt.subplot(2, 2, 4)
plt.hist(dados[3])
plt.axvline(x=uSNAP, color='red')
plt.xlabel('Velocidade (m/s)')

plt.savefig('paciente_vs_amostras.jpg')
######## ITEM C - (FIM) #########


#como encontrar as probabilidades de um dado evento usando a curva
#de gaus:
#https://www.youtube.com/watch?v=yhfODPGaMmY


######## ITEM D - (INICIO) #########
#loc - media
#scala - variancia

#A pergunta é: "probabilidade de um indivíduo saudável
#possuir valores de condutividade abaixo dos valores obtidos neste paciente"

#portanto minha h0: "Saudáveis tem medidas menores"
print(f"Probabilidade de um saudável ter mCMAP menor doque o valor do paceiente: {norm(loc=media_mCMAP, scale=var_mCMAP).cdf(mCMAP)}")
print(f"Probabilidade de um saudável ter mSNAP menor doque o valor do paceiente: {norm(loc=media_mSNAP, scale=var_mSNAP).cdf(mSNAP)}")
print(f"Probabilidade de um saudável ter uCMAP menor doque o valor do paceiente: {norm(loc=media_uCMAP, scale=var_uCMAP).cdf(uCMAP)}")
print(f"Probabilidade de um saudável ter uSNAP menor doque o valor do paceiente: {norm(loc=media_uSNAP, scale=var_uSNAP).cdf(uSNAP)}")
#se eu quisese a probabilidaed de ser maior dq o valor informado deveria fazer
#: 1 - resultado
#norm é usado para contruirmos uma normal com media e variancia personalizadas
#enquando cdf retorna a probabilidade dos dados serem menores dq o dado informado
#na cdf.

#como tds são maiores que 5% não posso rejeitar h0 (a hipotese de ser saudável),
######## ITEM D - (FIM) #########


######## ITEM E - (INICIO) #########
#https://numpy.org/doc/stable/reference/generated/numpy.cov.html
matriz_covariancia = np.cov(dados)

res = multivariate_normal.cdf(
    [mCMAP, mSNAP, uCMAP, uSNAP],
    mean=[media_mCMAP, media_mSNAP, media_uCMAP, media_uSNAP],
    cov=matriz_covariancia
)

print(f"Resultado de teste para todas as variaveis em conjunto {res}")

#pelo fato de ser menor dq 5% agora podemos rejeitar a hipotese dele ser saudável
#pois analisando as variaveis em conjunto ele se demonstra não saudável
######## ITEM E - (FIM) #########

#prob e estatistica pra macjine learn:
#https://www.youtube.com/watch?v=uFAILLxdajk





