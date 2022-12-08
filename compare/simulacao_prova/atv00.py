import scipy.io
import pandas as pd
from scipy.spatial import distance
from numpy.linalg import inv
import numpy as np
from scipy.stats import multivariate_normal


def naive_bayes(padrao, medias, covariancias, priors):
    verossemlhanca1 = multivariate_normal.pdf(padrao, mean=medias[0, :], cov=covariancias[0, :, :])
    verossemlhanca2 = multivariate_normal.pdf(padrao, mean=medias[1, :], cov=covariancias[1, :, :])
    prob1 = verossemlhanca1 * priors[0]
    prob2 = verossemlhanca2 * priors[1]

    return (prob1, prob2)


def euclidean(medias, padrao):
    dist1_euclidian = distance.euclidean(medias[0, :], padrao)
    dist2_euclidian = distance.euclidean(medias[1, :], padrao)

    return (dist1_euclidian, dist2_euclidian)


def mahalanobis(medias, padrao, covariancias):
    dist1_mahalanobis = distance.mahalanobis(medias[0, :], padrao, inv(covariancias[0, :, :]))
    dist2_mahalanobis = distance.mahalanobis(medias[1, :], padrao, inv(covariancias[1, :, :]))

    return (dist1_mahalanobis, dist2_mahalanobis)



mat = scipy.io.loadmat('Dados.mat')
padroes = mat['padroes']
classes = mat['classes']

colunas = ['feature1', 'feature2', 'feature3', 'feature4', 'rotulo']
dados = pd.DataFrame(np.concatenate((padroes, classes), axis=1), columns=colunas)

m1_c0 = dados[dados['rotulo'] == -1]['feature1'].mean()
m2_c0 = dados[dados['rotulo'] == -1]['feature2'].mean()
m3_c0 = dados[dados['rotulo'] == -1]['feature3'].mean()
m4_c0 = dados[dados['rotulo'] == -1]['feature4'].mean()

m1_c1 = dados[dados['rotulo'] == 1]['feature1'].mean()
m2_c1 = dados[dados['rotulo'] == 1]['feature2'].mean()
m3_c1 = dados[dados['rotulo'] == 1]['feature3'].mean()
m4_c1 = dados[dados['rotulo'] == 1]['feature4'].mean()


medias = np.array(
    [
        [m1_c0, m2_c0, m3_c0, m4_c0],
        [m1_c1, m2_c1, m3_c1, m4_c1]
    ]
)


select = ['feature1', 'feature2', 'feature3', 'feature4']
covariancia0 = np.cov(dados[dados['rotulo'] == -1][select].T, ddof=0)
covariancia1 = np.cov(dados[dados['rotulo'] == 1][select].T, ddof=0)


covariancias = np.zeros((2, 4, 4))
covariancias[0, :, :] = covariancia0
covariancias[1, :, :] = covariancia1
priors = np.array([0.5, 0.5])


previsto = []
for linha in range(dados.shape[0]):
    padrao = np.array(dados.iloc[linha, 0:-1])
    prob0, prob1 = naive_bayes(padrao, medias, covariancias, priors)

    if prob0 > prob1:
        previsto.append(-1)
    else:
        previsto.append(1)



union = []
union.append(previsto)
union.append(list(classes.T[0]))
resultado = pd.DataFrame(np.array(union).T, columns=['previsto', 'real'])

VP = resultado[(resultado['previsto'] == 1) & (resultado['real'] == 1)].shape[0]
FP = resultado[(resultado['previsto'] == 1) & (resultado['real'] == -1)].shape[0]
VN = resultado[(resultado['previsto'] == -1) & (resultado['real'] == -1)].shape[0]
FN = resultado[(resultado['previsto'] == -1) & (resultado['real'] == 1)].shape[0]

sensibilidade = VP / (VP + FN)
especificidade = VN / (VN + FP)
acuracia = (VP + VN) / resultado.shape[0]

print(f"A sensibilidade é {sensibilidade}")
print(f"A especificidade é {especificidade}")
print(f"A acuracia é {acuracia}")


#mahalanobis
previsto = []
for linha in range(dados.shape[0]):
    padrao = np.array(dados.iloc[linha, 0:-1])
    prob0, prob1 = mahalanobis(medias, padrao, covariancias)

    if prob0 < prob1:
        previsto.append(-1)
    else:
        previsto.append(1)



union = []
union.append(previsto)
union.append(list(classes.T[0]))
resultado = pd.DataFrame(np.array(union).T, columns=['previsto', 'real'])

VP = resultado[(resultado['previsto'] == 1) & (resultado['real'] == 1)].shape[0]
FP = resultado[(resultado['previsto'] == 1) & (resultado['real'] == -1)].shape[0]
VN = resultado[(resultado['previsto'] == -1) & (resultado['real'] == -1)].shape[0]
FN = resultado[(resultado['previsto'] == -1) & (resultado['real'] == 1)].shape[0]

sensibilidade = VP / (VP + FN)
especificidade = VN / (VN + FP)
acuracia = (VP + VN) / resultado.shape[0]

print(f"A sensibilidade é {sensibilidade}")
print(f"A especificidade é {especificidade}")
print(f"A acuracia é {acuracia}")


'''
Nota-se que a acurácia de ambos é a mesma porém o classificador mahalanobis
possui um sensibilidade maior, sendo assim, se no contexto for mais relevante 
detectar com precisão quem realmente está doente mahalanobis é indicado, caso 
seja mais relevante a precisão em quem não está doente bayes deverá ser escolhido.
'''