from gerandodadosgaussianos import gerandodadosgaussianos
import numpy as np
from scipy.stats import multivariate_normal

#sobre classificador baysiano:
#https://www.youtube.com/watch?v=Rq_hXHrdkbc
#https://www.youtube.com/watch?v=Bk2mSIMw_XE
#https://www.youtube.com/watch?v=8zAKWEOdGsg

'''
L - caracteristucas
M - classes

inputs:
M vetores com as média das classes (cada média é um vetor 1 x L) - ok

M matrizes com as covariâncias das classes (cada covariância é uma matriz L x L) - ok

lista com M números contendo os priors de cada classe. - ok

um vetor x que pertence a uma classe desconhecida. - ok

outputs:
A probabilidade de o padrão pertencer a cada classe, dado que o padrão é x;
A classificação de x de acordo com o classificador bayesiano.
'''

############### ITEM A - (INICIO) ###################
medias = np.array(
    [
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]
)

#gerando o shape da matriz de covariancia para cada classe
#2 - pq teremos duas classes cada uma em uma linha
#os demais 3 e 3 são a qtd de caracteristicas pois representaram a matriz de
#covariancia das caractersticas de cada classe
covariancias = np.zeros((2, 3, 3))
#preenchendo a covariancia da primeira classe
covariancias[0, :, :] = np.array(
    [
        [0.8, 0.01, 0.01],
        [0.01, 0.2, 0.01],
        [0.01, 0.01, 0.02]
    ]
)

covariancias[1, :, :] = np.array(
    [
        [0.8, 0.01, 0.01],
        [0.01, 0.2, 0.01],
        [0.01, 0.01, 0.02]
    ]
)


priors = np.array([0.5, 0.5])
padrao = np.array([0.1, 0.5, 0.1])


#para aplicar a gaussiana
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
def naive_bayes(padrao, medias, covariancias, priors):
    verossemlhanca1 = multivariate_normal.pdf(padrao, mean=medias[0, :], cov=covariancias[0, :, :])
    verossemlhanca2 = multivariate_normal.pdf(padrao, mean=medias[1, :], cov=covariancias[1, :, :])
    prob1 = verossemlhanca1 * priors[0]
    prob2 = verossemlhanca2 * priors[1]

    return (prob1, prob2)


print(naive_bayes(padrao, medias, covariancias, priors))





