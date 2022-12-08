from scipy.spatial import distance
from numpy.linalg import inv



'''

L - caracteristicas
M - classes.

inputs:
M vetores com as médias das classes (cada média é um
vetor 1 x L)

e uma matriz de covariância para o cálculo da métrica de Mahalanobis

vetor X que pertence a uma classe desconhecida.

outputs:

A distância euclidiana de X em relação a cada uma das classes;

A distância de Mahalanobis de X em relação a cada uma das classes;

A classificação de X de acordo com o classificador de mínima distância euclidiana;

A classificação de X de acordo com o classificador de mínima distância de Mahalanobis.
'''

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


covariancias = np.array(
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
dist1_euclidian = distance.euclidean(medias[0, :], padrao)
dist2_euclidian = distance.euclidean(medias[1, :], padrao)
print(dist1_euclidian, dist2_euclidian)


dist1_mahalanobis = distance.mahalanobis(medias[0, :], padrao, inv(covariancias))
dist2_mahalanobis = distance.mahalanobis(medias[1, :], padrao, inv(covariancias))
print(dist1_mahalanobis, dist2_mahalanobis)
#falta deixar generico e na disyancia
# de mahlanobis ele daria como classe 1 sendo a correta ao inves de 2

#para os classificadores de distancia minima:
#https://www.youtube.com/watch?v=saHURLZBEgk
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean
