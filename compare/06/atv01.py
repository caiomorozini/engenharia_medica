import numpy as np
from gerandodadosgaussianos import gerandodadosgaussianos

######### Gerando dados para testar implementação função - inicio #########

medias = np.array(
    [
        [-6, 6, 6],
        [6, 6, 6]
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
        [0.3, 1, 1],
        [1, 9, 1],
        [1, 1, 9]
    ]
)

#preenchendo a covariancia da segunda classe
covariancias[1, :, :] = np.array(
    [
        [0.3, 1, 1],
        [1, 9, 1],
        [1, 1, 9]
    ]
)

priors = np.array([0.5, 0.5])
qtd_dados = 400
#função que gera dados multivariados em distribuição gausiana
#input:
#médias para cada caracteristica de cada classe ex: [-6, 6, 6] é o vetor de médias da classe 1
#sendo -2 a média da caracteristica 1, o 6 da 2 e assim por diante
#qtd_dados - qtd de dados que devem ser gerados
#plotar - indica se quremos ver o plot dos dados (somente para 2 ou 3 dimensões)
#priors - probabilidade de um padrão pertencer a cada classe, ou seja,
# qtd de padrões gerados para
# cada classe se [0.6, 0.4] serão 60% da primeira classe e 40% da segunda
#seed - configura o numeri pseudoaleatório para que possamos comprara resultados
#angulo - angulação do plot

#onput:
#matriz de caracteristicas x padrões dos dados gerados
#classe para cada padrão gerado sendo que os indices dessa lista correspondem
# respectivamente a coluna
#da matriz de dados.
dados, classes = gerandodadosgaussianos(
    medias,
    covariancias,
    qtd_dados,
    priors,
    plotar=False,
    seed=0,
    angulo=[20, 120]
)
######### Gerando dados para testar implementação função - fim #########


def PCA(dados, m):

    '''
    :param dados: matriz de LXN caracteristicas X padrões
    :param m: quantidade de features desejadas no novo espaço

    :return: tupla contendo:
        * os autovalores em ordem descrescente com seus indices correspondedo a
        cada coluna dos autovetores
        * autovetores matriz LXL (caracteristica por caracteristica)
        * dados_novo_espaco: matriz m x N depois de aplicada a PCA (redução do espaço de L
        para m e retirada de dependência linear)
        * erro_quadratico_medio: demonstra a porcentagem que essas features tranformadas
        representam de todo o espaço
    '''

    matriz_covariancia = np.cov(dados, ddof=0)
    #Retorna os autovalores e autovetores de uma matriz.
    #Retorna dois objetos, uma matriz 1-D contendo os autovalores de
    # a e uma matriz ou matriz quadrada 2-D
    # dos autovetores correspondentes (em colunas).
    # ou seja autovalores = [1, 2, 3]
    #autovetores:
    #               [
    #                   [0 2 6],
    #                   [0 3 1],
    #                   [2 2 4],
    #               ]
    #sendo cada coluna um vetor dessa forma 1 seria o autovalor de 0 0 2
    # 2 seria de 2 3 2 e 3 de 6 1 4
    #Os autovalores estão em ordem crescente
    #A coluna é o autovetor normalizado correspondente ao autovalor .
    autovalores, autovetores = np.linalg.eigh(matriz_covariancia)
    autovalores = autovalores[::-1]
    autovetores = autovetores[:, ::-1]

    #embora a função cuspa os autovetores nas colunas para a matempatica funcionar
    #precisamos usar eles nas linhas por isso fazemos a tranposição, assim temos:
    # A(m x L) * D(L x N) = NE(m X N)
    #ssim continuamos com os mesmos padrões mas reduzimos o espaço de L caracteristicas
    #para m, com uma operação matempatica que garante a não linearidade entre elas, para
    #testar basta tirar a covariancia das caracteristicas e observar que o resltado é uma
    #matriz diagonal onde a diaginal são exatamente os autovalores.
    dados_novo_espaco = autovetores[:, 0:m].T @ dados
    erro_quadratico_medio = 1 - (sum(autovalores[0:m]) / sum(autovalores))


    return (autovalores, autovetores, dados_novo_espaco, erro_quadratico_medio)


autovalores, autovetores, dados_novo_espaco, erro_quadratico_medio = PCA(dados, m=2)
print(autovalores.shape)
print(autovetores.shape)
print(dados_novo_espaco.shape)
print(erro_quadratico_medio)
print(np.cov(dados_novo_espaco, ddof=0))
print(autovalores)



#sobre oq são autovalores e autovetores:
#https://www.youtube.com/watch?v=3UzV21Ak3uc
#https://www.youtube.com/watch?v=4stXqEmAc1s

#como pca é usada e para que serve:
#https://www.youtube.com/watch?v=p4bvCFygfW0
#https://www.youtube.com/watch?v=u8th43VOyCw
#https://www.youtube.com/watch?v=9-E2D8gTSA8


#sobre diagonalizaçao a partir de autovalores e autovetores:
#https://www.youtube.com/watch?v=uXc0n-hngAM





