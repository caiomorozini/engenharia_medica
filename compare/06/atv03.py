import numpy as np
import scipy.io
import time


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




def SVD(dados, m):

    '''

    :param dados: matriz de LXN caracteristicas X padrões
    :param m: quantidade de features desejadas no novo espaço

    :return: tupla contendo:
        * os autovalores em ordem descrescente com seus indices correspondedo a
        cada linha dos autovetores
        * autovetores matriz LXL (caracteristica por caracteristica)
        * dados_novo_espaco: matriz m x N depois de aplicada a PCA (redução do espaço de L
        para m e retirada de dependência linear)
        * erro_quadratico_medio: demonstra a porcentagem que essas features tranformadas
        representam de todo o espaço
    '''




    dados_norm = np.zeros_like(dados)
    for i in range(dados.shape[0]):
        dados_norm[i, :] = dados[i, :] - np.mean(dados, axis=1)[i]


    u, s, v = np.linalg.svd(dados_norm)
    #u - contem os autovetores da matriz de correlação de caractersticas, usa dimensão então deve ser
    # L x L nesse caso 2000 x 2000
    #s - contem a raiz quadrada dos autovalores
    #v - contem os autovetores da matriz de correlação de padões, usa dimensão então deve ser
    # N x N nesse caso 100 x 100
    #u sempre é usada para cacular o novo espaço, é a pca propriamete dita
    #mas nem sempre ele é caclulada primeiramente, as vezes v é matematicamente mais facil de achar
    #e depois u é caclulado a partir dela, o contrário tbm se aplica. por isso a svd é mais
    #rápdio q a pca crua, ela pega atalhaos matematicos

    autovetores = u.T
    autovalores = s ** 2
    #S(m x L) * (L x N) = D(m x N)
    #pagando o autovetor inteiro, mas apenas alguns autovetores (seleção de caracteristicas)
    #note que os indices estão um pouco diferentes da PCA pois aqui fiz a transposta antes
    dados_novo_espaco = autovetores[0:m, :] @ dados_norm
    erro_quadratico_medio = 1 - (sum(autovalores[0:m]) / sum(autovalores))


    return (autovalores, autovetores, dados_novo_espaco, erro_quadratico_medio)



'''
Sobre os dados:

100 padrões com alta dimensionalidade (2000). 
Caracteristicas X Padrões
'''

mat = scipy.io.loadmat('Dados_exercicio3.mat')
dados = mat['X']




m = 2
inicio = time.time()
autovalores_SVD, autovetores_SVD, dados_novo_espaco_SVD, erro_quadratico_medio_SVD = SVD(dados, m)
fim = time.time()
print(f"Erro quadratico médio SVD {erro_quadratico_medio_SVD}")
print(f"Tempo {fim - inicio}")
print(np.cov(dados_novo_espaco_SVD, ddof=0))
print(autovalores_SVD)


inicio = time.time()
autovalores_PCA, autovetores_PCA, dados_novo_espaco_PCA, erro_quadratico_medio_PCA = PCA(dados, m)
fim = time.time()
print(f"Erro quadratico médio PCA {erro_quadratico_medio_PCA}")
print(f"Tempo {fim - inicio}")
print(np.cov(dados_novo_espaco_PCA, ddof=0))
print(autovalores_PCA)





















