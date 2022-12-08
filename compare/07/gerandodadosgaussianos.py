import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def gerandodadosgaussianos(medias, covariancias, N, priors, plotar=True, seed=0, angulo=[0, 0]):
    # Essa funcao gera um conjunto de dados simulados representando um
    # determinado numero de caracteristicas em um determinado numero de classes.
    # As classes possuem medias distintas e covariancias distintas. Os dados
    # seguem uma distribuicao gaussiana.
    # INPUT:
    # -medias =  classes x caracteristicas (matriz contendo as medias das
    #    caracteristica para cada classe)
    # -covariancia =  classes x caracteristicas x caracteristicas (matrizes de
    #    covariancia para cada classe)
    # -N = numero de padroes a serem gerados
    # -priors = classes x 1 (prior de cada classe: probabilidade de um padrao
    #    pertencer a cada classe)
    # - plotar = True (faz grafico - 2 ou tres dimensoes), False (nao faz grafico)
    # -seed = controle do seed na geracao de dados aleatorios
    # - angulo = angulo da visualizacao em caso de plot 3d.
    #
    # OUTPUT:
    # - dadossim=caracteristicas x padroes: dados simulados
    # - classessim= vetor contendo o numero da classe (de 0 ate C-1) de
    #     cada padrao simulado.

    #M: numero de classes
    #L: numero aracteristicas
    M, L = np.shape(medias)

    ######### testando principais erros que poderiam surgir no iput - (inicio) #######
    if np.size(covariancias, axis=0) != M | np.size(covariancias, axis=1) != L | np.size(covariancias, axis=2) != L :
        print('Erro: confira a dimensao dos seus dados de input.')
        return    

    if np.size(priors) != M:
        print('Erro: confira a dimensao dos priors.')
        return

    if np.sum(priors) != 1 :
        print('Erro: confira os valores dos priors.')
        return
    ######### testando principais erros que poderiam surgir no iput - (fim) #######

    #tupla contendo a qtd de valores que debem ser gerados em cada classe
    Ni = tuple(np.round(priors * N))
    #configurando a semente do gerador pseudoaleatorio
    np.random.seed(seed)      
    #gerando dados classe a classe
    for i in range(M):
       #checando se tds os autovalores da matriz de covariancia da classe i são positivos,
       # #pq n pode ter negativo?
       if np.all(np.linalg.eigvals(covariancias[i]) > 0) == False :
           print('Erro: confira os valores da covariancia.')

       #gerando uma matriz (caracteristicas x qtd dados)
       # com dados multivairados normais para a classe i
       #com as médias e covariancias para cada carateistca informadas
       x = np.random.multivariate_normal(medias[i], covariancias[i], size=int(Ni[i]))
       #caso seja a primeira classe
       if i == 0:
           dadossim = x.T
           classessim = np.zeros(int(Ni[i]),)
       #para as demais classes
       else:
           #concatene os valores ja gerados dadossim com oq acabamos de gerar x.T
           #ao lado doque jap foi gerado
           dadossim = np.concatenate((dadossim, x.T), axis=1)
           #analogo para as classes, mas para sabermos que esse ´outra classe usamos o + i
           #sendo assim termos os primeiros vetores com valor 0 o segundo 1 e assim por diante
           classessim = np.concatenate((classessim, np.zeros(int(Ni[i]),) + i), axis=0)


    ####### Parte Responsável pelo plot - (inicio) ###########
    if plotar: 
        if L == 2: #2 caracteristicas, plot 2d
            plt.figure()
            for i in range(M):                
                plt.plot(dadossim[0, classessim == i], dadossim[1, classessim == i], 'o', fillstyle='none')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            plt.show()
        elif L == 3:
            plt.figure()
            ax = plt.axes(projection='3d')
            for i in range(M):                
                ax.plot(dadossim[0, classessim == i], dadossim[1, classessim == i], dadossim[2, classessim == i], 'o', fillstyle='none')

            ax.view_init(angulo[0], angulo[1])
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            plt.show()
        else:
            print('Grafico é exibido apenas para 2 ou 3 dimensões')
        ####### Parte Responsável pelo plot - (fim) ###########

    return dadossim, classessim  