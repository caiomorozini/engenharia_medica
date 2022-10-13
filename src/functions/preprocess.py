
import numpy as np
import scipy.stats as st


def remoutliers(padroes,p,method='desvio'):
    #Encontra outlires baseando-se em dois métodos possiveis:
    #  method = 'desvio': mediana mais/menos p x desvio
    #  method = 'quartis': quartis  +/- p x intervalo entre quartis
    # padroes = numpy array de uma característica ( N x 1)
    # p = numero de desvios ou de intervalos entre quartis a ser empregado 
    # retorna lista com as posicoes dos outliers no array
    if method =='desvio':
        md=np.median(padroes)
        std=np.std(padroes,ddof=1)
        th1=md+p*std
        th2=md-p*std
    elif method=='quartis':
        q3, q1 = np.percentile(padroes, [75 ,25])
        iqr=q3-q1
        th1=q3+p*iqr
        th2=q1-p*iqr
    outliers=(padroes>th1) | (padroes<th2)
    outs=[i for i, val in enumerate(outliers) if val]
    return outs


def normaliza(dados,metodo='linear',r=1):
    #Realiza a normalizacao de um conjunto de padroes
    # dados = numpy array com padroes de uma caracteristica N x 1
    # metodo ='linear' : normalizacao linear (padrao)
    #        = 'mmx': limitada entre -1 e 1
    #        = 'sfm': rescala nao linear no intervalo 0 a 1
    # r = parametro do metodo sfm (padrao =1)
    #A função retorna os dados normalizados
    if metodo=='linear':
        M=np.mean(dados)
        S=np.std(dados,ddof=1)
        dadosnorm=(dados-M)/S
    elif metodo=='mmx':
        dadosnorm=2*dados/(np.max(dados)-np.min(dados))
        dadosnorm=dadosnorm - (np.min(dadosnorm)+1)
    elif metodo=='sfm':
        x=dados-np.mean(dados)
        x=-x/(r*np.std(dados))
        dadosnorm=1/(1+np.exp(x))        
    return dadosnorm

def preselec(dados1,dados2,alfa):
    #A função retorna os indices das características que apresentaram 
    #significatividade ("rel") e os p-values das características na 
    #distinção entre classes ("p")
    #Inputs:
    # - dados1 = array características x padrões da primeira classe
    # - dados2 = array características x padrões da segunda classe
    # - alfa = taxa de erro tipo I do teste (por exemplo, alfa=0.05)
    Ncarac,Npad=dados1.shape
    Ncarac2,Npad2=dados2.shape
    if Ncarac2!= Ncarac:
        print('Erro: matrizes devem ter o mesmo numero de caracteristicas!')
        return
    p=np.zeros(Ncarac)
    for i in range(Ncarac):
        s1=st.shapiro(dados1[i,:])
        s2=st.shapiro(dados2[i,:])
        if (s1[1]<0.05) | (s2[1]<0.05):
            res=st.ranksums(dados1[i,:],dados2[i,:])   
            p[i]=res.pvalue
            print('Aviso: normalidade rejeitada para a caracteristica nº '+ str(i+1))
        else:
            res=st.ttest_ind(dados1[i,:],dados2[i,:])
            p[i]=res.pvalue
    relevantes=(p<alfa)
    rel=[i for i,val in enumerate(relevantes) if val]
    return rel,p
