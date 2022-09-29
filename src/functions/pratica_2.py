'''
Disciplina de Engenharia Médica Aplicada
Graduação em Engenharia Biomédica
Instituto de Ciência e Tecnologia da UNIFESP
Prof. Dr. Adenauer Casali
Atividade Prática - aula 08 (2022)
'''

import numpy as np
import scipy.fft as fft

def extraicarac(sin,freq,bandas):
  #Extrai características estatísticas e espectrais de um conjunto de sinais de mesma dimensão temporal.
  #Inputs:
  #   - sin = numpy array (num de sinais x tempo)
  #   - freq = frequencia de amostragem dos sinais (Hz)
  #   - bandas = dicionario com a informação das bandas de frequencia a serem extraídas. 
  #              Formato:{'nome da banda (string)':[freqinicial, freqfinal]}
  #              Exemplo: 
  #              bandas={'delta 1':[0.5,2.5],'delta 2':[2.5,4],'teta 1':[4,6],'teta 2':[6,8], 'alfa':[8,12],'beta':[12,20],'gama':[10,45]}
  #Output: 
  #   - retorna um array de trechos x características e uma lista com os nomes das 
  # caracteristicas correspondentes ao array
    
  (S,X)=np.shape(sin) #S = numero de sinais sinais; X = tamanho dos sinais no tempo
  nc=15 #numero de caracteristicas que serao extraidas
  car=np.zeros((S,nc)) #matriz das caracteristicas
  nomesc=[None]*nc
  
  for s in range(S):

    #média
     car[s,0]=np.mean(sin[s,:])
     nomesc[0]='media'

    #variancia
     var0=np.var(sin[s,:],ddof=1)
     car[s,1]=var0
     nomesc[1]='variancia'

    #mobilidade
     x1=np.diff(sin[s,:])
     var1=np.var(x1,ddof=1)
     mob=var1/var0
     car[s,2]=mob
     nomesc[2]='mobilidade'
          
    #complexidade estatística
     x2=np.diff(x1)
     var2=np.var(x2,ddof=1)
     ce=(var2/var1-var1/var0)**(1/2)
     car[s,3]=ce
     nomesc[3]='complexidade'

    ##calculando o espectro:
     yf = np.abs(fft.fft(sin[s,:]-car[s,0]))**2 
     yf=yf/np.size(yf)
     yf=yf[0:X//2]
     xf = np.linspace(0.0, 1.0/(2.0/freq), X//2)  
     Yf=yf/np.sum(yf) 

    #frequência central do espectro
     car[s,4]=np.sum(xf*Yf)
     nomesc[4]='f-central'

    #potencia na frequencia central
     ifc=np.abs(xf-car[s,4])==np.min(np.abs(xf-car[s,4]))
     car[s,5]=yf[ifc]
     nomesc[5]='P na fc'

    #largura de banda do espectro
     car[s,6]=np.sqrt(np.sum(((xf-car[s,4])**2)*Yf))
     nomesc[6]='l-banda'

    #frequência de margem do espectro
     sw=np.cumsum(Yf)
     f=np.max(np.where(sw<=0.9)[0])
     car[s,7]=xf[f]
     nomesc[7]='f-margem'

    #potências espectrais normalizadas nas seguintes bandas: 
    #delta 1 (0.5 a 2.5Hz)
     for ib, b in enumerate(bandas):
        car[s,8+ib]=sum(Yf[((xf>=bandas[b][0]) & (xf<=bandas[b][1]))])
        nomesc[8+ib]=b

  return (car,nomesc)

