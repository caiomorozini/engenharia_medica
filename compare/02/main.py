import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from extraicarac import extraicarac
import scipy.io
import pandas as pd

#O arquivo contém 1179 trechos, de 30 segundos cada, armazenados na
#variável “SINAL” (trechos x tempo). Os trechos são amostrados a 100Hz.

#encontrará também um vetor “ESTAGIOS” (trechos x 1) contendo a
#classificação de cada trecho em estágios de sono
#(0 = vigília, 1 = estágio 1, 2 = estágio 2, 3
#= estágio 3, 4 = estágio 4, 5 = REM).


############# ITEM A e B - (INICIO) ############
mat = scipy.io.loadmat('Dados.mat')
dados = mat['SINAL']
estagios = mat['ESTAGIOS']
print(dados.shape)
print(estagios.shape)



bandas = {
    'delta 1': [0.5, 2.5],
    'delta 2': [2.5, 4],
    'teta 1': [4, 6],
    'teta 2': [6, 8],
    'alfa': [8, 12],
    'beta': [12, 20],
    'gama': [20, 45]
}

#lista de listas
#primeira lista consiste em uma matriz onde cada coluna é uma caracteristica
#como média, potencia, etc e cada linha é o valor desse atributo para cada um dos sinais
#pois cada linha é um sinal
#a segunda lista contem os nomes dos atributos de cada coluna da matriz
res = extraicarac(sin=dados, freq=100, bandas=bandas)
print(res[0].shape)

df = pd.DataFrame(res[0], columns=res[1])

estagio_renomeado = []
for estagio in estagios:
    if estagio == 0:
        estagio_renomeado.append('vigília')
    elif estagio == 1:
        estagio_renomeado.append('estágio 1')
    elif estagio == 2:
        estagio_renomeado.append('estágio 2')
    elif estagio == 3:
        estagio_renomeado.append('estágio 3')
    elif estagio == 4:
        estagio_renomeado.append('estágio 4')
    else:
        estagio_renomeado.append('REM')

df['estágios'] = estagio_renomeado


aux_estagios = ['vigília', 'estágio 1', 'estágio 2', 'estágio 3', 'estágio 4', 'REM']
#varrer as caracteristicas (colunas da matriz)
for caracteristica in res[1]:
    plt.figure(figsize=(16, 10))
    #Em todos os sinais, use cada caracteristica para plotar
    # um histograma para cada estágio
    for pos, estagio_atual in enumerate(aux_estagios):

        plt.subplot(2, 3, pos + 1)
        #plota um histograma para um dado estágio df[df['estágios'] == estagio_atual]
        #a respeito de uma dada caracteristica
        plt.hist(df[df['estágios'] == estagio_atual][caracteristica])
        plt.title(f'{caracteristica} {estagio_atual}')


        if pos == 5:
            plt.savefig(f'grafico_{caracteristica}.png')

    plt.close()
############# ITEM A e B - (FIM) ############
       

############# ITEM C - (INICIO) ############
plt.figure()
x = df[df['estágios'] == 'vigília']['mobilidade']
y = df[df['estágios'] == 'vigília']['complexidade']
plt.scatter(x, y, color='blue', label='Vigilia')
x = df[df['estágios'] == 'REM']['mobilidade']
y = df[df['estágios'] == 'REM']['complexidade']
plt.scatter(x, y, color='green', label='REM')
x = df[df['estágios'] == 'estágio 4']['mobilidade']
y = df[df['estágios'] == 'estágio 4']['complexidade']
plt.scatter(x, y, color='red', label='Estagio 4')
plt.xlabel('Mobilidade')
plt.ylabel('Complexidade')
plt.legend()
plt.savefig('plot2d_caracteristicas.png')
############# ITEM C - (FIM) ############


############# ITEM D - (INICIO) ############
plt.figure()
ax = plt.axes(projection="3d")
x = df[df['estágios'] == 'vigília']['mobilidade']
y = df[df['estágios'] == 'vigília']['f-central']
z = df[df['estágios'] == 'vigília']['%delta 1']
ax.scatter3D(x, y, z, color='blue', label='vigília')
x = df[df['estágios'] == 'REM']['mobilidade']
y = df[df['estágios'] == 'REM']['f-central']
z = df[df['estágios'] == 'REM']['%delta 1']
ax.scatter3D(x, y, z, color='green', label='REM')
x = df[df['estágios'] == 'estágio 4']['mobilidade']
y = df[df['estágios'] == 'estágio 4']['f-central']
z = df[df['estágios'] == 'estágio 4']['%delta 1']
ax.scatter3D(x, y, z, color='red', label='estágio 4')
ax.set_xlabel('Mobilidade')
ax.set_ylabel('f-central')
ax.set_zlabel('%delta 1')
plt.legend()
plt.savefig('plot3d_caracteristicas.png')
############# ITEM D- (FIM) ############

