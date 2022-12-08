import scipy.io
import pandas as pd

'''
Sobre os dados:

19 476 homens com mais de 50 anos e que foram submetidos a testes de câncer de
próstata. 

Coluna 1: resultado do teste PSA (1 = positivo para a doença, 0 = negativo para a
doença);

Coluna 2: resultado do teste de toque retal DRE (1 = positivo para a doença, 0 =
negativo para doença); 

Coluna 3 = resultado da biópsia (1 = paciente com a doença, 0 =
paciente sem a doença). 
'''


######## ITEM A e B - (INICIO) #########
mat = scipy.io.loadmat('Atividade 2.mat')
dados = mat['dados']
df_dados = pd.DataFrame(dados, columns=['PSA', 'DRE', 'Rotulo'])

#especificidade - P(T=0|D=0) - probabilidade do teste dar negativo dado que n está donte
#sensibilidade - P(T=1|D=1) - probabilidade do teste dar possitivo dado que está donte
VP_PSA = df_dados[(df_dados['PSA'] == 1) & (df_dados['Rotulo'] == 1)].shape[0]
FN_PSA = df_dados[(df_dados['PSA'] == 0) & (df_dados['Rotulo'] == 1)].shape[0]
VP_DRE = df_dados[(df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 1)].shape[0]
FN_DRE = df_dados[(df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 1)].shape[0]

sensibilidade_PSA = VP_PSA / (VP_PSA + FN_PSA)
sensibilidade_DRE = VP_DRE / (VP_DRE + FN_DRE)

VN_DRE = df_dados[(df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 0)].shape[0]
FP_DRE = df_dados[(df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 0)].shape[0]
VN_PSA = df_dados[(df_dados['PSA'] == 0) & (df_dados['Rotulo'] == 0)].shape[0]
FP_PSA = df_dados[(df_dados['PSA'] == 1) & (df_dados['Rotulo'] == 0)].shape[0]

especificidade_PSA = VN_PSA / (VN_PSA + FP_PSA)
especificidade_DRE = VN_DRE / (VN_DRE + FP_DRE)

print(sensibilidade_PSA)
print(sensibilidade_DRE)
print(especificidade_PSA)
print(especificidade_DRE)
######## ITEM A e B - (FIM) #########


######## ITEM C - (INICIO) #########
apriori = 4.2 / 100
#Note que para a sequencia de problemas onde queremos testar qual a probabilidade do paciente
#estar doente dado que o teste deu positivo usaremos o seguinte bayes:
#P(D=1|T=1) = P(T=1|D=1) * P(D=1) / ( P(T=1|D=1)*P(D=1) + P(T=1|D=0)*P(D=0) )
#P(D=1|T=1): Probabilidade de estar doente dado que testou positvo
#P(T=1|D=1): Probabilidade de testar positivo dado que está doente
#P(D=1): a priori


#P(T=1|D=1): sensibilidade
#P(T=0|D=0): especificidade
#P(T=1|D=0) =  1 - P(T=0|D=0)
#P(D=1): a priori
prob_estar_doente_PSA = sensibilidade_PSA * apriori / (sensibilidade_PSA * apriori + (1 - especificidade_PSA) * (1 - apriori))
prob_estar_doente_DRE = sensibilidade_DRE * apriori / (sensibilidade_DRE * apriori + (1 - especificidade_DRE) * (1 - apriori))
print(prob_estar_doente_PSA)
print(prob_estar_doente_DRE)
#para resolver tal questão final da aula
#https://www.youtube.com/watch?v=WqqOGAIpVTI
######## C - (FIM) #########


######## D - (INICIO) #########
#Nesse caso ainda estamos com o mesmo objetivo, ou seja, descobrir P(D=1|T=1), então
#o formato do nosso bayes não muda, porém a construção das partes que o compõem
# serão alteradas a fim de compor os testes em uma condição and e outra or.

#Caso And para postivo em psa e dre
VP_PSA_E_DRE = df_dados[(df_dados['PSA'] == 1) & (df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 1)].shape[0]
FN_PSA_E_DRE = df_dados[
                   ((df_dados['PSA'] == 0) & (df_dados['DRE'] == 0) & (df_dados['Rotulo']) == 1) |
                   ((df_dados['PSA'] == 1) & (df_dados['DRE'] == 0) & (df_dados['Rotulo']) == 1) |
                   ((df_dados['PSA'] == 0) & (df_dados['DRE'] == 1) & (df_dados['Rotulo']) == 1)
].shape[0]


sensibilidade_PSA_E_DRE = VP_PSA_E_DRE / (VP_PSA_E_DRE + FN_PSA_E_DRE)


VN_DRE_E_PSA = df_dados[
    ((df_dados['DRE'] == 0) & (df_dados['PSA'] == 0) & (df_dados['Rotulo'] == 0)) |
    ((df_dados['DRE'] == 1) & (df_dados['PSA'] == 0) & (df_dados['Rotulo'] == 0)) |
    ((df_dados['DRE'] == 0) & (df_dados['PSA'] == 1) & (df_dados['Rotulo'] == 0))
].shape[0]
FP_DRE_E_PSA = df_dados[(df_dados['DRE'] == 1) & (df_dados['PSA'] == 1) & (df_dados['Rotulo'] == 0)].shape[0]


especificidade_PSA_E_DRE = VN_DRE_E_PSA / (VN_DRE_E_PSA + FP_DRE_E_PSA)

prob_estar_doente_DRE_E_PSA = sensibilidade_PSA_E_DRE * apriori / (sensibilidade_PSA_E_DRE * apriori + (1 - especificidade_PSA_E_DRE) * (1 - apriori))
print(prob_estar_doente_DRE_E_PSA)


#Caso Or para postivo em psa e dre
VP_PSA_OR_DRE = df_dados[
    ((df_dados['PSA'] == 1) & (df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 1)) |
    ((df_dados['PSA'] == 1) & (df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 1)) |
    ((df_dados['PSA'] == 0) & (df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 1))
].shape[0]
FN_PSA_OR_DRE = df_dados[(df_dados['PSA'] == 0) & (df_dados['DRE'] == 0) & (df_dados['Rotulo']) == 1].shape[0]


sensibilidade_PSA_OR_DRE = VP_PSA_OR_DRE / (VP_PSA_OR_DRE + FN_PSA_OR_DRE)


VN_DRE_OR_PSA = df_dados[(df_dados['DRE'] == 0) & (df_dados['PSA'] == 0) & (df_dados['Rotulo'] == 0)].shape[0]
FP_DRE_OR_PSA = df_dados[
    ((df_dados['PSA'] == 1) & (df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 0)) |
    ((df_dados['PSA'] == 1) & (df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 0)) |
    ((df_dados['PSA'] == 0) & (df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 0))
].shape[0]


especificidade_PSA_OR_DRE = VN_DRE_OR_PSA / (VN_DRE_OR_PSA + FP_DRE_OR_PSA)

prob_estar_doente_DRE_OR_PSA = sensibilidade_PSA_OR_DRE * apriori / (sensibilidade_PSA_OR_DRE * apriori + (1 - especificidade_PSA_OR_DRE) * (1 - apriori))
print(prob_estar_doente_DRE_OR_PSA)
######## D - (FIM) #########


######## E - (INICIO) #########
#Nesse ponto precisamos reavaliar o nosso bayes pois a pergunta passou a ser:
# probabilidade de ter cancer se teste der negativo
#aplicando bayes percebemos que precisamos obter algo do tipo:
#P(D=1|T=0) = P(T=0|D=1) * P(D=1) / (P(T=0|D=1) * P(D=1) + P(T=0|D=0) * P(D=0))

#o unico termo mais famozinho aqui é P(T=0|D=0) - especificidade


#Analisando DRE e PSA isoladamente:
P_T0_D1_PSA = df_dados[
    (df_dados['PSA'] == 0) & (df_dados['Rotulo'] == 1)
].shape[0] / df_dados[df_dados['Rotulo'] == 1].shape[0]

P_T0_D1_DRE = df_dados[
    (df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 1)
].shape[0] / df_dados[df_dados['Rotulo'] == 1].shape[0]


VN_DRE = df_dados[(df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 0)].shape[0]
FP_DRE = df_dados[(df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 0)].shape[0]
VN_PSA = df_dados[(df_dados['PSA'] == 0) & (df_dados['Rotulo'] == 0)].shape[0]
FP_PSA = df_dados[(df_dados['PSA'] == 1) & (df_dados['Rotulo'] == 0)].shape[0]

especificidade_PSA = VN_PSA / (VN_PSA + FP_PSA)
especificidade_DRE = VN_DRE / (VN_DRE + FP_DRE)

prob_estar_doente_PSA = P_T0_D1_PSA * apriori / (P_T0_D1_PSA * apriori + (especificidade_PSA) * (1 - apriori))
prob_estar_doente_DRE = P_T0_D1_DRE * apriori / (P_T0_D1_DRE * apriori + (especificidade_DRE) * (1 - apriori))
print(prob_estar_doente_PSA)
print(prob_estar_doente_DRE)



#analisando probabilidade de ter cancer se teste negativo para DRE e PSA em conjunto
P_T0_D1 = df_dados[
    ((df_dados['PSA'] == 0) & (df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 1))
].shape[0] / df_dados[df_dados['Rotulo'] == 1].shape[0]


VN_DRE_E_PSA = df_dados[
    ((df_dados['PSA'] == 0) & (df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 0))
].shape[0]

FP_DRE_E_PSA = df_dados[
    ((df_dados['PSA'] == 1) & (df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 0)) |
    ((df_dados['PSA'] == 0) & (df_dados['DRE'] == 1) & (df_dados['Rotulo'] == 0)) |
    ((df_dados['PSA'] == 1) & (df_dados['DRE'] == 0) & (df_dados['Rotulo'] == 0))
].shape[0]


especificidade_PSA_E_DRE = VN_DRE_E_PSA / (VN_DRE_E_PSA + FP_DRE_E_PSA)
print(P_T0_D1 * apriori / (P_T0_D1 * apriori + especificidade_PSA_E_DRE * (1 - apriori)))






