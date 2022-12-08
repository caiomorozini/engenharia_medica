import numpy as np
import pandas as pd

def ROC(valores, rotulos):

    df_aux = pd.DataFrame(np.array([valores, rotulos]).T, columns=['valor', 'rotulo'])
    df_aux.sort_values('valor', inplace=True)
    df_aux.reset_index(inplace=True)

    VP_percent = []
    FP_percent = []
    thresholds = list(df_aux['valor'])
    thresholds = [min(thresholds) - 1] + thresholds + [max(thresholds) + 1]
    for valor in thresholds:
        VP = df_aux[(df_aux['valor'] >= valor) & (df_aux['rotulo'] == 1)].shape[0]
        FP = df_aux[(df_aux['valor'] >= valor) & (df_aux['rotulo'] == 0)].shape[0]
        VN = df_aux[(df_aux['valor'] < valor) & (df_aux['rotulo'] == 0)].shape[0]
        FN = df_aux[(df_aux['valor'] < valor) & (df_aux['rotulo'] == 1)].shape[0]


        FP_percent.append(FP / (FP + VN))
        VP_percent.append(VP / (VP + FN))


    return (FP_percent, VP_percent)


def AUC(VP, FP):
    #https://www.youtube.com/watch?v=smNmTmeP-KA
    area = 0
    for i in range(len(VP) - 1):
        area += (VP[i] + VP[i + 1]) * (abs(FP[i + 1] - FP[i])) / 2

    return area





