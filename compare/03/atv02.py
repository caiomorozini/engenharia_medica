from sklearn import metrics
import matplotlib.pyplot as plt
from ROC import ROC, AUC
import scipy.io

mat = scipy.io.loadmat('Dados2.mat')
dados_ske_c1 = mat['ske'][:, 0]
dados_ske_c2 = mat['ske'][:, 1]
dados_med_c1 = mat['med'][:, 0]
dados_med_c2 = mat['med'][:, 1]

#nesse caso faço assim pq estão igualmente distribuidadas (mesma qtd por classe)
qtd_valores_por_rotulo = len(dados_ske_c1)
labels = [0] * qtd_valores_por_rotulo + [1] * qtd_valores_por_rotulo
scores_ske = list(dados_ske_c1) + list(dados_ske_c2)
FP, VP = ROC(scores_ske, labels)
print(f"auc ske minha: {AUC(VP, FP)}")
print(f"auc ske sklearn: {metrics.roc_auc_score(labels, scores_ske)}")
FP_sklearn, VP_sklearn, thresholds = metrics.roc_curve(labels, scores_ske)



plt.figure()
plt.subplot(1, 2, 1)
plt.plot(FP, VP)
plt.plot([0, 1], [0, 1])
plt.xlabel("% FP")
plt.ylabel("% VP")
plt.title("Gerado pela minha função")

plt.subplot(1, 2, 2)
plt.plot(FP_sklearn, VP_sklearn)
plt.plot([0, 1], [0, 1])
plt.xlabel("% FP")
plt.ylabel("% VP")
plt.title("Gerado skelarn")
plt.show()


scores_med = list(dados_med_c1) + list(dados_med_c2)
FP_sklearn, VP_sklearn, thresholds = metrics.roc_curve(labels, scores_med)
FP, VP = ROC(scores_med, labels)
print(f"med: {AUC(VP, FP)}")
print(f"auc med minha: {AUC(VP, FP)}")
print(f"auc med sklearn: {metrics.roc_auc_score(labels, scores_med)}")


plt.figure()
plt.subplot(1, 2, 1)
plt.plot(FP, VP)
plt.plot([0, 1], [0, 1])
plt.xlabel("% FP")
plt.ylabel("% VP")
plt.title("Gerado pela minha função")

plt.subplot(1, 2, 2)
plt.plot(FP_sklearn, VP_sklearn)
plt.plot([0, 1], [0, 1])
plt.xlabel("% FP")
plt.ylabel("% VP")
plt.title("Gerado skelarn")
plt.show()
#link documentação para poltar o grafico de roc
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
#FP, VP, thresholds = metrics.roc_curve(labels, scores_ske)
#caso suas labels não sejam do tipo -1, 1 ou 0, 1 você deverá indicar qual é o rótulo positovo
#por meio do parâmetro pos_label=valor_rotulopositvo
#documentaçao para calculo do auc
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html