import pandas as pd
from sklearn.model_selection import train_test_split;
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
import random as r

dados = pd.read_csv('C:\Users\willi.DESKTOP-REK1S4P\Desktop\PDFs e Documentos de CC\2K21\IA - ADRIAN(Ax1000)\Trabalho - Redes Neurais - Chuva na Australia\weatherAUS.csv')

# precision_score(zero_division=1)


rain = dados.groupby(['RainRoday', 'RainTomorrow'])

# print('\n', dados.head(5))

# print('\n\n\n')

# print('\n\n\n',"Data size:",(dados.size)/7,"items\n\nTraining Data:\n", dados)
# print(dados.shape)


dataShuffle = dados.sample(frac=1)  # shuffle

# dataShuffle.to_csv(r"C:\Users\willi.DESKTOP-REK1S4P\Desktop\dataShuffle.csv")

# print(dataShuffle.pivot_table(index='Class', aggfunc='size'))

attSet = dados.drop("Class", axis=1)
classSet = dados["Class"]

# print(classSet)

trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
    train_test_split(attSet, classSet, test_size=0.25, \
                     random_state=(int)(r.random() * 99999), stratify=classSet)

trainingAttSet, validAttSet, trainingClassSet, validClassSet = \
    train_test_split(trainingAttSet, trainingClassSet, test_size=0.33, \
                     random_state=(int)(r.random() * 99999), stratify=trainingClassSet)


def oneMinusNormDist(distMat):
    scaler = StandardScaler()
    scaledMat = scaler.fit_transform(distMat)

    retMat = 1 - scaledMat

    return retMat

    # MLP test


# print('\n\n\n\n\n\n\n\n\n\n\n\n\n COMEÇA AQUI \n\n\n\n\n\n\n\n\n\n\n\n\n')
# percepAcc = 0.0
# for i in range(1, 21):
#         trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#         train_test_split(attSet, classSet, test_size=0.25, \
#         random_state=(int)(r.random()*99999), stratify=classSet)


#         trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#         train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#         random_state=(int)(r.random()*99999), stratify=trainingClassSet)


#         percep = MLPClassifier(hidden_layer_sizes=(16,16), max_iter=(1000),\
#                                 learning_rate_init=0.01)
#                 #hidden layer sizes, max iter, learning rate init
#         percep.fit(trainingAttSet, trainingClassSet)
#         prev = percep.predict(testingAttSet)
#         res = classification_report(testingClassSet, prev)
#         print(res.split()[20])
#         percepAcc+= float(res.split()[20])

# percepAcc = percepAcc/20
# print('\n\n\n\nMLPerceptron accuracy:', percepAcc)


# KNN test
# print('\n\n\n\n\n\n\n\n\n\n\n\n\n COMEÇA AQUI \n\n\n\n\n\n\n\n\n\n\n\n\n')
# knnAcc = 0.0
# for i in range(1, 21):
#         trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#         train_test_split(attSet, classSet, test_size=0.25, \
#         random_state=(int)(r.random()*99999), stratify=classSet)


#         trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#         train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#         random_state=(int)(r.random()*99999), stratify=trainingClassSet)


#         knn = KNeighborsClassifier(n_neighbors = 32, weights = oneMinusNormDist, \
#                                    metric='euclidean')
#         knn.fit(trainingAttSet, trainingClassSet)
#         prev = knn.predict(testingAttSet)
#         res = classification_report(testingClassSet, prev)
#         print(res.split()[20])
#         knnAcc+= float(res.split()[20])

# knnAcc = knnAcc/20
# print('\n\n\n\nKNN accuracy:', knnAcc)


# DT test
# print('\n\n\n\n\n\n\n\n\n\n\n\n\n COMEÇA AQUI \n\n\n\n\n\n\n\n\n\n\n\n\n')
# dtAcc = 0.0

# for i in range(1, 21):
#         trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#         train_test_split(attSet, classSet, test_size=0.25, \
#         random_state=(int)(r.random()*99999), stratify=classSet)


#         trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#         train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#         random_state=(int)(r.random()*99999), stratify=trainingClassSet)


#         dt = DecisionTreeClassifier()
#         dt.fit(trainingAttSet, trainingClassSet)
#         prev = dt.predict(testingAttSet)
#         res = classification_report(testingClassSet, prev)
#         print(res.split()[20])
#         dtAcc+= float(res.split()[20])

# dtAcc = dtAcc/20
# print('\n\n\n\nDT accuracy:', dtAcc)


# NB test
# print('\n\n\n\n\n\n\n\n\n\n\n\n\n COMEÇA AQUI \n\n\n\n\n\n\n\n\n\n\n\n\n')
# nbAcc = 0.0

# for i in range(1, 21):
#         trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#         train_test_split(attSet, classSet, test_size=0.25, \
#         random_state=(int)(r.random()*99999), stratify=classSet)


#         trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#         train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#         random_state=(int)(r.random()*99999), stratify=trainingClassSet)


#         nb = GaussianNB()
#         nb.fit(trainingAttSet, trainingClassSet)
#         prev = nb.predict(testingAttSet)
#         res = classification_report(testingClassSet, prev)
#         print(res.split()[20])
#         nbAcc+= float(res.split()[20])

# nbAcc = nbAcc/20
# print('\n\n\n\nNB accuracy:', nbAcc)


# SVM test
# print('\n\n\n\n\n\n\n\n\n\n\n\n\n COMEÇA AQUI \n\n\n\n\n\n\n\n\n\n\n\n\n')
# svmAcc = 0.0

# for i in range(1, 21):
#         trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#         train_test_split(attSet, classSet, test_size=0.25, \
#         random_state=(int)(r.random()*99999), stratify=classSet)


#         trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#         train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#         random_state=(int)(r.random()*99999), stratify=trainingClassSet)


#         svm = SVC(C = 1.00, kernel='poly')
#         svm.fit(trainingAttSet, trainingClassSet)
#         prev = svm.predict(testingAttSet)
#         res = classification_report(testingClassSet, prev)
#         #print(res)
#         print(res.split()[20])
#         svmAcc+= float(res.split()[20])

# svmAcc = svmAcc/20
# print('\n\n\n\nSVM accuracy:', svmAcc)


knnMean = [0.67, 0.65, 0.64, 0.62, 0.59, 0.59, 0.62, 0.67, 0.62, 0.66, \
           0.66, 0.71, 0.62, 0.64, 0.62, 0.63, 0.67, 0.66, 0.71, 0.65]

dtMean = [0.61, 0.66, 0.58, 0.55, 0.66, 0.67, 0.62, 0.65, 0.69, 0.57, \
          0.54, 0.59, 0.68, 0.60, 0.61, 0.57, 0.71, 0.62, 0.70, 0.56]

nbMean = [0.66, 0.61, 0.51, 0.51, 0.50, 0.62, 0.52, 0.61, 0.60, 0.58, \
          0.60, 0.61, 0.59, 0.62, 0.52, 0.56, 0.54, 0.50, 0.52, 0.62]

svmMean = [0.61, 0.68, 0.69, 0.71, 0.68, 0.76, 0.62, 0.61, 0.65, 0.60, \
           0.67, 0.63, 0.75, 0.72, 0.74, 0.67, 0.67, 0.68, 0.56, 0.61]

mlpMean = [0.59, 0.62, 0.65, 0.62, 0.69, 0.63, 0.61, 0.70, 0.73, 0.66, \
           0.63, 0.69, 0.74, 0.75, 0.67, 0.67, 0.67, 0.73, 0.62, 0.68]

#                              statistical tests
#                                   kruskal

# kruskalTest = kruskal(knnMean, dtMean, nbMean, svmMean, mlpMean)
# print('\n\n\n\n',kruskalTest, '\n\n\n\n')


#                               mann-whitney


# for i in range(0,5):
#     for j in range(1,6):
#         if(j<=i):
#             j = j

#         elif (i==0):
#             if j==1:
#                 print('KNN vs DT:\n',mannwhitneyu(knnMean, dtMean))
#             elif j==2:
#                 print('KNN vs NB:\n',mannwhitneyu(knnMean, nbMean))
#             elif j==3:
#                 print('KNN vs SVM:\n',mannwhitneyu(knnMean, svmMean))
#             elif j==4:
#                 print('KNN vs MLP:\n',mannwhitneyu(knnMean, mlpMean))

#         elif (i==1):
#             if j==2:
#                 print('DT vs NB:\n',mannwhitneyu(dtMean, nbMean))
#             elif j==3:
#                 print('DT vs SVM:\n',mannwhitneyu(dtMean, svmMean))
#             elif j==4:
#                 print('DT vs MLP:\n',mannwhitneyu(dtMean, mlpMean))

#         elif (i==2):
#             if j==3:
#                 print('NB vs SVM:\n',mannwhitneyu(nbMean, svmMean))
#             elif j==4:
#                 print('NB vs MLP:\n',mannwhitneyu(nbMean, mlpMean))

#         elif (i==3):
#             if j==4:
#                 print('SVM vs MLP:\n',mannwhitneyu(svmMean, mlpMean))

mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, \
                    learning_rate_init=0.01)
mlp.fit(trainingAttSet, trainingClassSet)
mlpPredict = mlp.predict(testingAttSet)
MLPTV = mlp.predict_proba(testingAttSet)


def soma(KNN, DT, NB, SVM, MLP):
    soma = KNN + DT + NB + SVM + MLP
    somaR = []

    for i in range(0, 87):
        if soma[i][0] > soma[i][1]:
            somaR.append(1)
        else:
            somaR.append(2)

    return somaR


def produto(KNN, DT, NB, SVM, MLP):
    produto = KNN * DT * NB * SVM * MLP
    prodR = []

    for i in range(0, 87):
        if produto[i][0] > produto[i][1]:
            prodR.append(1)
        else:
            prodR.append(2)

    return prodR


def bordaCount(KNN, DT, NB, SVM, MLP):
    bcknn = []
    bcdt = []
    bcnb = []
    bcsvm = []
    bcmlp = []

    for i in range(0, 5):
        if i == 0:
            for j in range(0, 87):
                if KNN[j][0] > KNN[j][1]:
                    bcknn.append([2, 1])
                else:
                    bcknn.append([1, 2])
        elif i == 1:
            for j in range(0, 87):
                if DT[j][0] > DT[j][1]:
                    bcdt.append([2, 1])
                else:
                    bcdt.append([1, 2])
        elif i == 2:
            for j in range(0, 87):
                if NB[j][0] > NB[j][1]:
                    bcnb.append([2, 1])
                else:
                    bcnb.append([1, 2])
        elif i == 3:
            for j in range(0, 87):
                if SVM[j][0] > SVM[j][1]:
                    bcsvm.append([2, 1])
                else:
                    bcsvm.append([1, 2])
        elif i == 4:
            for j in range(0, 87):
                if MLP[j][0] > MLP[j][1]:
                    bcmlp.append([2, 1])
                else:
                    bcmlp.append([1, 2])

    bordCount = soma(bcdt, bcknn, bcmlp, bcnb, bcsvm)

    return bordCount


# print(KNNTV)


#                       MCS (multi classifier systems )
#                                   Soma


# somaMCS = soma(KNNTV, DTTV, NBTV, SVMTV, MLPTV)

# print(somaMCS)


# somaAcc=0

# for i in range(0,20):
#     trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#     train_test_split(attSet, classSet, test_size=0.25, \
#                       random_state=(int)(r.random()*99999), stratify=classSet)


#     trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#     train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#                       random_state=(int)(r.random()*99999), stratify=trainingClassSet)

#     res = classification_report(testingClassSet, somaMCS)
#     # print(res)5
#     print(i, ' = ',res.split()[20])
#     somaAcc+= float(res.split()[20])

# somaAcc = somaAcc/20
# print('\n\n\n\n somaSMC mean accuracy = ',somaAcc)


# #                               Produto

# prodMCS = produto(KNNTV, DTTV, NBTV, SVMTV, MLPTV)
# prodAcc=0

# for i in range(0,20):
#     trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#     train_test_split(attSet, classSet, test_size=0.25, \
#                       random_state=(int)(r.random()*99999), stratify=classSet)


#     trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#     train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#                       random_state=(int)(r.random()*99999), stratify=trainingClassSet)

#     res = classification_report(testingClassSet, prodMCS)
#     # print(res)5
#     print(i, ' = ',res.split()[20])
#     prodAcc+= float(res.split()[20])

# prodAcc = prodAcc/20
# print('\n\n\n\n prodSMC mean accuracy = ',prodAcc)


# #                           Borda Count


# borCoMCS = bordaCount(KNNTV, DTTV, NBTV, SVMTV, MLPTV)
# borCoAcc = 0.0

# for i in range(0,20):
#     trainingAttSet, testingAttSet, trainingClassSet, testingClassSet = \
#     train_test_split(attSet, classSet, test_size=0.25, \
#                           random_state=(int)(r.random()*99999), stratify=classSet)


#     trainingAttSet, validAttSet, trainingClassSet, validClassSet =\
#     train_test_split(trainingAttSet, trainingClassSet, test_size=0.33,\
#                           random_state=(int)(r.random()*99999), stratify=trainingClassSet)

#     res = classification_report(testingClassSet, borCoMCS)
#     # print(res)
#     print(i, ' = ',res.split()[20])
#     borCoAcc+= float(res.split()[20])

# borCoAcc = borCoAcc/20
# print('\n\n\n\n borCoSMC mean accuracy = ',borCoAcc)


somaMean = [0.44, 0.61, 0.44, 0.51, 0.58, 0.56, 0.61, 0.47, 0.49, 0.51, \
            0.49, 0.51, 0.54, 0.54, 0.54, 0.49, 0.51, 0.56, 0.61, 0.51]

produtoMean = [0.50, 0.48, 0.49, 0.55, 0.57, 0.45, 0.43, 0.55, 0.50, 0.62, \
               0.48, 0.45, 0.43, 0.64, 0.57, 0.50, 0.52, 0.55, 0.50, 0.50]

borCoMean = [0.48, 0.59, 0.48, 0.55, 0.45, 0.48, 0.48, 0.62, 0.50, 0.52, \
             0.43, 0.59, 0.50, 0.57, 0.52, 0.62, 0.55, 0.64, 0.64, 0.48]

#                              statistical tests
#                                   kruskal

# kruskalTest = kruskal(somaMean, produtoMean, borCoMean)
# print('\n\n\n\n',kruskalTest, '\n\n\n\n')


#                               mann-whitney


# for i in range(0,3):
#     for j in range(1,3):
#         if(j<=i):
#             j = j

#         elif (i==0):
#             if j==1:
#                 print('Soma vs Produto:\n',mannwhitneyu(somaMean, produtoMean))
#             elif j==2:
#                 print('Soma vs Borda Count:\n',mannwhitneyu(somaMean, borCoMean))


#         elif (i==1):
#             if j==2:
#                 print('Produto vs Borda Count:\n',mannwhitneyu(produtoMean, borCoMean))
