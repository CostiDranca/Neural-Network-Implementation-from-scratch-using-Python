import numpy as np
import random
import sys
import pandas as pd
from sklearn.datasets import load_iris

def purelin(weights,bias,X):
    resultat=0.0
    for i in range(len(X)):
        resultat+=weights[i]*X[i]
    resultat+=bias
    return resultat

def logsig(weights, bias,X):
    resultat=0.0
    for i in range(0,len(weights)):
        resultat+=weights[i]*X[i]
    resultat+=bias
    return 1.0/(1.0+np.exp(-resultat))

def logsigMat(weights,bias,X):
    result=np.array([0.0]*len(weights[0]))
    col=len(weights[0])
    for j in range(0,col):
        for i in range(len(weights)):
            result[j]+=weights[i][j]*X[i]
    for i in range(0,len(bias)):
        result[i]+=bias[i]
    for i in range(len(result)):
        result[i]=1.0/(1.0+np.exp(-result[i]))
    return result

def scalarProduct(X,a):
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j]=a*X[i][j]
    return X

def produs(X):
    Y=X
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j]=Y[i][j]*Y[i][j]
    return Y

def produsVec(X):
    Y=X
    for i in range(len(Y)):
        Y[i]=Y[i]*Y[i]
    return Y

def aplicRata(r,X,epsilon,rata):
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j]=(rata/np.sqrt(epsilon+r[i][j]))*X[i][j]
    return X

def aplicRataVec(r,X,epsilon,rata):
    for i in range(len(X)):
        X[i]=(rata/np.sqrt(epsilon+r[i]))*X[i]
    return X


iris = load_iris()

a = iris.data[0:50]
b = iris.data[50:100]
c = iris.data[100:150]
permut = np.random.permutation(50)
dataInput = np.array([a[permut[0]]])
labels = np.array([1, 0, 0])
dataTest = np.array([a[permut[1]]])
labelsTest = np.array([1, 0, 0])

for i in range(2, len(a)):
    if i >= 41:
        dataTest = np.vstack([dataTest, np.array(a[permut[i]])])
        labelsTest = np.vstack([labelsTest, np.array([1, 0, 0])])
    else:
        dataInput=np.vstack([dataInput,np.array(a[permut[i]])])
        labels=np.vstack([labels,np.array([1,0,0])])

permut = np.random.permutation(50)
for i in range(len(b)):
    if i >= 40:
        dataTest = np.vstack([dataTest, np.array(b[permut[i]])])
        labelsTest = np.vstack([labelsTest, np.array([0, 1, 0])])
    else:
        dataInput=np.vstack([dataInput,np.array(b[permut[i]])])
        labels=np.vstack([labels,np.array([0,1,0])])

permut = np.random.permutation(50)
for i in range(len(c)):
    if i >= 40:
        dataTest = np.vstack([dataTest, np.array(c[permut[i]])])
        labelsTest = np.vstack([labelsTest, np.array([0, 0, 1])])
    else:
        dataInput=np.vstack([dataInput,np.array(c[permut[i]])])
        labels=np.vstack([labels,np.array([0,0,1])])

weightsOut=np.array([[(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],
            [(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],
            [(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],
            [(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],
            [(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5]],dtype='float64')
biasOut=np.array([(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],dtype='float64')


weightsHid=np.array([[(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],
            [(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],
            [(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],
            [(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5]],dtype='float64')
biasHid=np.array([(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5,(random.random()*3)-1.5],dtype='float64')

# This is a preset for trained weights and biases, they can be commented for training a new network
# This network has a error = 8.959189753382729 and a Success Rate = 100%
"""
weightsOut=[[-20.79299199,   0.1847319,    8.81703376],
 [-21.20449858,   0.83456487,  10.67579898],
 [-21.45395865,  -1.45395807,  10.09560399],
 [-19.03838993,  -0.85550148,   9.33591963],
 [-20.41883901,  -1.05622396,   9.73490506]]
biasOut=[  37.02521395,   0.87045963, -22.62997913]
weightsHid=[[-0.51220136, -2.08645221, -4.27529558,  1.33990289,  0.04954917],
 [-3.16453262, -0.10232599, -5.53834225,  1.19588763,  0.24206695],
 [ 4.1697274,  -0.52231721,  7.53370379,  0.78336367, -1.02639363],
 [ 1.93673021, -1.28032128,  9.69738608, -1.43817041, -2.06858733]]
biasHid=[ -0.52382896,   0.69218212, -11.06672414,  -1.26062715,   1.82981413]
"""

permutare=np.random.permutation(120)
AUXINPUT=np.array([dataInput[permutare[0]]])
AUXLABELS=np.array([labels[permutare[0]]])
for i in range(1,len(permutare)):
    AUXINPUT=np.vstack([AUXINPUT,dataInput[permutare[i]]])
    AUXLABELS=np.vstack([AUXLABELS,labels[permutare[i]]])
dataInput=AUXINPUT
labels=AUXLABELS

# This sections counts the number of flowers from training set for each class
nr1=0
nr2=0
nr3=0
for i in range(len(labels)):
    if np.array_equal(labels[i],[1,0,0]):
        nr1+=1
    if np.array_equal(labels[i],[0,1,0]):
        nr2+=1
    if np.array_equal(labels[i],[0,0,1]):
        nr3+=1
print (nr1)
print (nr2)
print (nr3)

# This is the sections for setting the parameters for training
rataO=0.01
rataH=0.01
epsilon=0.000001
p=0.9
rataMoment=0.01
epoca=0
nrEpochs=1000
eroare=999999.0
UpdateOut = np.array([[0.0]*len(weightsOut[0])]*len(weightsOut),dtype='float64')
UpdateHid=np.array([[0.0]*len(weightsHid[0])]*len(weightsHid),dtype='float64')
UpBiasOut=np.array([0.0]*len(weightsOut[0]),dtype='float64')
UpBiasHid=np.array([0.0]*len(weightsHid[0]),dtype='float64')

print("I'm still training")
while epoca<=nrEpochs and eroare>=5:

    # This Area is for random select from InputData
    permutare = np.random.permutation(120)
    AUXINPUT = np.array([dataInput[permutare[0]]])
    AUXLABELS = np.array([labels[permutare[0]]])
    for i in range(1, len(permutare)):
        AUXINPUT = np.vstack([AUXINPUT, dataInput[permutare[i]]])
        AUXLABELS = np.vstack([AUXLABELS, labels[permutare[i]]])
    dataInput = AUXINPUT
    labels = AUXLABELS


    epoca+=1
    eroare=0
    # This is for the momentum, it uses the previous values
    UpdateOut=rataMoment*UpdateOut
    UpdateHid = rataMoment * UpdateHid
    UpBiasOut=rataMoment*UpBiasOut
    UpBiasHid=rataMoment*UpBiasHid

    for i in range(len(dataInput)):
        resultHid=logsigMat(weightsHid,biasHid,dataInput[i])
        resultOut=logsigMat(weightsOut,biasOut,resultHid)
        dif=labels[i]-resultOut
        #print labels[i],resultOut
        for coloana in range(len(weightsHid[0])):
            eroareLayerOut = 0.0
            for j in range(len(dif)):
                eroareLayerOut += dif[j] * resultOut[j] * (1 - resultOut[j]) * weightsOut[coloana][j]
            for linie in range(len(weightsHid)):
                UpdateHid[linie][coloana]+=eroareLayerOut*resultHid[coloana]*(1-resultHid[coloana])*dataInput[i][linie]
            UpBiasHid[coloana]+=eroareLayerOut*resultHid[coloana]*(1-resultHid[coloana])
        for coloana in range(len(weightsOut[0])):
            for linie in range(len(weightsOut)):
                UpdateOut[linie][coloana]+=(dif[coloana])*resultOut[coloana]*(1-resultOut[coloana])*resultHid[coloana]
            UpBiasOut[coloana]+=(dif[coloana])*resultOut[coloana]*(1-resultOut[coloana])
        for l in range(len(dif)):
            eroare+=(dif[l]**2)/2.0

    weightsOut+=rataO*UpdateOut
    biasOut+=rataO*UpBiasOut
    weightsHid+=rataH*UpdateHid
    biasHid+=rataH*UpBiasHid
    if ((epoca%100==0) or (epoca ==1)):
        print("Epoch: ",epoca," Error: ", eroare)
print("Finish with training")
nrReusite=0.0
print(eroare)
print(weightsOut)
print(biasOut)
print(weightsHid)
print(biasHid)

for i in range(len(dataTest)):
        result=logsigMat(weightsOut,biasOut,logsigMat(weightsHid,biasHid,dataTest[i]))
        #print(result)
        for j in range(len(result)):
            if(result[j]<0.5):result[j]=0
            else: result[j]=1
        print(labelsTest[i],result)
        if np.array_equal(labelsTest[i],result):
            nrReusite+=1.0
print("Success Rate : ",nrReusite/(len(dataTest)))
