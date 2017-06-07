from numpy import *
from time import sleep
from .optStruct import *
from .SMO import *

# Select any J not equal to i
def selectJrand(i, m):
    j = i
    while (j==i):
        j = int(random.uniform(0, m))
    return j

# Clip Alpha
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# Calculer kernel function
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros(m, 1))
    if kTup[0] == 'lin': K = X * A.T # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matr ix like Matlab
    else: raise NameError('Kernel type non connu')
    return K

# Calculer distance entre point et margin line
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.ekCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        # loop through valid Ecache values and f ind the one that maximizes delta E
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej

