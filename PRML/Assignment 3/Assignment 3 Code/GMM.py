import os
os.getcwd()
os.chdir('C:/Users/mjasw/Downloads/34/34')
import math as m
import random
import pandas as pd
import seaborn as sn
from pandas.plotting import table
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

input = np.loadtxt("train.txt", dtype='f', delimiter=',')
data =pd.DataFrame(input)

data1 = data.iloc[0:1250, 0:2]
data2 = data.iloc[1250:2500, 0:2]
print(data2)


def Kmeans(data, K=16, epoch=16, seedVal=73):
    np.random.seed(seedVal)
    d = data
    centId = np.random.choice(d.shape[0] - 1, K, replace=False)
    clust = {}
    cent = []
    cent = d[centId, :]
    cent = np.vstack(cent)
    count = epoch
    while count:
        newCent = []
        g = np.zeros((d.shape[0], K))

        for i in range(K):
            clust[i] = []

        for i in range(d.shape[0]):
            dist = np.sqrt(np.sum(np.square(d[i] - cent), axis=1))
            minI = np.argmin(dist)
            clust[minI].append(i)
            g[i][minI] = 1

        for k in clust.keys():
            newCent.append(np.mean(d[clust[k], :], axis=0))
        newCent = np.vstack(newCent)
        cent = np.vstack(newCent)
        count -= 1
    return g, clust


def N(x, mu, C):
    d = np.matmul(np.linalg.inv((np.matmul(np.transpose(C), C))), np.transpose(C))
    term1 = -0.5 * (np.matmul(np.matmul(x - mu, d), np.transpose(x - mu)))
    term2 = (pow(2 * m.pi, -0.5 * x.shape[0])) * (pow(abs(np.linalg.det(C)), -0.5))
    return term2 * (np.exp(term1))


def GMM(data, Q, epoch=3):
    d = data.to_numpy()
    n = d.shape[0]

    def likelihood(p):
        res = 0
        for i in range(n):
            sum = 0
            for j in range(Q):
                sum = sum + p[2][j] * N(d[i:i + 1, :], p[3][j], p[4][j])
            sum = m.log(sum)
            res = res + sum
        return res

    def mean(g, Nq):
        mu = []
        for i in range(Q):
            sum = np.zeros((1, d.shape[1]))
            for j in range(n):
                sum = sum + (g[j][i] * d[j])
            mu.append(sum)

        for j in range(Q):
            mu[j] = mu[j] / Nq[j]
        return mu

    def covariance(g, mu, Nq):
        c = []

        for i in range(Q):
            sum = np.zeros((d.shape[1], d.shape[1]))
            for j in range(n):
                x = d[j:j + 1, :]
                sum = sum + g[j][i] * (np.matmul(np.transpose(x - mu[i]), (x - mu[i])))
            for k in range(d.shape[1]):
                sum[k][k] = sum[k][k] + pow(10, -6)
            c.append(sum / Nq[i])
        return c

    def gamma(pi, mu, C):
        k = np.zeros((n, Q))

        for i in range(n):
            for j in range(Q):
                k[i][j] = pi[j] * (N(d[i:i + 1, :], mu[j], C[j]))
        g = np.zeros((n, Q))

        for i in range(n):
            for j in range(Q):
                g[i][j] = k[i][j] / np.sum(k, axis=1)[i]
        return g

    rOld = Kmeans(d, Q)[0]
    NqOld = np.sum(rOld, axis=0)
    piOld = NqOld / n
    muOld = mean(rOld, NqOld)
    cOld = covariance(rOld, muOld, NqOld)
    thetaOld = [rOld, NqOld, piOld, muOld, cOld]

    def newParam(thetaOld):
        p = thetaOld
        rNew = gamma(p[2], p[3], p[4])
        NqNew = np.sum(rNew, axis=0)
        piNew = NqNew / n
        muNew = mean(rNew, NqNew)
        CNew = covariance(rNew, muNew, NqNew)
        return [rNew, NqNew, piNew, muNew, CNew]

    thetaNew = newParam(thetaOld)
    iter = 0
    old = likelihood(thetaOld)
    new = likelihood(thetaNew)

    while iter < epoch or new - old < 0.001:
        thetaOld = thetaNew
        thetaNew = newParam(thetaOld)
        old = new
        new = likelihood(thetaNew)
        iter = iter + 1
        print(iter)
    return thetaNew

data_1 = GMM(data1, 16)
data_2 = GMM(data2, 16)

plt.scatter(data1[0], data1[1])
plt.scatter(data2[0], data2[1])

x = np.linspace(-15, 15, 200)
y = np.linspace(-15, 15, 200)

X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariateGaussian(pos, mu, sigma):
    n = mu.shape[0]
    sigmaDet = np.linalg.det(sigma)
    sigmaInv = np.linalg.inv(sigma)
    N = np.sqrt((2*np.pi)**n * sigmaDet)
    fac = np.einsum('...k,kl,...l->...', pos - mu, sigmaInv, pos - mu)
    return np.exp( -fac / 2) / N

pi1 = (data_1[2])
mean1 = (data_1[3])
cov1 = (data_1[4])
gaussian1 = [[0 for x in range(200)] for y in range(200)]
for i in range(16):
    gaussian1  = gaussian1 + pi1[i] * multivariateGaussian(pos, mean1[i], cov1[i])

pi2 = (data_2[2])
mean2 = (data_2[3])
cov2 = (data_2[4])
gaussian2 = [[0 for x in range(200)] for y in range(200)]
for i in range(16):
    gaussian2  = gaussian2 + pi2[i] * multivariateGaussian(pos,mean2[i],cov2[i])

plt.contour (X,Y,gaussian1)
plt.contour (X,Y,gaussian2)