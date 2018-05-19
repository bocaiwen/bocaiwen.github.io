import numpy as np
import scipy as sci
import numpy.matlib

from scipy.special import logsumexp


#import matplotlib.pyplot as plt

#%matplotlib inline
#plt.rcParams["figure.figsize"] =[12,9]

transp = np.array([[0.99, 0.01], [0.02, 0.98]])
emissp = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

pi = np.array([0.5, 0.5])

def simulate(N):
    hs = []
    xs = []
    ht = np.random.choice([0,1], p=pi)
    hs.append(ht)
    for _ in range(N-1):
        ht = np.random.choice([0, 1],p=transp[ht, :])
        hs.append(ht)
    for h in hs:
        xt = np.random.choice(range(6), p=emissp[h, :])
        xs.append(xt)
    return hs, xs

def HMMViterbi(v):
    N = len(v)
    trace = np.zeros((N, 2))
    trace[0, :] = np.log(pi) + np.log(emissp[:, v[0]])
    traj = np.ones((N, 2)) * -1
    for i in range(1, N):
        #s0 = np.log(transp[:, 0]) + np.log(emissp[0, v[i]]) + trace[i-1, :]
        #trace[i, 0] = np.max(s0)
        #traj[i, 0] = np.argmax(s0)
        #s1 = np.log(transp[:, 1]) + np.log(emissp[1, v[i]]) + trace[i-1, :]
        #trace[i, 1] = np.max(s1)
        #traj[i, 1] = np.argmax(s1)
        s = np.log(transp.T) + np.log(emissp[:, v[i]]) + trace[i-1,:]
        trace[i,:] = np.max(s.T, axis=0)
        traj[i, :] = np.argmax(s.T, axis=0)
    hs = []
    last = np.argmax(trace[N-1,:])
    hs.append(last)
    for i in range(N-1, 0, -1):
        last = int(traj[i, last])
        hs.append(last)
    return list(reversed(hs))

def divide(n, d):
    r = np.zeros(n.shape)
    for i in range(len(n)):
        r[i, :] = n[i, :] / d[i]
    return r


def condp(pin):
    if len(pin.shape) == 1:
        return np.divide(pin, np.sum(pin))
    p = np.sum(pin, axis=1)
    return divide(pin, p)


def condexp(logp):
    if len(logp.shape) == 1:
        maxlogp = np.max(logp)
        logp -= maxlogp
    else:
        maxlogp = np.max(logp, axis=1)
        logp = (logp.T - maxlogp).T
    return condp(np.exp(logp))


def HMMforward(v, N, K, pi_, emissp_, transp_):
    logalpha = np.ones((N, K)) * -np.inf
    logalpha[0, :] = np.log(pi_) + np.log(emissp_[:, v[0]])
    for i in range(1, N):
        logalpha[i, :] = np.log(emissp_[:, v[i]]) + logsumexp(logalpha[i-1, :] + np.log(transp_.T), axis=1)
    return logalpha, logsumexp(logalpha[N-1,:])

def HMMbackward(v, N, K, emissp_, transp_):
    logbeta = np.ones((N, K)) * -np.inf
    logbeta[N-1, :] = np.zeros(K)
    for i in range(N-2, -1, -1):
        logbeta[i, :] = logsumexp(logbeta[i+1, :] + np.log(emissp_[:, v[i+1]]) + np.log(transp_), axis=1)
    return logbeta

def HMMsmooth(logalpha, logbeta, v, N, K, emissp_, transp_):
    r = np.zeros((N, K))
    for i in range(N):
        r[i, :] = logalpha[i, :] + logbeta[i, :]
    r = condexp(r)
    B = np.zeros((K, K, N))
    for i in range(1, N):
        #B[:, :, i] = (condexp(logalpha[i-1, :]) * condexp(logbeta[i, :]) * emissp_[:, v[i]] * transp_.T).T
        #B[:, :, i] = B[:, :, i] / np.sum(B[:, :, i])
        for k1 in range(K):
            for k2 in range(K):
                B[k1, k2, i] = logalpha[i-1, k1] + logbeta[i, k2] + np.log(emissp_[k2, v[i]]) + np.log(transp_[k1, k2])
        logmax = np.max(B[:, :, i])
        B[:, :, i] -= logmax
        B[:, :, i] = np.exp(B[:, :, i])
        B[:, :, i] = B[:, :, i] / np.sum(B[:, :, i])
    return r, B

def HMMsmooth(logalpha, logbeta, v, N, K, emissp_, transp_):
    r = np.zeros((N, K))
    for i in range(N):
        r[i, :] = logalpha[i, :] + logbeta[i, :]
    r = condexp(r)
    B = np.zeros((K, K, N))
    for i in range(1, N):
        for k1 in range(K):
            #for k2 in range(K):
            #    B[k1, k2, i] = logalpha[i-1, k1] + logbeta[i, k2] + np.log(emissp_[k2, v[i]]) + np.log(transp_[k1, k2])
            B[k1, :, i] = logalpha[i-1, k1] + logbeta[i, :] + np.log(emissp_[:, v[i]]) + np.log(transp_[k1, :])
        logmax = np.max(B[:, :, i])
        B[:, :, i] -= logmax
        B[:, :, i] = np.exp(B[:, :, i])
        B[:, :, i] = B[:, :, i] / np.sum(B[:, :, i])
    return r, B

def HMMsmooth(logalpha, logbeta, v, N, K, emissp_, transp_):
    r = np.zeros((N, K))
    for i in range(N):
        r[i, :] = logalpha[i, :] + logbeta[i, :]
    r = condexp(r)
    A = np.zeros((K, K, N))
    for i in range(1, N):
        #for k1 in range(K):
            #for k2 in range(K):
            #    A[k1, k2, i] = logalpha[i-1, k1] + logbeta[i, k2] + np.log(emissp_[k2, v[i]]) + np.log(transp_[k1, k2])
        #    A[k1, :, i] = logalpha[i-1, k1] + logbeta[i, :] + np.log(emissp_[:, v[i]]) + np.log(transp_[k1, :])
        A[:, :, i] = np.matlib.repmat(logalpha[i-1, :].reshape(-1,1), 1, 2) + logbeta[i, :] + np.log(emissp_[:, v[i]]) + np.log(transp_[:, :])
        logmax = np.max(A[:, :, i])
        A[:, :, i] -= logmax
        A[:, :, i] = np.exp(A[:, :, i])
        A[:, :, i] = A[:, :, i] / np.sum(A[:, :, i])
    return r, A

def HMMem(V, N, K, D, niters):
    ph1 = condp(np.random.rand(K)) # pi
    #ph1 = pi
    phthtp = condp(np.random.rand(K, K)) # transp
    #phthtp = condp(np.array([[0.6, 0.4],[0.4, 0.6]])) # transp
    #phthtp = transp
    pvtht = condp(np.random.rand(K, D)) # emissp
    #pvtht = emissp
    times = 0
    lastllik = -np.inf
    for i in range(niters):
        a = np.zeros(K)
        A = np.zeros((K, K))
        B = np.zeros((K, D))
        llik = 0
        for m in range(len(V)):
            v = V[m]
            # E-step
            logalpha, llik_ = HMMforward(v, N, K, ph1, pvtht, phthtp)
            logbeta = HMMbackward(v, N, K, pvtht, phthtp)
            #print("logalpha: {}".format(logalpha))
            #print("logbeta: {}".format(logbeta))
            r, A_ = HMMsmooth(logalpha, logbeta, v, N, K, pvtht, phthtp)
            llik += llik_
            #print("r: {}".format(r))
            #print("A_: {}".format(A_))
            # collect
            a += r[0, :]
            A += np.sum(A_, axis=2)
            #print("A : {}".format(A))
            for j in range(N):
                B[:, v[j]] += r[j, :]
        # M-step
        ph1 = condp(a)
        phthtp = condp(A)
        pvtht = condp(B)
        llik /= len(V)
        #print("ph1: {}".format(ph1))
        #print("phthtp: {}".format(phthtp))
        #print("pvtht: {}".format(pvtht))
        print("---- log likelihood: {}".format(llik))
        if llik - lastllik < 0.003:
            times += 1
        else:
            times = 0
        if times >= 5:
            break
        lastllik = llik
    return ph1, phthtp, pvtht, llik

M = 100
N = 800
vs = np.zeros((M, N), dtype=int)
for m in range(M):
    _, v = simulate(N)
    vs[m, :] = np.array(v, dtype=int)

best = -np.inf
for _ in range(10):
    ph1, phthtp, pvtht, llik = HMMem(vs, len(vs), 2, 6, 2000)
    print("----------------------------------")
    if llik > best:
        best = llik
        print("ph1: {}".format(ph1))
        print("phthtp: {}".format(phthtp))
        print("pvtht: {}".format(pvtht))
        print("---- log likelihood: {}".format(llik))
