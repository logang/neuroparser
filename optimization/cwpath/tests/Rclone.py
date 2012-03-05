import numpy as np

def lassoR(X, Y, l1, tol=1.0e-10):
    """
    "Literal" translation of R script into python.
    """
    
    it = 0
    Y -= Y.mean() 	
    X = np.asarray(X)
    n, p = X.shape
    beta = np.zeros(p)
    S = np.dot(X.T, Y)
    r = Y

    err = np.inf

    vals = []
    C = np.zeros(X.shape)
    while err > tol:
        it += 1
        vals.append(beta.copy())
        for j in range(p):
            r = Y - np.dot(X, beta)
            S = (X[:,j] * (r + X[:,j] * beta[j])).sum()
            beta[j] = np.sign(S) * pospart(np.fabs(S) - l1 / np.sqrt(n)) / n
        if it > 1:
            err = np.fabs(beta - vals[-1]).sum() / p
    return np.array(vals[-1])
