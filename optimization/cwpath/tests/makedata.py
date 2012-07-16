import numpy as np

# Basic data                                                                                                                                                                                            
def basic_data(n,p,SNR=1.):
    # input signal and noisy signal                                                                                                                                                                    
    Xsig = np.random.randn(n,p)
    Xsig -= np.mean(Xsig, axis=0)
    Xsig /= np.std(Xsig, axis=0)

    X = Xsig + np.random.randn(n,p)/SNR
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    # coefficients                                                                                                                                                                                     
    coefs = np.zeros((p,))
    coefs[0:5] = 5.

    # output                                                                                                                                                                                           
    y = np.dot(Xsig, coefs)
    y -= np.mean(y)
    y /= np.std(y)

    return X,y

if __name__ == '__main__':
    X,y = basic_data(50,100,SNR=100)
    plot = False

    if plot:
        import pylab as pl
        pl.subplot(411)
        pl.scatter(X[:,0],y)
        pl.subplot(412)
        pl.scatter(X[:,1],y)
        pl.subplot(413)
        pl.scatter(X[:,2],y)
        pl.subplot(414)
        pl.scatter(X[:,3],y)
        pl.show()

    np.save('X',X)
    np.save('Y',y)
