import numpy as np
chol = np.linalg.cholesky

def gen_correlated_matrix( n, p, corr1 = 0.8, corr2 = None ):
        """
        Generate correlated random normal matrix using the Cholesky decomposition
        of the partial autocorrelation matrix 'sigma', which should be p \times p.
        """
        if corr2 is None:
                corr2 = corr1
                
        # generate partial autocorrelation matrices
        ys, xs = np.mgrid[:p, :p]
        sigma1 = corr1 ** abs(ys - xs)
        ys, xs = np.mgrid[:n, :n]
        sigma2 = corr2 ** abs(ys - xs)
        
        # get cholesky factoriztions
        C1 = np.matrix(chol(sigma1))
        C2 = np.matrix(chol(sigma2))

        # generate data with (independently) correlated rows and colums
        # (should probably add a spectral version)
        X = np.random.randn(n,p)
        return C2*X*C1
        
def gen_correlated_instance( n, p, corr1, corr2 = None, signal = None ):
        """
        Generate a particular n \times p image instance with correlated noise.
        """
        if signal is None:
                return gen_correlated_matrix( n, p, corr1 = corr1, corr2 = corr2 )
        else:
                return signal + gen_correlated_matrix( n, p, corr1 = corr1, corr2 = corr2 )
        
def gen_correlated_dataset( num_obs, n, p, corr1 = 0.8, corr2 = None, signal=None ):
        """
        Generate a data set consisting of num_obs instances of signal + correlated noise
        with correlation parameter 'corr' and size n \times p, 
        """
        dataset = []
        for i in xrange(num_obs):
                dataset.append( gen_correlated_instance( n, p, corr1, corr2,  signal ) )
        return dataset

def classification_dataset( n, p, num_obs_class1, num_obs_class2, SNR = 1, corr1 = 0.9, corr2 = None ):
        """
        Generate a two class data set with spatially correlated signal and noise.  
        """
        # class 1 signal
        signal1 = np.zeros( (n,p) )
        signal1[10:15, 10:15] = 1.*SNR
        signal1[30:40, 30:40] = -1.*SNR
        signal1[60:80, 60:80] = 1.*SNR

        # class 2 signal
        signal2 = np.zeros( (n,p) )
        signal2[30:45, 30:45] = 1.*SNR
        signal2[30:50, 60:80] = 1.*SNR

        # make data
        x = gen_correlated_dataset( num_obs_class1, n, p, corr1 = corr1, corr2 = corr2, signal=signal1 )
        x.extend( gen_correlated_dataset( num_obs_class2, n, p, corr1 = corr1, corr2 = corr2, signal=signal2 ) )
        class_labels = np.hstack( ( np.zeros( num_obs_class1 ), np.ones( num_obs_class2 ) ) ).T

        # generate data matrix
        for i in xrange( len(x) ):
                if i == 0:
                        X = x[0].flatten()
                else:
                        X = np.vstack( (X, x[i].flatten()) )

        return (class_labels, X)

if __name__ == "__main__":
        import pylab as pl
        # image parameters
        n = 100
        p = 100
        # random iid matrix for comparison
        y = np.random.randn(n,p)
        # generate data
        signal = np.zeros( (n,p) )
        signal[10:15, 10:15] = 1.
        signal[30:40, 30:40] = -1.
        signal[60:80, 60:80] = 1.
        x = gen_correlated_dataset( 100, n, p, signal=signal)
        # plot data
        pl.subplot(211)
        pl.imshow(y,interpolation=None)
        pl.subplot(212)
        pl.imshow(x[0],interpolation=None)
        pl.show()

        # classification data
        num_obs_class1 = 100
        num_obs_class2 = 100
        class_data = classification_dataset( n, p, num_obs_class1, num_obs_class2, SNR = 1, corr1 = 0.9, corr2 = None )
        1/0
