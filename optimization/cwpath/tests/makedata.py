import numpy as np
import scipy.signal as signal
import pylab as pl

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

def spatial_data(n=1000, im_side=100, SNR=10., outlier_frac=0.05, thresh=0.1):
    # input signal image and noise image, p must have a square root.
    pos_im = smoothed_point_process_2D( 0.0009, im_side, im_side, 20 )
    neg_im = smoothed_point_process_2D( 0.0009, im_side, im_side, 20 )
    im = pos_im - neg_im
    im[np.fabs(im) < thresh*np.max(np.fabs(im))] = 0
    side = pos_im.shape[0]
    p = side**2
    y = []
    for i in xrange(n):
#        y.append(np.random.uniform())
        coin = np.random.binomial(1,0.5)
        outlier_coin = np.random.binomial(1,outlier_frac)
        y.append(float(coin))

        if i==0:
            Xsig = (y[i]*im.flatten('F') + np.zeros(im.shape).flatten('F')).reshape((1,p))
            Xnoise = np.random.normal(0.0,1.0,size=p).reshape((1,p))
            outlier_sign = 1
            if outlier_coin:
                print "outlier!"
                Xnoise += np.random.laplace(0.0,10.0,size=p).reshape((1,p))
        else:
            sig = (y[i]*im.flatten('F') + np.zeros(im.shape).flatten('F')).reshape((1,p))
            Xsig = np.vstack((Xsig, sig))
            noise = np.random.normal(0.0,1.0,size=p).reshape((1,p))
            if outlier_coin:
                print "outlier!"
                outlier_sign *= -1
                noise += np.random.laplace(0.0,10.0,size=p).reshape((1,p))
            Xnoise = np.vstack((Xnoise, noise))

    # standardize
    X = Xsig + Xnoise/SNR
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    
    # output
    y = np.array(y)
    y -= np.mean(y)
    y /= np.std(y)

    return X,y,im,outlier_frac    

def gauss_kern( size, sizey = None ):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g    = np.exp(-(x**2/float(size)+y**2/float(sizey)))

    return g / g.sum()

def gauss_blur(im, n, ny = None):
    """ 
    Blurs the image by convolving with a gaussian kernel of typical
    size n. The optional keyword argument ny allows for a different
    size in the y direction.
    """
    g = gauss_kern( n, sizey = ny)
    improc = signal.convolve( im, g, mode='valid')

    return(improc)

def point_process_2D( eta, x, y ):
    """ 
    Creates a random Poisson process in 2D with intensity parameter eta. 
    """
    return np.random.poisson( eta, size = (x,y) )

def smoothed_point_process_2D(eta=None, x=None, y=None, blur_width=None): 
    """ 
    Creates a smoothed random Poisson process in 2D with intensity parameter eta. 
    """
    if eta is not None:
        eta = eta
    else:
        eta = 0.001
    if x is not None:
        x = x
    else:
        x = 100
    if y is not None:
        y = y
    else:
        y = 100
    im = point_process_2D( eta, x, y )
    return gauss_blur( im, blur_width )

if __name__ == '__main__':
    spatial_test = True
    basic_test = False

    if spatial_test:
        # make 1000 points for training, 1000 for validation, and 1000 for test
        X,Y,sig_im,outlier_frac = spatial_data(n=3000)
        np.savez("Data",X=X,Y=Y,sig_im=sig_im, outlier_frac=outlier_frac)
        pl.imsave("sig.png",sig_im)

    if basic_test:
        X,y = basic_data(50,100,SNR=100)
        np.save('X',X)
        np.save('Y',y)
