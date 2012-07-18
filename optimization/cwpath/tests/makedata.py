import numpy as np
import scipy.signal as signal

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

def spatial_data(n=100, im_side=100, SNR=1., outlier_frac=0.1):
    # input signal image and noise image, p must have a square root.
    pos_im = smoothed_point_process_2D( 0.001, im_side, im_side, 20 )
    neg_im = smoothed_point_process_2D( 0.001, im_side, im_side, 20 )
    side = pos_im.shape[0]
    p = side**2
    y = []
    for i in xrange(n):
        y.append(np.random.uniform())
        if i==0:
            Xsig = ((1-y[i])*pos_im.flatten() + y[i]*neg_im.flatten()).reshape((1,p))
            Xnoise = ((1-outlier_frac)*np.random.normal(0.0,1.0,size=p) + outlier_frac*np.random.laplace(0.0,5.0,size=p)).reshape((1,p))
        else:
            sig = ((1-y[i])*pos_im.flatten() + y[i]*neg_im.flatten()).reshape((1,p))
            Xsig = np.vstack((Xsig, sig))
            noise = ((1-outlier_frac)*np.random.normal(0.0,1.0,size=p) + outlier_frac*np.random.laplace(0.0,5.0,size=p)).reshape((1,p))
            Xnoise = np.vstack((Xnoise, noise))

    # standardize
    X = Xsig + Xnoise/SNR
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    
    # output
    y = np.array(y)
    y -= np.mean(y)
    y /= np.std(y)

    return X,y,pos_im,neg_im,outlier_frac    

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

    X,Y,pos_sig,neg_sig,outlier_frac = spatial_data()
    np.savez("Data",X=X,Y=Y,pos_sig=pos_sig, neg_sig=neg_sig, outlier_frac=outlier_frac)

#    X,y = basic_data(50,100,SNR=100)
#    plot = False

    # if plot:
    #     import pylab as pl
    #     pl.subplot(411)
    #     pl.scatter(X[:,0],y)
    #     pl.subplot(412)
    #     pl.scatter(X[:,1],y)
    #     pl.subplot(413)
    #     pl.scatter(X[:,2],y)
    #     pl.subplot(414)
    #     pl.scatter(X[:,3],y)
    #     pl.show()

    #np.save('X',X)
    #np.save('Y',y)
