import numpy as np

def construct_laplacian_3D(vol):

    # Smoothing regularization.  Compute the laplacian of the volume.
    lapvol = np.copy(vol)
    lapvol[0:ny-1,:,:] -= 1/6.0 * vol[1:ny  ,:,:]
    lapvol[1:ny  ,:,:] -= 1/6.0 * vol[0:ny-1,:,:]
    lapvol[:,0:nx-1,:] -= 1/6.0 * vol[:,1:nx  ,:]
    lapvol[:,1:nx  ,:] -= 1/6.0 * vol[:,0:nx-1,:]
    lapvol[:,:,0:nz-1] -= 1/6.0 * vol[:,:,1:nz  ]
    lapvol[:,:,1:nz  ] -= 1/6.0 * vol[:,:,0:nz-1]

    # Zero out laplacian around the edges.
    lapvol[0,:,:] = 0.0;
    lapvol[:,0,:] = 0.0;
    lapvol[:,:,0] = 0.0;
    lapvol[ny-1,:,:] = 0.0;
    lapvol[:,nx-1,:] = 0.0;
    lapvol[:,:,nz-1] = 0.0;

if __name__ is '__main__':
    pass
