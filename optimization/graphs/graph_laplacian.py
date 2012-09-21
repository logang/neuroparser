import numpy as np

def construct_adjacency_list(nx,ny,nz,return_full=False):
    """
    Generate and return a list of lists for which the ith list contains the indices of the 
    adjacent voxels.
    """
    nvoxels = nx*ny*nz

    from scipy.sparse import coo_matrix

    y_coords = np.reshape(np.tile(np.arange(ny, dtype=np.int32), (nx*nz, 1)), (nvoxels), order='f')
    x_coords = np.reshape(np.tile(np.reshape(np.tile(np.arange(nx, dtype=np.int32), (nz, 1)),
                                             (nz*nx), order='f'), (ny, 1)), (nx*ny*nz))
    z_coords = np.tile(np.arange(nz, dtype=np.int32), (nx*ny))

    F = coo_matrix((nvoxels, nvoxels), dtype=np.float32)
    diag_coords = y_coords*nx*nz + x_coords*nz + z_coords

    # Form the z+1 difference entries
    valid_idxs = np.nonzero(z_coords+1 < nz)
    diff_coords = y_coords*nx*nz + x_coords*nz + (z_coords+1)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the z-1 difference entries
    valid_idxs = np.nonzero(z_coords-1 >= 0)
    diff_coords = y_coords*nx*nz + x_coords*nz + (z_coords-1)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the x+1 difference entries
    valid_idxs = np.nonzero(x_coords+1 < nx)
    diff_coords = y_coords*nx*nz + (x_coords+1)*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the x-1 difference entries
    valid_idxs = np.nonzero(x_coords-1 >= 0)
    diff_coords = y_coords*nx*nz + (x_coords-1)*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the y+1 difference entries
    valid_idxs = np.nonzero(y_coords+1 < ny)
    diff_coords = (y_coords+1)*nx*nz + x_coords*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the y-1 difference entries
    valid_idxs = np.nonzero(y_coords-1 >= 0)
    diff_coords = (y_coords-1)*nx*nz + x_coords*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    A = F.tolil().rows.tolist()
    if return_full:
        return A, F.todense()
    else:
        return A

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

def sparse_Laplacian_matrix(nx, ny, nz):
    '''
    Builds a sparse, square matrix of local first differences,
    equivalent to the graph Laplacian (Degree - Adjacency) matrix.
    This is defined as:
    
            n   for i==j where n is the number of voxels adjacent to x_i (node degree)
    F_ij = -1   for i \neq j but adjacent to j
            0   otherwise
    '''
    nvoxels = nx*ny*nz

    from scipy.sparse import coo_matrix

    y_coords = np.reshape(np.tile(np.arange(ny, dtype=np.int32), (nx*nz, 1)), (nvoxels), order='f')
    x_coords = np.reshape(np.tile(np.reshape(np.tile(np.arange(nx, dtype=np.int32), (nz, 1)),
                                             (nz*nx), order='f'), (ny, 1)), (nx*ny*nz))
    z_coords = np.tile(np.arange(nz, dtype=np.int32), (nx*ny))

    # Form the diagonal entries of F (should be equal to number of neighbors)
    diag_coords = y_coords*nx*nz + x_coords*nz + z_coords
    F = coo_matrix((np.ones(nvoxels)*6, (diag_coords, diag_coords)),
                   shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the z+1 difference entries
    valid_idxs = np.nonzero(z_coords+1 < nz)
    diff_coords = y_coords*nx*nz + x_coords*nz + (z_coords+1)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the z-1 difference entries
    valid_idxs = np.nonzero(z_coords-1 >= 0)
    diff_coords = y_coords*nx*nz + x_coords*nz + (z_coords-1)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the x+1 difference entries
    valid_idxs = np.nonzero(x_coords+1 < nx)
    diff_coords = y_coords*nx*nz + (x_coords+1)*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the x-1 difference entries
    valid_idxs = np.nonzero(x_coords-1 >= 0)
    diff_coords = y_coords*nx*nz + (x_coords-1)*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the y+1 difference entries
    valid_idxs = np.nonzero(y_coords+1 < ny)
    diff_coords = (y_coords+1)*nx*nz + x_coords*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Form the y-1 difference entries
    valid_idxs = np.nonzero(y_coords-1 >= 0)
    diff_coords = (y_coords-1)*nx*nz + x_coords*nz + z_coords
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diff_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Fix edge coeffs so that the entire matrix, when multiplied by a vector of only ones, equals zero.

    # Fix z = 0 coeffs
    valid_idxs = np.nonzero(z_coords == 0)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diag_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Fix z = nz coeffs
    valid_idxs = np.nonzero(z_coords == nz-1)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diag_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Fix y = 0 coeffs
    valid_idxs = np.nonzero(y_coords == 0)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diag_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Fix y = ny coeffs
    valid_idxs = np.nonzero(y_coords == ny-1)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diag_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Fix x = 0 coeffs
    valid_idxs = np.nonzero(x_coords == 0)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diag_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)

    # Fix x = nx coeffs
    valid_idxs = np.nonzero(x_coords == nx-1)
    F = F + coo_matrix((np.ones(len(valid_idxs[0]))*-1.0, (diag_coords[valid_idxs], diag_coords[valid_idxs])),
                       shape = (nvoxels, nvoxels), dtype=np.float32)
    # Tests
    try:
        ones = np.ones((F.shape[0],))
        Finner1 = (F*ones).reshape(ny,nx,nz)
        assert(np.sum(Finner1) == 0.0)
    except:
        print "First differences matrix has non-zero inner product with vector of ones!!!"
    try:
        ones = np.ones((F.shape[0],))
        Finner1 = (F.T*ones).reshape(ny,nx,nz)
        assert(np.sum(Finner1) == 0.0)
    except:
        print "First differences matrix transpose has non-zero inner product with vector of ones!!!"

    return F

if __name__ is '__main__':
    L = construct_adjacency_list(10,10,1)
    1/0
