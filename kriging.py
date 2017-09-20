#!/usr/bin/env python
#https://docs.anaconda.com/numbapro/cudalib
import numpy as np
from scipy.spatial.distance import cdist
from utilities import pairwise
from numba import jit
import numba
from accelerate.cuda.blas import Blas

print(numba.__version__)

@jit
def distanceUPoint(data, u):
    data[:,0] = (( data[:,0] - u[0][0] ) + ( data[:,1] - u[0][1] ))

def kmatrices( data, covfct, u, N=0 ):
    '''
    Input  (data)  ndarray, data
           (model) modeling function
                    - spherical
                    - exponential
                    - gaussian
           (u)     unsampled point
           (N)     number of neighboring points
                   to consider, if zero use all
    '''
    # u needs to be two dimensional for cdist()
    if np.ndim( u ) == 1:
        u = [u]
    print data[:,:]
    print u

    # distance between u and each data point in P
    print len(data)
    print u[0]
    distanceUPoint(data, u) #-->> GPU
    d = data
    
    # add these distances to P
    P = np.hstack(( data, d ))
    # if N>0, take the N closest points,
    if N > 0:
        P = P[d[:,0].argsort()[:N]]
    # otherwise, use all of the points
    else:
        N = len( P )

    # apply the covariance model to the distances
    k = covfct( P[:,3] )
    # check for nan values in k
    if np.any( np.isnan( k ) ):
        raise ValueError('The vector of covariances, k, contains NaN values')
    # cast as a matrix
    k = np.matrix( k ).T

    # form a matrix of distances between existing data points
    K = pairwise( P[:,:2] )
    # apply the covariance model to these distances
    K = covfct( K.ravel() )
    # check for nan values in K
    if np.any( np.isnan( K ) ):
        raise ValueError('The matrix of covariances, K, contains NaN values')
    # re-cast as a NumPy array -- thanks M.L.
    K = np.array( K )
    # reshape into an array
    K = K.reshape(N,N)
    # cast as a matrix
    K = np.matrix( K )

    return K, k, P

def transposeMatrix(m):
    t = []
    for r in range(len(m)):
        tRow = []
        for c in range(len(m[r])):
            if c == r:
                tRow.append(m[r][c])
            else:
                tRow.append(m[c][r])
        t.append(tRow)
    return t

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    #for c in range(len(m)):
        #determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    f_sum(m, determinant)
    return determinant


def simple( data, covfct, u, N=0, nugget=0 ):
    
    # calculate the matrices K, and k
    K, k, P = kmatrices( data, covfct, u, N )
    print len(K)
    # calculate the kriging weights
    weights =  Blas().gemv('N',len(K), len(K), 1.0, K, K, 0.0, P)
    weights = np.array( weights )

    # calculate k' * K * k for
    # the kriging variance
    kvar = k.T * weights

    # mean of the variable
    mu = np.mean( data[:,2] )
    
    # calculate the residuals
    residuals = P[:,2] - mu

    # calculate the estimation
    estimation = np.dot( weights.T, residuals ) + mu

    # calculate the sill and the 
    # kriging standard deviation
    sill = np.var( data[:,2] )
    kvar = float( sill + nugget - kvar )
    kstd = np.sqrt( kvar )

    return float( estimation ), kstd


def krige( data, covfct, grid, method='simple', N=0, nugget=0 ):
    '''
    Krige an <Nx2> array of points representing a grid.
    
    Use either simple or ordinary kriging, some number N
    of neighboring points, and a nugget value.
    '''
    k = lambda d, c, u, N, nug: simple( d, c, u, N, nug )
    M = len( grid )
    est, kstd = np.zeros(( M, 1 )), np.zeros(( M, 1 ))
    for i in range( M ):
        est[i], kstd[i] = k( data, covfct, grid[i], N, nugget )
    return est, kstd
