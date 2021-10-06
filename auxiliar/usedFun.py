import numpy as np
from scipy import interpolate

# =============================================================================
# Numerical derivation
# =============================================================================
def NumDer(xc,yc,n):
    """ 
    Evaluate te n- numerical derivate
    
    INPUT:
    xc: x cordinate
    yc: y cordinate (array same len as x)
    n: derivate order
    
    OUTPUT:
    der: array of same len as xc and yc
    """
    xDer = xc
    yDer = yc

    for i in range(n):
        yDer = np.diff(yDer)/np.diff(xDer)
        xDer = 0.5*(xDer[:-1] + xDer[1:])      
    
    f = interpolate.interp1d(xDer, yDer , fill_value = 'extrapolate')
            
    yDer = np.interp(xc, xDer, yDer)
    xDer = xc   
    
    for i in range(n):
        yDer[i] = f(xDer[i])
        yDer[-i-1] = f(xDer[-i-1])

    return yDer


# =============================================================================
# Solves Linear Homogenous System
# =============================================================================

def SolveLinSys(A, eps = 1E-8):
    """
    Solves the homogenuos linear system by single value decomposition method
    (SVD) related to A matrix

    INPUT: A: Square matrix
    OUTPUT: Possible solution by SVD method
    """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)

    
    return null_space.T