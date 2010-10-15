import numpy as np
# ==============================================================================
def lobpcg( operator_A, X, constraints = 'classical' ):
    k = X.shape[1]
    lambd = np.zeros( k )
    return lambd, X
# ==============================================================================
def _inner( x, y ):
    '''
    Inner product of two vectors x, y
    '''
    return 0.0
# ==============================================================================