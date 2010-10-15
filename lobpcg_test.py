from scipy.sparse.linalg import lobpcg as scipy_lobpcg
from lobpcg import lobpcg as new_lobpcg
from scipy.sparse import identity, spdiags
from scipy.linalg import norm
import numpy as np

# ==============================================================================
n = 100
k = 3
X = np.random.rand( n, k )
A = identity( n )
# ------------------------------------------------------------------------------
eigenvalues, eigenvectors = new_lobpcg( A, X )

for kk in xrange( k ):
    print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
print
# ------------------------------------------------------------------------------
eigenvalues, eigenvectors = scipy_lobpcg( A, X, verbosityLevel = 0 )

for kk in xrange( k ):
    print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
print
# ==============================================================================
n = 10
k = 3
X = np.random.rand( n, k ) + 1j * np.random.rand( n, k )

e = np.ones( n )
data = np.array( [-1j*e, 2.0*e, 1j*e ] )
A = spdiags( data, [-1,0,1], n, n )
# ------------------------------------------------------------------------------
eigenvalues, eigenvectors = new_lobpcg( A, X )

for kk in xrange( k ):
    print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
print
# ------------------------------------------------------------------------------
eigenvalues, eigenvectors = scipy_lobpcg( A, X, verbosityLevel = 0 )

for kk in xrange( k ):
    print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
# ==============================================================================