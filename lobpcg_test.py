from scipy.sparse.linalg import lobpcg as scipy_lobpcg
from lobpcg import lobpcg as new_lobpcg
from lobpcg_scipy import lobpcg as new_lobpcg2
from scipy.sparse import identity, spdiags
from scipy.linalg import norm
import numpy as np

## ==============================================================================
n = 100
k = 5
X = np.random.rand( n, k )
A = identity( n )
## ------------------------------------------------------------------------------
#eigenvalues, eigenvectors = new_lobpcg( A, X )

#for kk in xrange( k ):
    #print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
#print
## ------------------------------------------------------------------------------
#eigenvalues, eigenvectors = scipy_lobpcg( A, X, verbosityLevel = 0 )

#for kk in xrange( k ):
    #print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
##print
## ==============================================================================
n = 100
k = 5
X = np.ones( (n, k) )
X[:,0] = range(1,n+1)
X[:,1] = range(1,2*n+1,2)
X[:,2] = range(1,3*n+1,3)
X[:,3] = range(1,5*n+1,5)
X[:,4] = range(1,6*n+1,6)


print X

e = np.ones( n )
data = np.array( [-e, 2.0*e, -e ] )
A = spdiags( data, [-1,0,1], n, n )

#A = np.random.rand( n, n ) + 1j * np.random.rand( n, n )
#A = 0.5 * ( A + A.T.conjugate() )

# ------------------------------------------------------------------------------
eigenvalues, eigenvectors = new_lobpcg( A, X, maxiter = 1, tolerance = 1.0e-10, verbosity = 1 )

for kk in xrange( k ):
    print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
print
# ------------------------------------------------------------------------------
#eigenvalues, eigenvectors = scipy_lobpcg( A, X, verbosityLevel = 0 )

#for kk in xrange( k ):
    #print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
#==============================================================================
#n = 10
#k = 2
##X = np.random.rand( n, k ) + 1j * np.random.rand( n, k )
#X = np.ones( (n, k) )
#X[0,1] = 0.0

#e = np.ones( n )
#data = np.array( [-1j*e, 2.0*e, 1j*e ] )
#A = spdiags( data, [-1,0,1], n, n )
## ------------------------------------------------------------------------------
#eigenvalues, eigenvectors = new_lobpcg( A, X, verbosity = 0, maxiter = 1000 )

#for kk in xrange( k ):
    #print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
#print
# ------------------------------------------------------------------------------
#eigenvalues, eigenvectors = scipy_lobpcg( A, X, verbosityLevel = 0 )

#for kk in xrange( k ):
    #print "Res: ", norm( A*eigenvectors[:,kk] - eigenvalues[kk]*eigenvectors[:,kk] )
# ==============================================================================