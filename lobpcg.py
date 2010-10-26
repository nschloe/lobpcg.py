import numpy as np
from scipy.linalg import cholesky, orth, eig, eigh, solve, svd
from scipy.sparse import spdiags
from math import log10, log
from scipy import mean
# ==============================================================================
def lobpcg( A,
            blockvector_x,
            constraints = None, # 'conventional', 'symmetric'
            T = None, # preconditioner
            B = None, # generalized eigenvalue problem
            maxiter = None,
            verbosity = 0,
            tolerance = None,
            blockvector_y = None
          ):

    n = A.shape[0]
    block_size = blockvector_x.shape[1]

    if not maxiter:
        maxiter = min( n, 20 )

    eps = np.finfo(np.double).eps
    if not tolerance:
        residual_tolerance = np.sqrt(eps) * n
    else:
        residual_tolerance = tolerance

    size_y = 0

    if verbosity > 0:
        print "Starting LOBPCG..."

    # constraints preprocessing
#constraints preprocessing
#if isempty(blockvector_y)
    #constraintStyle = 0
#else
    #%    constraintStyle = SYMMETRIC_CONSTRAINTS % more accurate?
    #constraintStyle = CONVENTIONAL_CONSTRAINTS
#end
    if constraints == 'conventional':
        if B is None:
            gram_y = _block_vdot( blockvector_y, blockvector_y )
        else:
            # Y * B * Y
            blockvector_by = B * blockvector_y
            gram_y = _block_vdot( blockvector_y, blockvector_by )

        assert np.allclose( gram_y, gram_y.T.conjugate() )

        if B is None:
            YX = _block_vdot( blockvector_y, blockvector_x )
        else:
            YX = _block_vdot( blockvector_by, blockvector_x )
        tmp2 = gram_y.solve( YX )
        blockvector_x -= blockvector_y * tmp2
    elif constraints == 'symmetric':
        raise NotImplementedError()

    # --------------------------------------------------------------------------
    # B-orthonormalizing the initial vectors
    if B is None:
        #blockvector_x, gram_xbx = qr( blockvector_x,
                                      #overwrite_a = True,
                                      #econ = True
                                    #)

        gram_xbx = _block_vdot( blockvector_x, blockvector_x )
        assert np.allclose( gram_xbx, gram_xbx.T.conjugate() )

        try:
            gram_xbx = cholesky( gram_xbx )
        except np.linalg.LinAlgError:
            print '\nThe initial approximation after constraints is not full rank.\n'
            raise
        #print 'gram_xbx'
        #print gram_xbx
        # TODO A certain difference between the MATLAB and this implementation
        # originates here: Look closely at all the digits on the solution
        # blockvector_x. This can become significant for the iteration.
        # Make sure this solve operation carries through cleanly.
        blockvector_x = solve( gram_xbx.T, blockvector_x.T ).T
        print 'blockvector_x'
        print blockvector_x
    else:
        blockvector_x, blockvector_bx = orth( B, blockvector_x )
        blockvector_bx = B * blockvector_x
        gram_xbx = _block_vdot( blockvector_x, blockvector_bx )
        assert np.allclose ( gram_xbx, gram_xbx.T.conjugate() )

        try:
            gram_xbx = cholesky( gram_xbx )
        except np.linalg.LinAlgError:
            print 'The initial approximation after constraints is not full rank ' + \
                  'or/and operatorB is not positive definite'
            raise
        blockvector_x = blockvector_x / gram_xbx
        blockvector_bx = blockvector_bx / gram_xbx
    # --------------------------------------------------------------------------
    # Checking if the problem is big enough for the algorithm,
    # i.e. n-size_y > 5*block_size
    # Theoretically, the algorithm should be able to run if
    # n-size_y > 3*block_size,
    # but the extreme cases might be unstable, so we use 5 instead of 3 here.
    if n-size_y < 5*block_size:
        print 'The problem size is too small, relative to the block size.' + \
              'Try using eig() or eigs() instead.'
        raise
    # --------------------------------------------------------------------------
    # Preallocation
    residual_norms_history = np.zeros( (block_size, maxiter) )
    lambda_history = np.zeros( (block_size, maxiter+1) )
    condest_g_history = np.zeros( maxiter+1 )

    blockvector_br = np.zeros( (n, block_size), dtype = complex )
    blockVector_ar = np.zeros( (n, block_size), dtype = complex )
    blockvector_p  = np.zeros( (n, block_size), dtype = complex )
    blockvector_ap = np.zeros( (n, block_size), dtype = complex )
    blockvector_bp = np.zeros( (n, block_size), dtype = complex )
    # --------------------------------------------------------------------------
    # Initial settings for the loop
    blockvector_ax = A * blockvector_x

    gram_xax = _block_vdot( blockvector_x, blockvector_ax )
    assert np.allclose ( gram_xax, gram_xax.T.conjugate() )

    eigenvalues, eigenvectors  = _symeig( gram_xax )
    assert( (eigenvalues.imag==0).all() )
    eigenvalues = eigenvalues.real

    #if issparse(blockvector_x):
        #coordX = sparse( eigenvectors )

    blockvector_x = np.dot( blockvector_x, eigenvectors )

    blockvector_ax = np.dot( blockvector_ax, eigenvectors )

    
    if B is not None:
        blockvector_bx = blockvector_bx * eigenvectors
    # clear coordX

    condest_g_history[0] = -log10(eps) / 2 #if too small cause unnecessary restarts
    lambda_history[ :block_size, 1 ] = eigenvalues

    active_mask = np.ones( block_size, dtype=bool )
    current_block_size = block_size  # iterate all
    restart = 1 # steepest descent

    # The main part of the method is the loop of the CG method: begin
    for iteration_number in xrange( maxiter ):
        # ----------------------------------------------------------------------
        ## Computing the active residuals
        #if B is None:
            #if current_block_size > 1:
                #blockvector_r[:, active_mask] = blockvector_ax[:, active_mask] \
                                             #- blockvector_x[:, active_mask] * spdiags(eigenvalues[active_mask],0,current_block_size,current_block_size)
            #else:
                ## to make blockvector_r full when lambda is just a scalar
                #blockvector_r[:, active_mask] = blockvector_ax[:, active_mask] \
                                            #- blockvector_x[:, active_mask] * eigenvalues(active_mask)
        #else:
            #if current_block_size > 1:
                #blockvector_r[:, active_mask] = blockvector_ax[:, active_mask] \
                                             #- blockvector_bx[:, active_mask] * spdiags(eigenvalues(active_mask),0,current_block_size,current_block_size)
            #else:
                ## to make blockvector_r full when lambda is just a scalar
                #blockvector_r[:, active_mask] = blockvector_ax[:, active_mask] \
                                           #-  blockvector_bx[:, active_mask] * eigenvalues(active_mask)
        # ----------------------------------------------------------------------
        # compute all residuals
        if B is None:
            if block_size > 1:
                #print 'blockvector_x', iteration_number
                #print blockvector_x
                #print
                blockvector_r = blockvector_ax \
                              - blockvector_x * eigenvalues
                #print eigenvalues
            else:
                # to make blockvector_r full when lambda is just a scalar
                blockvector_r = blockvector_ax \
                              - blockvector_x * eigenvalues

        else:
            if block_size > 1:
                blockvector_r = blockvector_ax \
                              - blockvector_bx * eigenvalues
            else:
                # to make blockvector_r full when lambda is just a scalar
                blockvector_r = blockvector_ax \
                              - blockvector_bx * eigenvalues
        # ----------------------------------------------------------------------
        # Satisfying the constraints for the active residulas
        if constraints == 'symmetric':
            raise NotImplementedError()
        # ----------------------------------------------------------------------
        #print 'blockvector_r', iteration_number
        #print blockvector_r
        #print
        residual_norms = np.sqrt( sum( blockvector_r.conjugate() * blockvector_r ).T.conjugate() )
        assert( ( residual_norms.imag == 0 ).all() )
        residual_norms = residual_norms.real
        residual_norms_history[:block_size, iteration_number] = residual_norms

        #print 'residual_norms'
        #print residual_norms

        #print iteration_number, residual_norms

        # index antifreeze
        active_mask = (residual_norms > residual_tolerance) & active_mask

        current_block_size = sum( active_mask )
        if  current_block_size == 0:
            failureFlag = 0 # all eigenpairs converged
            print "converged1"
            break

        # Applying the preconditioner T to the active residulas
        if T is not None:
            blockvector_r[:, active_mask] = T * blockvector_r[:, active_mask]

        if constraints == 'conventional':
            if B is None:
                tmp2 = gram_y.solve( _block_vdot(blockvector_y, blockvector_r[:, active_mask]) )
                blockvector_r[:, active_mask] = blockvector_r[:, active_mask] \
                                              - blockvector_y * tmp2
            else:
                tmp2 = gram_y.solve( _block_vdot( blockvector_by, blockvector_r[:, active_mask]) )
                blockvector_r[:, active_mask] = blockvector_r[:, active_mask] \
                                              - blockvector_y * tmp2

        # Making active (preconditioned) residuals orthogonal to blockvector_x
        if B is None:
            # blockvector_r(:,activeMask) = blockvector_r(:,activeMask) - ...
            #                              blockVectorX*(blockVectorX'*blockvector_r(:,activeMask));
            blockvector_r[:, active_mask] -= np.dot( blockvector_x, _block_vdot( blockvector_x,  blockvector_r[:, active_mask] ) )
        else:
            blockvector_r[:, active_mask] -= np.dot( blockvector_x, _block_vdot( blockvector_bx, blockvector_r[:, active_mask] ) )

        #print '\nblockvector_r[:, active_mask]', iteration_number
        #print blockvector_r[:, active_mask]

        # Making active residuals orthonormal
        if B is None:
            # [blockvector_r[:, active_mask],gram_rbr]=...
            # qr(blockvector_r[:, active_mask],0) # to increase stability
            gram_rbr = _block_vdot( blockvector_r[:, active_mask],
                                    blockvector_r[:, active_mask]
                                  )

            assert np.allclose( gram_rbr, gram_rbr.T.conjugate() )

            try:
                gram_rbr = cholesky( gram_rbr )
            except np.linalg.LinAlgError:
                print blockvector_r
                print '\nThe residual is not full rank.\n'
                raise

            #print 'gram_rbr'
            #print gram_rbr

            #print 'blockvector_r'
            #print blockvector_r
            blockvector_r [ :, active_mask ] = solve( gram_rbr.T, blockvector_r [:, active_mask].T ).T
            #print 'blockvector_r'
            #print blockvector_r

        else:
            blockvector_br[:, active_mask] = B * blockvector_r[ :, active_mask ]

            gram_rbr = _block_vdot( blockvector_r [ :, active_mask ],
                                    blockvector_br[ :, active_mask ]
                                  )
            assert np.allclose( gram_rbr, gram_rbr.T.conjugate() )

            try:
                gram_rbr = cholesky( gram_rbr )
            except np.linalg.LinAlgError:
                print 'The residual is not full rank or/and operatorB is not positive definite.'
                raise

            blockvector_r [ :, active_mask ] = solve( gram_rbr.T, blockvector_r [:, active_mask].T ).T
            blockvector_br[ :, active_mask ] = solve( gram_rbr.T, blockvector_br[:, active_mask].T ).T

        blockVector_ar[:, active_mask] = A * blockvector_r[:, active_mask]

        if iteration_number > 0:
            # Making active conjugate directions orthonormal
            if B is None:
                # [blockvector_p[:, active_mask],gram_pbp] = qr(blockvector_p[:, active_mask],0)
                gram_pbp =  _block_vdot( blockvector_p[:, active_mask],
                                         blockvector_p[:, active_mask]
                                       )
                assert np.allclose( gram_pbp, gram_pbp.T.conjugate() )

                try:
                    gram_pbp = cholesky( gram_pbp )
                except np.linalg.LinAlgError:
                    print 'The direction matrix is not full rank.'
                    raise
                blockvector_p [:, active_mask] = solve( gram_pbp.T, blockvector_p [:, active_mask].T ).T
                blockvector_ap[:, active_mask] = solve( gram_pbp.T, blockvector_ap[:, active_mask].T ).T

            else:
                gram_pbp = _block_vdot( blockvector_p [:, active_mask],
                                        blockvector_bp[:, active_mask]
                                      )
                assert np.allclose( gram.pbp, gram_pbp.T.conjugate() )

                try:
                    gram_pbp = cholesky( gram_pbp )
                except np.linalg.LinAlgError:
                    print 'The direction matrix is not full rank or/and operatorB is not positive definite.'
                    raise

                blockvector_p [:, active_mask] = solve( gram_pbp.T, blockvector_p [:, active_mask].T ).T
                blockvector_ap[:, active_mask] = solve( gram_pbp.T, blockvector_ap[:, active_mask].T ).T
                blockvector_bp[:, active_mask] = solve( gram_pbp.T, blockvector_bp[:, active_mask].T ).T

            # clear gram_pbp

        condest_g_mean = mean( condest_g_history[ max( 0, iteration_number-9- round(log(current_block_size))):iteration_number+1] )

        # restart=1

        # The Raileight-Ritz method for [blockvector_x blockvector_r blockvector_p]
        if (residual_norms > eps**0.6).all():
            explicit_gram_flag = False
        else:
            explicit_gram_flag = True # suggested by Garrett Moran, private

        active_r_size = blockvector_r[:, active_mask].shape[1]
        if iteration_number == 0:
            active_p_size = 0
            restart = 1
        else:
            active_p_size = blockvector_p[:, active_mask].shape[1]
            restart = 0

        #print
        #print 'blockvector_ax'
        #print blockvector_ax
        #print
        #print 'blockvector_r[:, active_mask]'
        #print blockvector_r[:, active_mask]
        #print
        gram_xar = _block_vdot( blockvector_ax,
                                blockvector_r[:, active_mask]
                              )

        #print 'blockVector_ar[:, active_mask]', blockVector_ar[:, active_mask]
        gram_rar = _block_vdot( blockVector_ar[:, active_mask],
                                blockvector_r[:, active_mask]
                              )
        assert np.allclose( gram_rar, gram_rar.T.conjugate() )

        if explicit_gram_flag:
            gram_xax = _block_vdot( blockvector_ax, blockvector_x )
            assert np.allclose( gram_xax, gram_xax.T.conjugate() )
            if B is None:
                gram_xbx = _block_vdot( blockvector_x, blockvector_x )
                gram_rbr = _block_vdot( blockvector_r[:, active_mask],
                                        blockvector_r[:, active_mask]
                                      )
                gram_xbr = _block_vdot( blockvector_x,
                                        blockvector_r[:, active_mask]
                                      )
            else:
                gram_xbx = _block_vdot( blockvector_bx, blockvector_x )
                gram_rbr = _block_vdot( blockvector_br[:, active_mask],
                                        blockvector_r [:, active_mask]
                                      )
                gram_xbr = _block_vdot( blockvector_bx,
                                        blockvector_r[:, active_mask]
                                      )

            gram_xbx = 0.5 * ( gram_xbx + gram_xbx.T.conjugate() )
            gram_rbr = 0.5 * ( gram_rbr + gram_rbr.T.conjugate() )

        #print ' ############################### '
        for cond_try in xrange( 2 ): # cond_try == 2 when restart
            if not restart:
                gram_xap = _block_vdot( blockvector_ax,
                                        blockvector_p[:, active_mask]
                                      )
                gram_rap = _block_vdot( blockVector_ar[:, active_mask],
                                        blockvector_p [:, active_mask]
                                      )
                gram_pap = _block_vdot( blockvector_ap[:, active_mask],
                                        blockvector_p [:, active_mask]
                                      )
                gram_pap = 0.5 * ( gram_pap + gram_pap.T.conjugate() )

                if explicit_gram_flag:
                    #print '0a'
                    gram_a = np.bmat( [ [ gram_xax,               gram_xar,               gram_xap  ],
                                        [ gram_xar.T.conjugate(), gram_rar,               gram_rap ],
                                        [ gram_xap.T.conjugate(), gram_rap.T.conjugate(), gram_pap ]
                                      ]
                                    )
                else:
                    #print '0'
                    #print gram_xar.T.conjugate()
                    gram_a = np.bmat( [ [ np.diag(eigenvalues),     gram_xar,                gram_xap ],
                                        [ gram_xar.T.conjugate(),   gram_rar,                gram_rap ],
                                        [ gram_xap.T.conjugate(),   gram_rap.T.conjugate(),  gram_pap ]
                                      ]
                                    )

                # clear gram_xap gram_rap gram_pap
                if B is None:
                    #print 'blockvector_x', blockvector_x
                    #print 'blockvector_p[:, active_mask]', blockvector_p[:, active_mask]
                    gram_xbp = _block_vdot( blockvector_x,
                                            blockvector_p[:, active_mask]
                                          )
                    gram_rbp = _block_vdot( blockvector_r[:, active_mask],
                                            blockvector_p[:, active_mask]
                                          )
                else:
                    gram_xbp = _block_vdot( blockvector_bx,
                                            blockvector_p[:, active_mask]
                                          )
                    gram_rbp = _block_vdot( blockvector_br[:, active_mask],
                                            blockvector_p[:, active_mask]
                                          )
                    # or blockvector_r[:, active_mask]'*blockvector_bp[:, active_mask]

                if explicit_gram_flag:
                    if B is None:
                        gram_pbp = _block_vdot( blockvector_p[:, active_mask],
                                                blockvector_p[:, active_mask]
                                              )
                    else:
                        gram_pbp = _block_vdot( blockvector_bp[:, active_mask],
                                                blockvector_p[:, active_mask]
                                              )
                    gram_pbp = 0.5 * ( gram_pbp + gram_pbp.T.conjugate() )

                    gram_b = np.bmat( [ [ gram_xbx,               gram_xbr,               gram_xbp ],
                                        [ gram_xbr.T.conjugate(), gram_rbr,               gram_rbp ],
                                        [ gram_xbp.T.conjugate(), gram_rbp.T.conjugate(), gram_pbp ]
                                      ]
                                    )
                    # clear   gram_pbp
                else:
                    #print 'gram_xbp', gram_xbp
                    gram_b = np.bmat( [ [ np.eye(block_size),                      np.zeros((block_size, active_r_size)), gram_xbp ],
                                        [ np.zeros((block_size, active_r_size)).T, np.eye(active_r_size),                 gram_rbp ],
                                        [ gram_xbp.T.conjugate(),                  gram_rbp.T.conjugate(),                np.eye(active_p_size) ]
                                      ]
                                    )

                # clear gram_xbp  gram_rbp

            else:
                if explicit_gram_flag:
                    #print '1'
                    gram_a = np.bmat( [ [ gram_xax,               gram_xar ],
                                        [ gram_xar.T.conjugate(), gram_rar ]
                                      ]
                                    )
                    gram_b = np.bmat( [ [ gram_xbx,               gram_xbr ],
                                        [ gram_xbr.T.conjugate(), np.eye(active_r_size) ]
                                      ]
                                    )
                    # clear gram_xax gram_xbx gram_xbr
                else:
                    #print '2'
                    #print 'gram_xar'
                    #print gram_xar
                    gram_a = np.bmat( [ [ np.diag(eigenvalues),   gram_xar ],
                                        [ gram_xar.T.conjugate(), gram_rar ]
                                      ]
                                    )
                    gram_b = np.eye( block_size + active_r_size )

                # clear gram_xar gram_rar

            #print 'gram_b'
            #print gram_b
            cond_est_g = log10( np.linalg.cond(gram_b) ) + 1
            if ( cond_est_g/condest_g_mean > 2 and cond_est_g > 2 ) or cond_est_g > 8:
                # black magic - need to guess the restart
                if verbosity:
                    print  'Restart on step %i as cond_est_g %5.4e \n' % \
                           (iteration_number, cond_est_g)
                if cond_try == 1 and not restart:
                    restart = 1 # steepest descent restart for stability
                else:
                    Warning( 'Gram matrix ill-conditioned: results unpredictable.' )
            else:
                print "converged2"
                break

        #print
        #print 'gram_a\n', gram_a
        #print 'gram_b\n', gram_b

        # gram_a and gram_b are both symmetric and real, so there should be
        # some way to exploit this. eigh() is not suitable, thought, as gram_b
        # may very well be indefinite. Hence, use eig() for now.
        eigenvalues, eigenvectors = _symeig( gram_a, gram_b )
        assert ( eigenvalues.imag == 0.0 ).all()
        eigenvalues = eigenvalues.real
        # Unfortunately, eig() forgets to normalize with generalized
        # eigenvalue problems, see
        # http://projects.scipy.org/scipy/ticket/1308.
        # Help out.
        #eigenvectors /= np.sqrt( sum( eigenvectors.conjugate() * eigenvectors ) )

        ii = np.argsort( eigenvalues )[:block_size]
        if True: #largest
            ii = ii[::-1]
        if verbosity > 10:
            print ii

        eigenvalues = eigenvalues[ii]
        eigenvectors = eigenvectors[:,ii]


        #print 'eigenV'
        #print eigenvalues
        #print eigenvectors
        #print 'eigenVEND'
        #eigenvalues  = eigenvalues [:block_size]
        #eigenvectors = eigenvectors[:, :block_size]
        #print 'eigenvalues', eigenvalues
        #print 'eigenvectors', eigenvectors
        # clear gram_a gram_b

        #if issparse( blockvector_x ):
            #eigenvectors = sparse( eigenvectors )

        if not restart:
            blockvector_p = np.dot( blockvector_r[:, active_mask], eigenvectors[block_size            :block_size+active_r_size              , :] ) \
                         + np.dot( blockvector_p[:, active_mask], eigenvectors[block_size+active_r_size:block_size + active_r_size+active_p_size, :] )
            blockvector_ap = np.dot( blockVector_ar[:, active_mask], eigenvectors[block_size:block_size+active_r_size, :] )\
                          + np.dot( blockvector_ap[:, active_mask], eigenvectors[block_size+active_r_size:block_size + active_r_size+active_p_size , :] )
            if not B is None:
                blockvector_bp = np.dot( blockvector_br[:, active_mask], eigenvectors[block_size:block_size+active_r_size, :] ) \
                              + np.dot( blockvector_bp[:, active_mask], eigenvectors[block_size+active_r_size:block_size+active_r_size+active_p_size, :] )
        else: # use block steepest descent
            blockvector_p =  np.dot( blockvector_r [:, active_mask], eigenvectors[block_size:block_size+active_r_size, :] )
            blockvector_ap = np.dot( blockVector_ar[:, active_mask], eigenvectors[block_size:block_size+active_r_size, :] )
            if not B is None:
                blockvector_bp = np.dot( blockvector_br[:, active_mask],
                                        eigenvectors[block_size:block_size+active_r_size, :]
                                      )

        #print 'updating blockvector_x'
        #print blockvector_p.real
        #print
        #print 'eigenvectors[:block_size, :]'
        #print eigenvectors[:block_size, :]
        #print
        blockvector_x = np.dot( blockvector_x, eigenvectors[:block_size, :] ) \
                      + blockvector_p
        #print blockvector_x.real

        #print 'blockvector_ax', blockvector_ax
        #print 'eigenvectors[:block_size, :]', eigenvectors[:block_size, :]
        #print 'blockvector_ap', blockvector_ap
        blockvector_ax = np.dot( blockvector_ax, eigenvectors[:block_size, :] ) \
                       + blockvector_ap
        if not B is None:
            blockvector_bx = np.dot( blockvector_bx, eigenvectors[:block_size, :] ) \
                          + blockvector_bp

        assert( (eigenvalues.imag == 0).all() )
        eigenvalues = eigenvalues.real
        lambda_history[:block_size, iteration_number+1] = eigenvalues
        condest_g_history[iteration_number+1] = cond_est_g

        if verbosity > 0:
            print 'Iteration %i current block size %i \n' % (iteration_number, current_block_size)
            print 'Eigenvalues', eigenvalues
            print 'Residual norms ', residual_norms
    # The main step of the method was the CG cycle: end

    # Postprocessing

    # Making sure blockvector_x's "exactly" satisfy the blockvector_y constrains??

    # Making sure blockvector_x's are "exactly" othonormalized by final "exact" RR
    if B is None:
        gram_xbx = _block_vdot( blockvector_x, blockvector_x )
    else:
        blockvector_bx = B * blockvector_x
        gram_xbx = _block_vdot( blockvector_x, blockvector_bx )
    gram_xbx = 0.5 * ( gram_xbx.T.conjugate() + gram_xbx )


    blockvector_ax = A * blockvector_x
    gram_xax = _block_vdot( blockvector_x, blockvector_ax )
    gram_xax = 0.5 * ( gram_xax.T.conjugate() + gram_xax )

    # Raileigh-Ritz for blockvector_x, which is already operatorB-orthonormal
    eigenvalues, eigenvectors = _symeig( gram_xax, gram_xbx )
    assert ( eigenvalues.imag == 0 ).all()
    eigenvalues = eigenvalues.real

    #if issparse(blockvector_x):
        #eigenvectors = sparse(eigenvectors)

    blockvector_x  = np.dot( blockvector_x,  eigenvectors )
    blockvector_ax = np.dot( blockvector_ax, eigenvectors )
    if B is not None:
        blockvector_bx = np.dot( blockvector_bx, eigenvectors )


    # Computing all residuals
    if B is None:
        if block_size > 1:
            blockvector_r = blockvector_ax \
                         - blockvector_x * eigenvalues
        else:
            # to make blockvector_r full when lambda is just a scalar
            blockvector_r = blockvector_ax \
                         - blockvector_x * eigenvalues
    else:
        if block_size > 1:
            blockvector_r = blockvector_ax \
                          - blockvector_bx * eigenvalues
        else:
            # to make blockvector_r full when lambda is just a scalar
            blockvector_r = blockvector_ax \
                          - blockvector_bx * eigenvalues

    residual_norms = np.sqrt(sum( blockvector_r.conjugate() *blockvector_r ).T.conjugate())
    assert( ( residual_norms.imag == 0 ).all() )
    residual_norms = residual_norms.real
    residual_norms_history[:block_size, iteration_number] = residual_norms

    if verbosity > 0:
        print 'Final eigenvalues ', eigenvalues
        print 'Final residual norms ', residual_norms


    return eigenvalues, blockvector_x
    #return failureFlag, \
           #lambda_history[:block_size, :iteration_number], \
           #residual_norms_history[:block_size, :iteration_number-1]


    #current_block_size = sum(active_mask)
    #if  current_block_size == 0
        #failureFlag=0 %all eigenpairs converged
        #break
    #end
# ==============================================================================
def _block_vdot( blockvector_x, blockvector_y ):
    '''
    Block version of the inner product.
    '''
    a = np.dot( blockvector_x.T.conjugate(), blockvector_y )
    return a
    #k = blockvector_x.shape[1]
    #A = np.zeros( (k, k) )
    #for k1 in xrange(k):
        #for k2 in xrange(k):
            #A[k1, k2] = np.vdot( blockvector_x[:, k1], blockvector_y[:, k2] )
    #return A
# ==============================================================================
def _symeig( mtxA, mtxB = None, eigenvectors = True, select = None ):
    '''
    Solves a symmetric eigenvalue problem as similarly as MATLAB(R) does.
    '''
    import scipy.linalg as sla
    import scipy.lib.lapack as ll
    if select is None:
        if np.iscomplexobj( mtxA ):
            if mtxB is None:
                fun = ll.get_lapack_funcs( ['heev'], arrays = (mtxA,) )[0]
            else:
                fun = ll.get_lapack_funcs( ['hegv'], arrays = (mtxA,) )[0]
        else:
            if mtxB is None:
                fun = ll.get_lapack_funcs( ['syev'], arrays = (mtxA,) )[0]
            else:
                fun = ll.get_lapack_funcs( ['sygv'], arrays = (mtxA,) )[0]
##         print fun
        if mtxB is None:
            out = fun( mtxA )
        else:
            out = fun( mtxA, mtxB )
##         print w
##         print v
##         print info
##         from _symeig import _symeig
##         print _symeig( mtxA, mtxB )
    else:
        out = sla.eig( mtxA, mtxB, right = eigenvectors )
        w = out[0]
        ii = np.argsort( w )
        w = w[slice( *select )]
        if eigenvectors:
            v = out[1][:,ii]
            v = v[:,slice( *select )]
            out = w, v, 0
        else:
            out = w, 0

    return out[:-1]
# ==============================================================================