import logging
import numpy as np
import scipy.linalg
import scipy.sparse


def l1decode_pd(x0,
                A,
                At,
                y,
                pdtol=1e-3,
                pdmaxiter=50,
                cgtol=1e-8,
                cgmaxiter=200):
    '''
    Decoding via linear programming.
    Solve
    min_x  ||b-Ax||_1 .

    Recast as the linear program
    min_{x,u} sum(u)  s.t.  -Ax - u + y <= 0
                             Ax - u - y <= 0
    and solve using primal-dual interior point method.

    Usage: xp = l1decode_pd(x0, A, At, y, pdtol, pdmaxiter, cgtol, cgmaxiter)

    x0 - Nx1 vector, initial point.

    A - Either a handle to a function that takes a N vector and returns a M
        vector, or a MxN matrix.  If A is a function handle, the algorithm
        operates in "largescale" mode, solving the Newton systems via the
        Conjugate Gradients algorithm.

    At - Handle to a function that takes an M vector and returns an N vector.
         If A is a matrix, At is ignored.

    y - Mx1 observed code (M > N).

    pdtol - Tolerance for primal-dual algorithm (algorithm terminates if
        the duality gap is less than pdtol).
        Default = 1e-3.

    pdmaxiter - Maximum number of primal-dual iterations.
        Default = 50.

    cgtol - Tolerance for Conjugate Gradients; ignored if A is a matrix.
        Default = 1e-8.

    cgmaxiter - Maximum number of iterations for Conjugate Gradients; ignored
        if A is a matrix.
        Default = 200.

    Matlab version written by: Justin Romberg, Caltech
    Email: jrom@acm.caltech.edu
    Created: October 2005

    Incomplete export to python by Lixin Xue, April 2021.
    '''
    if type(A) == np.ndarray:
        largescale = False
    else:
        assert hasattr(A, '__call__'), \
            f"A must be a np array or a function handle, not {type(A)}"
        largescale = True

    N = x0.shape[0]
    M = y.shape[0]

    alpha = 0.01
    beta = 0.5
    mu = 10

    gradf0 = np.concatenate([np.zeros((N, 1)), np.ones((M, 1))], axis=0)

    x = x0
    if largescale:
        Ax = A(x)
    else:
        Ax = A @ x
    u = 0.95 * np.abs(y - Ax) + 0.10 * np.max(np.abs(y - Ax))

    fu1 = Ax - y - u
    fu2 = -Ax + y - u

    lamu1 = -1 / fu1
    lamu2 = -1 / fu2

    if largescale:
        Atv = At(lamu1 - lamu2)
    else:
        Atv = A.T @ (lamu1 - lamu2)

    sdg = -(fu1.T @ lamu1 + fu2.T @ lamu2).item()
    tau = mu * 2 * M / sdg

    rcent = np.concatenate([-lamu1 * fu1, -lamu2 * fu2], axis=0) - 1 / tau
    rdual = gradf0 + np.concatenate([Atv, -lamu1 - lamu2], axis=0)
    resnorm = np.linalg.norm(np.concatenate([rcent, rdual], axis=0))

    pditer = 0
    done = sdg < pdtol or pditer >= pdmaxiter
    while not done:
        pditer += 1

        w2 = -1 - 1 / tau * (1 / fu1 + 1 / fu2)

        sig1 = -lamu1 / fu1 - lamu2 / fu2
        sig2 = lamu1 / fu1 - lamu2 / fu2
        sigx = sig1 - sig2**2 / sig1

        if largescale:
            w1 = -1 / tau * At(-1 / fu1 + 1 / fu2)
            w1p = w1 - At((sig2 / sig1) * w2)
            # TODO: add conjugate gradient solver here for large scale problem
            assert False, "cgsolve not impmented yet"
        else:
            w1 = -1 / tau * (A.T @ (-1 / fu1 + 1 / fu2))
            w1p = w1 - A.T @ ((sig2 / sig1) * w2)
            H11p = A.T @ (scipy.sparse.diags(sigx[:, 0], offsets=0) @ A)

            try:
                dx = scipy.linalg.solve(H11p, w1p, assume_a='pos')
            except np.linalg.LinAlgError as err:
                logging.error(f"numpy.linalg.LinAlgError: {err}")
                return None
            Adx = A @ dx

        du = (w2 - sig2 * Adx) / sig1

        dlamu1 = -(lamu1 / fu1) * (Adx - du) - lamu1 - (1 / tau) * (1 / fu1)
        dlamu2 = (lamu2 / fu2) * (Adx + du) - lamu2 - (1 / tau) * (1 / fu2)

        if largescale:
            Atdv = At(dlamu1 - dlamu2)
        else:
            Atdv = A.T @ (dlamu1 - dlamu2)

        # make sure that the step is feasible:
        # keeps lamu1,lamu2 > 0, fu1,fu2 < 0
        indl = dlamu1 < 0
        indu = dlamu2 < 0
        s = min([
            1,
            np.amin(-lamu1[indl] / dlamu1[indl]),
            np.amin(-lamu2[indu] / dlamu2[indu])
        ])
        indl = (Adx - du) > 0
        indu = (-Adx - du) > 0
        if np.any(indl):
            indl_min = np.amin(-fu1[indl] / (Adx[indl] - du[indl]))
        else:
            indl_min = s + 1

        if np.any(indu):
            indu_min = np.amin(-fu2[indu] / (-Adx[indu] - du[indu]))
        else:
            indu_min = s + 1

        s = min([s, indl_min, indu_min])
        s = 0.99 * s

        suffdec = 0
        backiter = 0
        while not suffdec:
            xp = x + s * dx
            up = u + s * du
            Axp = Ax + s * Adx
            Atvp = Atv + s * Atdv
            lamu1p = lamu1 + s * dlamu1
            lamu2p = lamu2 + s * dlamu2
            fu1p = Axp - y - up
            fu2p = -Axp + y - up
            rdp = gradf0 + np.concatenate([Atvp, -lamu1p - lamu2p], axis=0)
            rcp = np.concatenate([-lamu1p * fu1p, -lamu2p * fu2p],
                                 axis=0) - (1 / tau)
            suffdec = np.linalg.norm(np.concatenate(
                [rdp, rcp], axis=0)) <= (1 - alpha * s) * resnorm
            s *= beta
            backiter += 1
            if backiter > 32:
                print("Stuck backtracking, returning last iterate.  "
                      "(See Section 4 of notes for more information.)")
                xp = x
                return xp

        # next iteration
        x = xp
        u = up
        Ax = Axp
        Atv = Atvp
        lamu1 = lamu1p
        lamu2 = lamu2p
        fu1 = fu1p
        fu2 = fu2p

        # surrogate duality gap
        sdg = -(fu1.T @ lamu1 + fu2.T @ lamu2).item()
        tau = mu * 2 * M / sdg
        rcent = np.concatenate([-lamu1 * fu1, -lamu2 * fu2],
                               axis=0) - (1 / tau)
        rdual = rdp
        resnorm = np.linalg.norm(np.concatenate([rdual, rcent], axis=0))

        done = (sdg < pdtol) or (pditer >= pdmaxiter)

        # print(f'Iteration = {pditer}, tau = {tau:8.3e}, '
        #       f'Primal = {np.sum(u):8.3e}, PDgap = {sdg:8.3e}, '
        #       f'Dual res = {np.linalg.norm(rdual):8.3e}')
        if largescale:
            # TODO: add print info for CG here
            print('')
    return xp
