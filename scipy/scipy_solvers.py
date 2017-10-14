import argparse
import time

import numpy as np
from scipy.io import mmread
import scipy.sparse.linalg as LA
import scipy.sparse as sp

def read_vector(flname):
    with open(flname) as fl:
        length = int(next(fl))
        vec = []
        for ln in fl:
            vec.append(float(ln))
    return np.array(vec)

def custom_cg(A, b, M, max_iter, tol):
    x = np.random.randn(*b.shape)
    r_0 = b - A.dot(b)
    z_0 = M.matvec(r_0)
    p = z_0
    k = 0
    for i in xrange(max_iter):
        print "Iteration", (i+1)
        A_p = A.dot(p)
        alpha = r_0.dot(z_0) / p.dot(A_p)
        x += alpha * p
        r_1 = r_0 - alpha * A_p

        print "Residual L2 norm", np.linalg.norm(r_1)

        if np.linalg.norm(r_1) < tol:
            return x, 0
        
        z_1 = M.matvec(r_1)
        beta = z_1.dot(r_1) / z_0.dot(r_0)
        p = z_1 + beta * p

        r_0 = r_1
        z_0 = z_1

    return x, 0
        

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--matrix-fl",
                        required=True,
                        type=str)

    parser.add_argument("--input-vec",
                        required=True,
                        type=str)

    parser.add_argument("--preconditioner",
                        type=str,
                        required=True,
                        choices=["diagonal",
                                 "ilu"])

    parser.add_argument("--solver",
                        type=str,
                        required=True,
                        choices=["cg",
                                 "gmres",
                                 "custom-cg"])

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    A = mmread(args.matrix_fl).tocsr()
    b = read_vector(args.input_vec)

    before = time.clock()
    
    if args.preconditioner == "diagonal":
        D = sp.diags(A.diagonal(), offsets=0).tocsr()
        M_x = lambda x: LA.spsolve(D, x)
        M = LA.LinearOperator(A.shape, M_x)
    elif args.preconditioner == "ilu":
        M2 = LA.spilu(A.tocsc(),
                      drop_tol=1e-8,
                      fill_factor=100)
        M_x = lambda x: M2.solve(x)
        M = LA.LinearOperator(A.shape, M_x)
    else:
        raise Exception, "Unknown preconditioner %s" % args.preconditioner

    if args.solver == "gmres":
        x, result = LA.gmres(A, b, M=M, tol=1e-6, maxiter=5000, restart=50)
    elif args.solver == "cg":
        x, result = LA.cg(A, b, tol=1e-6, maxiter=5000, M=M)
    elif args.solver == "custom-cg":
        x, result = custom_cg(A, b, M, 5000, 1e-6)
    else:
        raise Exception, "Uknown solver %s" % args.solver
    
    after = time.clock()

    elapsed = after - before
    print "Took %s seconds to solve" % elapsed
    print result
