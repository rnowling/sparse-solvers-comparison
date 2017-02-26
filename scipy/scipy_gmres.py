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

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--matrix-fl",
                        required=True,
                        type=str)

    parser.add_argument("--input-vec",
                        required=True,
                        type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    A = mmread(args.matrix_fl).tocsr()
    b = read_vector(args.input_vec)

    before = time.clock()
    D = sp.diags(A.diagonal(), offsets=0).tocsr()
    M_x = lambda x: LA.spsolve(D, x)
    M = LA.LinearOperator(A.shape, M_x)
    x, result = LA.gmres(A, b, M=M, tol=1e-6, maxiter=5000, restart=50)
    after = time.clock()

    elapsed = after - before
    print "Took %s seconds to solve" % elapsed
    print result
