import numpy as np
import numexpr as ne
from scipy.linalg.blas import sgemm
from sklearn.preprocessing import KernelCenterer


class KernelsHelper(object):

    @staticmethod
    def gram_matrix(X, y=None, kernel="linear", bandwidth=1, centered=False):
        k = None

        if kernel == "linear":
            if y is None:
                k = sgemm(alpha=1.0, a=X, b=X, trans_b=True)
            else:
                k = sgemm(alpha=1.0, a=X, b=y, trans_b=True)

        elif kernel == "rbf":
            # print("rbf kernel: ")
            euc_dist = np.einsum('ij,ij->i', X, X)
            if y is not None:
                euc_dist_y = np.einsum('ij,ij->i', y, y)
            else:
                euc_dist_y = euc_dist
                y = X

            k = ne.evaluate('exp(-b * (A + B - 2 * C))', {
                'A': euc_dist[:, None],
                'B': euc_dist_y[None, :],
                'C': sgemm(alpha=1.0, a=X, b=y, trans_b=True),
                'b': bandwidth,
            })

        # elif kernel == "rbf":
        #     print("rbf kernel: ")
        #     euc_dist = np.einsum('ij,ij->i', X, X)
        #     k = ne.evaluate('exp(-b * (A + B - 2 * C))', {
        #         'A': euc_dist[:, None],
        #         'B': euc_dist[None, :],
        #         'C': sgemm(alpha=1.0, a=X, b=X, trans_b=True),
        #         'b': bandwidth,
        #     })

        if centered and k.shape[0] == k.shape[1]:
            N = X.shape[0]
            identity_n = np.ones((N, N)) / N
            first_term = k
            second_term = np.dot(identity_n, k)
            third_term = np.dot(k, identity_n)
            fourth_term = np.dot(identity_n, third_term)
            k = first_term - second_term - third_term + fourth_term

        return k
