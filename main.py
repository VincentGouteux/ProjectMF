import numpy as np


def gradientDescent(R, U, V, alpha, nFeatures):
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] > 0:
                r_hat_ij = np.dot(U[i, :], V.T[:, j])
                eij = R[i, j] - r_hat_ij
                for q in range(nFeatures):
                    U[i, q] = U[i, q] + alpha * eij * V[j, q]
                    V[j, q] = V[j, q] + alpha * eij * U[i, q]
    return U, V


def computeTotalError(R, U, V):
    error = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] > 0:
                error += R[i, j] - np.dot(U[i, :], V.T[:, j])
    return error


def computeApproximatedMatrix(U, V):
    return np.dot(U, V.T)
