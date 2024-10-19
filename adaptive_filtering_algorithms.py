import numpy as np

from scipy import linalg

def convmtx(h, n):
    '''
    Convolution matrix, same as convmtx does in matlab
    '''
    return linalg.toeplitz(
        np.hstack([h, np.zeros(n-1)]),
        np.hstack([h[0], np.zeros(n-1)]),
    )

def lms(x, y, L, mu):
    N = y.shape[0]
    w = np.zeros(L,)
    yhat = np.zeros(N,)
    e = np.zeros(N,)

    # zero-pad input signal
    x = np.concatenate((np.zeros(L-1,), x), axis=0)

    for n in range(0, N):
        x_n = x[n:n+L]
        yhat[n] = w.T @ x_n
        e[n] = y[n] - yhat[n]
        w = w + mu*e[n]*x_n
        
    return yhat, e

def nlms(x, y, L, mu, delta):
    N = y.shape[0]
    w = np.zeros(L,)
    yhat = np.zeros(N,)
    e = np.zeros(N,)

    if x.ndim == 1:
        X = convmtx(x, L).T
    else:
        X = x

    for n in range(0, N):
        x_n = X[:,n]
        yhat[n] = w.T @ x_n
        e[n] = y[n] - yhat[n]
        w = w + (mu / (delta+x_n.T@x_n)) * x_n * e[n]
        
    return yhat, e

def rls(x: np.ndarray, y: np.ndarray, L: int, beta: float, lambda_: float):
    '''
    Input
        x: input signal
        y: desired signal
        L: filter length
        beta: forget factor
        lambda_: regularization

    Output
        yhat: filter output
    '''
    yhat = np.zeros(len(y))
    e = np.zeros(len(y))

    if x.ndim == 1:
        X = np.fliplr(convmtx(x,L)).T
    else:
        X = x

    # start RLS
    # initialize
    theta = np.zeros((L, 1))  # theta in the book
    P = 1/lambda_*np.eye(L)

    for n in range(len(y)):
        x_n = X[:,n, None]

        # get filter output
        yhat[n] = theta.T@x_n

        # update iteration
        e[n] = y[n] - yhat[n]
        z = P @ x_n
        denominator = beta + x_n.T @ z
        K_n = z/denominator
        theta = theta + K_n*e[n]
        P = (P - K_n @ z.T)/beta

    return yhat, e