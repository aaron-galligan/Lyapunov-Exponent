import numpy as np
from custom_functions import *





def LyapunovQR(M, xini, N):
    
    def df(x_arr):
        h = 1e-6

        J = MyJacobian(M,x_arr,h)
        J = np.squeeze(J, axis = 2)
        return J
    
    
    
    n = xini.size #n is the dim we are working in, i.e. M: R^n --> R^n
    x = np.zeros((N, n))
    x[0] = xini

    A = np.zeros((N, n, n))
    A[0] = df(x[0]) 


    Q = np.zeros((N, n, n))
    Q[0] = np.identity(n)
    r = np.zeros((N-1, n))
    #r[0] = np.diag(Q[0])

    i = 0
    while i < N-1:
        x[i+1] = M(x[i])

        A[i+1] = df(x[i+1]) 

        Q[i+1], R = np.linalg.qr(np.matmul(A[i], Q[i]))
        '''
        print('')
        print('Q[i], R ')
        print(Q[i])
        print(R)
        '''
        r[i] = np.diag(R)
        
        
        for j in range(n):
            if(r[i, j] < 0):
                r[i, j] = -r[i, j]
                Q[i+1, :, j] = -Q[i+1, :, j]
        i+=1

    lyapunov_exp = (1/N)*sum(np.log(r))
    return lyapunov_exp