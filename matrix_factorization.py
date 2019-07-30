import numpy as np

__author__ = 'winkemoji'


class matrix_factorization:
    def __init__(self, R, iter, learning_rate, regularization_param, hidden_factor):
        '''
        :param R: Matrix need to factorization.
        :param iter: Number of iteration.
        :param learning_rate: Controlling the learning progress of the model
        :param regularization_param: Preventing overfitting
        :param hidden_factor: Recommend 100
        '''
        self.R = R
        self.n_f = hidden_factor
        self.lr = learning_rate
        self.rp = regularization_param
        self.P = np.random.rand(self.n_f, R.shape[0]).T
        self.Q = np.random.rand(self.n_f, R.shape[1])
        self.iter = iter
        pass

    def fit(self, verbose=False):
        '''
        :param verbose: print details like R_hat and Error
        '''
        rs = {}
        for it in range(self.iter):
            for i in range(len(self.R)):
                for j in range(len(self.R[i])):
                     if self.R[i][j] > 0:
                        eij = self.R[i][j] - np.dot(self.P[i, :], self.Q[:, j])
                        for k in range(self.n_f):
                            self.P[i][k] = self.P[i][k] + self.lr * (2 * eij * self.Q[k][j] - self.rp * self.P[i][k])
                            self.Q[k][j] = self.Q[k][j] + self.rp * (2 * eij * self.P[i][k] - self.rp * self.Q[k][j])
            E = 0
            for i in range(len(self.R)):
                for j in range(len(self.R[i])):
                     if self.R[i][j] > 0:
                        E = E + pow(self.R[i][j] - np.dot(self.P[i, :], self.Q[:, j]), 2)
                        for k in range(self.n_f):
                            E = E + (self.rp / 2) * (pow(self.P[i][k], 2) + pow(self.Q[k][j], 2))
            if verbose and it % 100 == 0 is not False:
                print('**', it, '**')
                print(np.dot(self.P, self.Q))
                print('Error', E)
            rs.update(E=E)
            if E < 0.05:
                break
        rs.update(P=self.P)
        rs.update(Q=self.Q)
        return rs


R = np.array([[0, 0, 1], [2, 5 , 2]])

mf = matrix_factorization(R=R, iter=5000, learning_rate=.002, regularization_param=0.02, hidden_factor=50)
result = mf.fit(verbose=False)
print(result['P'])
print(result['Q'])