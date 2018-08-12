import numpy as np
from scipy.sparse import coo_matrix


class CGL_rank(object):
    '''
    Args:
        `data`: Data object with 3 member variables. See docstrings of `Data` class

    Reference:
        Section 2, 
    '''

    def __init__(self):
        '''
        The model has only 1 traninable parameter F, and non-trainable parameter X
        and data T. While $F = X*A*X^T$ (Eq.8) and $K = X * X^T$, the whole model consists
        of the following parameters and stored data:

        Data:
            `X`: Bag-of-concepts representations of all courses, [num_courses, num_dimensions]
            `chunks`: Size 3 dict for 3-fold data, consisting of raw course pairs and labels

        Trainable parameters:
            `F`:
            `F_ij`: Derived from `F`
            `F_ik`: Derived from `F`

        Non-traninable parameters:
            `T`: Course Triplets
            `K`: 
            `K_inv`:
            `F`
        '''
        self.T_train = None

        self.C = None
        self.F = None
        self.F_ij = None
        self.F_ik = None

        self.Gradient = None
        self.loss = np.inf

    def load_data(self, data):
        self.opt = data.opt

        self.X = data.X
        self.num_courses = data.X.shape[0]
        self.chunks = data.chunks
        self.K = np.dot(data.X, data.X.transpose()) + \
            1e-8 * np.identity(self.num_courses)
        self.K_inv = np.linalg.inv(self.K)

        self.T = data.T

    def train(self):
        '''
        Algorithm 1 CGL.Rank with Nestrerov's Acceelerated Gradient Descent

        Reference:
            Algorithm 1 in Section 3.2.2
        '''
        for k in range(1, 4):
            print("\nFold {}".format(k))
            k_train = k - 1
            k_val = k % 3
            k_test = (k + 1) % 3

            for C in self.opt.C_pool:
                print("\nC = {}".format(C))
                # Prepare train data
                self.T_train = self.T[k_train]
                # Python uses 0-based indexing so we minus 1
                II = np.concatenate(
                    (self.T_train[:, 0], self.T_train[:, 0])) - 1
                JK = np.concatenate(
                    (self.T_train[:, 1], self.T_train[:, 2])) - 1

                # Initialize parameters
                self.C = C
                self.F = np.zeros((self.num_courses, self.num_courses))
                num_pairs = self.T_train.shape[0]

                for r in range(self.opt.maxIter):
                    # Use deltas to compute gradients
                    # Unlike using sub2ind in the original Matlab codes, we index arrays
                    # directly from coordinates
                    self.F_ij = self.F[list(self.T_train[:, 0] - 1),
                                       list(self.T_train[:, 1] - 1)]
                    self.F_ik = self.F[list(self.T_train[:, 0] - 1),
                                       list(self.T_train[:, 2] - 1)]
                    assert self.F_ij.shape == self.F_ik.shape == (num_pairs,)
                    # Line delta_ijk
                    deltas = np.maximum(np.zeros(num_pairs),
                                        np.ones(num_pairs) - self.F_ij + self.F_ik)
                    # We use scipy.sparse.coo_matrix as the sparse function in matlab
                    self.Gradient = self.C * coo_matrix((np.concatenate((deltas, -deltas)),
                                                         (II, JK)), shape=(self.num_courses, self.num_courses)).toarray()

                    # Compute the Newton direction N using preconditioned conjugate gradient
                    N, res, iteration = self._matrix_free_pcg(10, 1e-6)

                    # Simple backtracking search to determine the step size s
                    if self.opt.backTracking:
                        s = 1
                        F_0 = self.F
                        for t in range(10):
                            F_t = F_0 + s * (N - F_0)
                            loss = self._Loss(F_t, num_pairs)

                            if loss > self.loss:
                                s = s * 0.95
                            else:
                                self.loss = loss
                                self.F = F_t
                    else:
                        self.F = self.F - N

                    #print("Iteration:{} gradient:{} loss:{}".format(
                    #    r, np.sqrt(np.sum(self.Gradient ** 2)), self.loss))

                    map_eval = self.evaluate_map(self.chunks[k_val])
                    print("Validation result:{}".format(map_eval))
                    auc_val = self.evaluate_auc(self.F, self.T[k_val])
                    print("AUC result:{}".format(auc_val))

    def _Regularization_Map(self, F):
        return self.K_inv.dot(F).dot(self.K_inv)

    def _Precondition_Map(self, F):
        return self.K.dot(F).dot(self.K)

    def _Hessian_Map(self, X):
        # Compute the Hessian (as a linear mapping from its row space to column space)
        return self.C * self._lambda_mumltiply(X, self.T_train, self.F) + self._Regularization_Map(X)

    def _Loss(self, F, num_pairs):
        # Unlike the original matlab codes, we avoid using lambda expressions in building
        # the objective function to improve readability.
        hinge = np.maximum(np.zeros(num_pairs), np.ones(
            num_pairs) - self.F_ij + self.F_ik)
        empirical_loss = np.sum(np.square(hinge))

        regularization_loss = np.sum(self._Regularization_Map(F) * F)

        loss = self.C * empirical_loss + regularization_loss
        return loss

    def _matrix_free_pcg(self, maxIter, tol):
        x = self.F
        b = self.Gradient

        r = b - self._Hessian_Map(x)

        res0 = np.sqrt(np.sum(r**2))

        z = self._Precondition_Map(r)
        p = z

        rz_old = np.sum(r * z)
        for i in range(maxIter):
            res = np.sqrt(np.sum(r**2)) / res0
            if res < tol:
                break

            Ap = self._Hessian_Map(p)
            alpha = rz_old / np.sum(p * Ap)

            x = x + alpha * p
            r = r - alpha * Ap

            z = self._Precondition_Map(r)
            rz_new = np.sum(z * r)

            beta = rz_new / rz_old

            p = z + beta * p
            rz_old = rz_new

        return x, res, i

    def _lambda_mumltiply(self, A, T, F):
        V = np.zeros(F.shape)
        for t in range(T.shape[0]):
            i = T[t, 0] - 1
            j = T[t, 1] - 1
            k = T[t, 2] - 1
            if F[i, j] - F[i, k] < 1:
                delta = A[i, j] - A[i, k]
                V[i, j] += delta
                V[i, k] -= delta

        return V

    def evaluate_map(self, val_chunk):
        score = np.zeros(val_chunk['num_pairs'])

        for i in range(val_chunk['num_pairs']):
            u = val_chunk['pairs'][i, 0] - 1
            v = val_chunk['pairs'][i, 1] - 1
            score[i] = self.F[u, v]

        st_score = np.concatenate((np.expand_dims(
            score, axis=1), val_chunk['pairs'], np.expand_dims(val_chunk['labels'], axis=1)), axis=1)

        Q = 0
        sap = 0

        for i in set(list(val_chunk['pairs'][:, 0])):
            rl = st_score[st_score[:, 1] == i + 1, :]
            n_p = 0
            s_p = 0
            for k in range(rl.shape[0]):
                if rl[k, 3] == 1:
                    n_p += 1
                    s_p += n_p / (k + 1)

            if n_p > 0:
                Q += 1
                a_p = s_p / n_p
                sap += a_p

        return sap / Q

    def evaluate_ndcg(self, val_chunk, T):
        score = np.zeros(val_chunk['num_pairs'])

        for i in range(val_chunk['num_pairs']):
            u = val_chunk['pairs'][i, 0] - 1
            v = val_chunk['pairs'][i, 1] - 1
            score[i] = self.F[u, v]

        st_score = np.concatenate((np.expand_dims(
            score, axis=1), val_chunk['pairs'], np.maximum(np.expand_dims(val_chunk['labels'], np.zeros(val_chunk['labels'].shape)), axis=1)), axis=1)

        ndcg = np.zeros(self.opt.topK)
        nq = np.zeros(self.opt.topK)

        for i in set(list(val_chunk['pairs'][:, 0])):
            rl = st_score[st_score[:, 1] == i + 1, :]
            dcg = np.zeros(self.opt.topK)

            for k in range(self.opt.topK):
                dcg[k:self.opt.topK] += rl[k, 3] / (np.log(k+1)/np.log(2))
            
            for k in range(self.opt.topK):
                pass


    def evaluate_auc(self, Y, T):
        Y_ij = Y[list(T[:, 0] - 1), list(T[:, 1] - 1)]
        Y_ik = Y[list(T[:, 0] - 1), list(T[:, 2] - 1)]

        auc = sum((Y_ij - Y_ik) > 0) / T.shape[0]

        return auc