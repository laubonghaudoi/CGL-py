class Opts(object):
    def __init__(self, dataset, sparse=False):
        self.course_file = 'data/' + dataset + '.lsvm'
        self.prereq_file = 'data/' + dataset + '.link'
        self.sparse = sparse
        # CGL-sparse
        if sparse:
            self.maxIter = 200
            self.lam = 10
            self.eta = 1e-3
        # CGL
        else:
            self.maxIter = 10
        
        self.backTracking = True
        self.C_pool = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

        # CGL-trans
        self.transductive = False
        self.knn = 60   # Node degree of kNN graph
        self.diffusion = False

        self.topK = 3
        self.quiet = True