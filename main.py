import numpy as np

from cgl_rank import CGL_rank
from data import Data
from opt import Opts

if __name__ == '__main__':

    print("Loading data...")
    opt = Opts('cmu')
    data_cmu = Data(opt)
    print("Data loaded")
    
    '''
    if data_cmu.opt.transductive:
        pass
    elif data_cmu.opt.sparse:
        pass
    else:
    '''
    model = CGL_rank()
    model.load_data(data_cmu)
    model.train()