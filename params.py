
import numpy as np

def ParamsAll( params_tmp ):

    param = {}
    param['R'] = params_tmp.R
    param['Ns'] = params_tmp.Ns
    param['Nd'] = params_tmp.Nd
    param['sigma_S'] = params_tmp.sigma_S
    param['sigmaD'] = params_tmp.sigmaD
    param['mu'] = params_tmp.mu
    param['tao'] = params_tmp.tao
    param['muP'] = params_tmp.muP
    param['taoP'] = params_tmp.taoP
    param['minNp'] = params_tmp.minNp
    param['maxNp'] = params_tmp.maxNp    # 12 olabilir

    param['mu1P'] = params_tmp.mu1P
    param['tao1P'] = params_tmp.tao1P
    param['mu2P'] = params_tmp.mu2P
    param['tao2P'] = params_tmp.tao2P
    param['mu3P'] = params_tmp.mu3P
    param['tao3P'] = params_tmp.tao3P

    param['delta_S'] = 2 * params_tmp.R / params_tmp.Ns
    param['delta_D'] = (1 / params_tmp.Nd) * 2 * np.pi
    k = np.arange(1,params_tmp.Nd)
    param['delta_phi'] = 180 * (-np.pi + (k - 1 / 2) * param['delta_D']) / np.pi

    return param
