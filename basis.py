#!/usr/bin/env python3

###############################################################
# SPATIOTEMPORAL BASIS ESTIMATION
# functions to estimate spatial and temporal basis
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: May 29, 2023 
###############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool

import os
import sys
from sklearn import linear_model

import mat73
import hdf5storage as hf
import scipy.io as scio

VERBOSE = True


def temporal_basis(K):
    shx, shy, shz, sht = K.shape
    ACS_mask = np.zeros_like(data, dtype=int)
    ACS_mask[:, shy // 2 - 12:shy // 2 + 12, :, :] = 1


    ACS_data = ACS_mask * K
    low_image = ifft2c(ACS_data)
    Casorati = low_image.reshape(shx * shy, shz, sht)
    basis = np.zeros((shz, sht, sht), dtype=complex)

    for z in range(shz):
        U, S, VH = np.linalg.svd(Casorati[:, z, :], full_matrices=False)
        basis[z, :, :] = VH

    return basis

def spatial_basis(X_under, TB):
    rcn = ifft2c(X_under)
    shx, shy, shz, sht = rcn.shape
    basis = np.zeros_like(rcn)
    args = [(x, y, z, rcn, TB) for x in range(shx) for y in range(shy) for z in range(shz)]
    with Pool(5) as p:
        results = p.starmap(fit_basis_at_index, args)
    for r in results:
        x, y, z, B = r
        basis[x, y, z, :] = B

    return basis

def fit_basis_at_index(x, y, z, rcn, TB):
    print("Processing entry", x, y, z, end="\r")
    X = rcn[x, y, z, :]
    T = TB[z]
    real_lr = linear_model.LinearRegression()
    real_lr.fit(np.real(T), np.real(X))
    imag_lr = linear_model.LinearRegression()
    imag_lr.fit(np.imag(T), np.imag(X))
    return (x, y, z, real_lr.coef_ + 1j * imag_lr.coef_)

def write_basis(spatial_basis=None, temporal_basis=None, path=None, contrast="T1"):
    out = {}
    out["spatial_basis"] = spatial_basis 
    out["temporal_basis"] = temporal_basis
    writemat(key="spatial_basis", data=spatial_basis, path=os.path.join(path, "spatial_basis.mat"))
    writemat(key="temporal_basis", data=temporal_basis, path=os.path.join(path, "temporal_basis.mat"))
    return True


def loadmat(key=None, path=None):
    data = mat73.loadmat(path)
    array = data.get(key)
    return array

def ifft2c(x):
    # 获取 x 的 shape
    S = np.shape(x)

    # 计算缩放因子
    fctr = S[0] * S[1]

    # 重塑 x
    x = np.reshape(x, (S[0], S[1], np.prod(S[2:])))

    # 初始化结果数组
    res = np.zeros(np.shape(x), dtype=complex)

    # 对每一个通道执行二维傅立叶逆变换
    for n in range(np.shape(x)[2]):
        res[:,:,n] = np.sqrt(fctr) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x[:,:,n])))

    # 重塑结果数组
    res = np.reshape(res, S)

    return res

def writemat(key=None, data=None, path=None):
    assert(key), "Please pass in key"
    assert(data.ndim > 0), "Please pass in data"
    assert(path), "Please pass in path"
    hf.savemat(path, {key:data}, appendmat=False)
    return path
  
if __name__ == '__main__':
    fully_sampled_path = "/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/FullSample/P001/T1map.mat"
    data = loadmat(key='kspace_single_full', path=fully_sampled_path)

    # writepath = writemat(key='kspace_single_full', data=data, path = "/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/FullSample/P001/T1map_write.mat")
    # data_read = loadmat(key='kspace_single_full', path=writepath)
    # print(np.allclose(data, data_read))

    fft_recon = ifft2c(data)

    under_sampled_path = "/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/AccFactor04/P001/T1map.mat"
    under_data = loadmat(key='kspace_single_sub04', path=under_sampled_path)

    TB = temporal_basis(under_data)
    SB = spatial_basis(under_data, TB)
    SB_full = spatial_basis(data, TB)

    write_basis(spatial_basis=SB, temporal_basis=TB, path="/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/AccFactor04/P001/")
    write_basis(spatial_basis=SB_full, temporal_basis=TB, path="/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/FullSample/P001/")

