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
import cfl

from matio import loadmat, writemat

import time

__all__ = ['spatial_temporal_basis', 'outer_product']

def norm(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))

def spatial_temporal_basis(X, L=3):
    shx, shy, shz, sht = X.shape

    Casorati = X.reshape(shx * shy, shz, sht)
    temporal_basis = np.zeros((shz, sht, L), dtype=complex)
    spatial_basis = np.zeros((shx, shy, shz, L), dtype=complex)
    Svals = np.zeros((shz, L))

    for z in range(shz):
        U, S, VH = np.linalg.svd(Casorati[:, z, :], full_matrices=False)
        spatial_basis[:, :, z, :] = U.reshape((shx, shy, sht))[:, :, :L]
        temporal_basis[z, :, :] = VH.T[:, :L]
        Svals[z, :] = S[:L]
    return spatial_basis, Svals, temporal_basis

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

def outer_product(SB, Svals, TB):
    SB_Cas = SB.reshape(SB.shape[0] * SB.shape[1], SB.shape[2], SB.shape[3])
    out = np.zeros((SB.shape[0], SB.shape[1], SB.shape[2], TB.shape[1]), dtype=complex)
    print(SB_Cas.shape, Svals.shape, TB.shape)
    for z in range(SB.shape[2]):
        arr = SB_Cas[:, z, :] @ np.diag(Svals[z]) @ TB[z].T
        out[:, :, z, :] = arr.reshape(SB.shape[0], SB.shape[1], TB.shape[1])
    return out 

if __name__ == '__main__':
    print("Running basis computation")
    fully_sampled_path = "/hdd/Data/CMRxRecon/MultiCoil/Mapping/TrainingSet/FullSample/P001/T1map.mat"
    under_sampled_path = "/hdd/Data/CMRxRecon/MultiCoil/Mapping/TrainingSet/AccFactor04/P001/T1map.mat"

    data = loadmat(key='kspace_full', path=fully_sampled_path)
    # data = loadmat(key='kspace_sub04', path=under_sampled_path)

    fft_recon = ifft2c(data)
    fft_recon_sos = np.sqrt(np.sum(fft_recon**2, axis=2))
    SB, Svals, TB = spatial_temporal_basis(fft_recon_sos, L=3)
    cfl.writecfl("fft_recon", fft_recon_sos)
    cfl.writecfl("SB", SB)

    # SB_full = spatial_basis(fft_recon_sos, TB_full)
    # cfl.writecfl("SB_full_svals", SB_full)

    # out = outer_product(SB_full, TB_full)
    # cfl.writecfl("outer_product_svals", out)

    op = outer_product(SB, Svals, TB)
    cfl.writecfl("outer_product", op)



