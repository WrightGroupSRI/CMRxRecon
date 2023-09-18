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

from matio import loadmat, writemat

import time
from cmrxrecon.io import loadmat, writemat

__all__ = ['spatial_basis', 'temporal_basis']

def temporal_basis(K):
    shx, shy, shc, shz, sht = K.shape
    ACS_mask = np.zeros((shx, shy, shz, sht), dtype=int)
    ACS_mask[:, shy // 2 - 12:shy // 2 + 12, :, :] = 1


    ACS_data = ACS_mask * np.sqrt(np.sum(K**2, axis=2))
    low_image = ifft2c(ACS_data)
    Casorati = low_image.reshape(shx * shy, shz, sht)
    basis = np.zeros((shz, sht, sht), dtype=complex)

    for z in range(shz):
        U, S, VH = np.linalg.svd(Casorati[:, z, :], full_matrices=False)
        basis[z, :, :] = VH

    return basis

def spatial_basis(X_under, TB, num_workers=os.cpu_count()):
    rcn = ifft2c(X_under)
    shx, shy, sht, shz = rcn.shape
    basis = np.zeros_like(rcn)
    args = [(x, y, z, rcn, TB) for x in range(shx) for y in range(shy) for z in range(shz)]
    with Pool(processes=num_workers) as p:
        results = p.starmap(fit_basis_at_index, args)
    for r in results:
        x, y, z, B = r
        basis[x, y, :, z] = B

    return basis

def fit_basis_at_index(x, y, z, rcn, TB):
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

def build_basis_dir(dir_path, key):
    sub_dirs = os.listdir(dir_path)
    for sub_dir in sub_dirs:
        if os.path.isfile(os.path.join(dir_path, sub_dir)):
            continue
        path = os.path.join(dir_path, sub_dir, 'T1map.mat')
        print(f'Running file {path}')
        data = loadmat(key=key, path=path)

        TB = temporal_basis(data)
        SB = spatial_basis(data, TB)
        try: 
            os.makedirs(os.path.join('/home/kadotab/scratch/dataset', key, sub_dir))
        except:
            pass
        write_basis(spatial_basis=SB, temporal_basis=TB, path=os.path.join('/home/kadotab/scratch/dataset/', key, sub_dir))

if __name__ == "__main__":  
    dir_data ='/home/kadotab/projects/def-mchiew/kadotab/SingleCoil/Mapping/TrainingSet' 
    dir_fully_sampled = (os.path.join(dir_data, 'FullSample'))
    dir_04_sampled = (os.path.join(dir_data, 'AccFactor04'))
    dir_08_sampled = (os.path.join(dir_data, 'AccFactor08'))
    dir_10_sampled = (os.path.join(dir_data, 'AccFactor10'))

    #build_basis_dir(dir_fully_sampled, 'kspace_single_full')
    build_basis_dir(dir_04_sampled, 'kspace_single_sub04')
    #build_basis_dir(dir_08_sampled, 'kspace_single_sub08')
    #build_basis_dir(dir_10_sampled, 'kspace_single_sub10')
