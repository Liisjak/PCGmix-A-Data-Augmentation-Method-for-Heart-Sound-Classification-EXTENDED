from tkinter import wantobjects
import numpy as np
import pandas as pd
import os
import torch
import random
import scipy
import copy
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import itertools
from tsp_solver.greedy import solve_tsp
from tsp_solver.util import path_cost
from audiomentations import AddGaussianSNR
from scipy.spatial.distance import pdist, squareform
from python_tsp.heuristics import solve_tsp_local_search
import time

import latent_space
import saliency
import train_model
import utils

# Load the wav to CVD mappings
ROOT = os.path.join('..', '..', '..', 'mnt', 'eol', 'Zacasno', 'davidsusic', 'CHF')
CVDS_MAP = os.path.join(ROOT, 'data', 'physionet', 'cvds_map.csv')
cvds_map = pd.read_csv(CVDS_MAP)
                     
def cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, sig_len, method, device, overlap=10):
    # check how long the connected instance will be and limit it to sig_len
    d1_cut_len = f1[cut]
    d2_cut_len = f2[-1]-f2[cut]
    d_connected_len = d1_cut_len + d2_cut_len
    # define a new instance
    d_new = torch.zeros((num_channels, sig_len)).to(device)
    # connect the first part of d1 and second part of d2
    last_f1_paste_frame = min(d_connected_len, sig_len)
    d_new[:, 0:f1[cut]] = d1[:, 0:f1[cut]]
    d_new[:, f1[cut]:last_f1_paste_frame] = d2[:, f2[cut]:f2[cut]+last_f1_paste_frame-f1[cut]]
    if '(smooth)' in method:
        # limit overlap if parts are shorter than the segments' parts
        overlap = min(overlap, d1_cut_len, d2_cut_len, f1[-1]-f1[cut], f2[cut])
        # smooth the connection point
        interp_coefs1 = 1-sigmoid(overlap)
        interp_coefs1 = np.repeat(interp_coefs1[None, :], num_channels, axis=0)
        interp_coefs1 = torch.tensor(interp_coefs1).to(device)
        interp_coefs2 = sigmoid(overlap)
        interp_coefs2 = np.repeat(interp_coefs2[None, :], num_channels, axis=0)
        interp_coefs2 = torch.tensor(interp_coefs2).to(device)
        d_new[:, f1[cut]-overlap:f1[cut]+overlap] = torch.add(torch.mul(d1[:, f1[cut]-overlap:f1[cut]+overlap], interp_coefs1), torch.mul(d2[:, f2[cut]-overlap:f2[cut]+overlap], interp_coefs2))
    # calculate the new frames
    f_new = list(f1[:cut+1])
    f_new = f_new + list(f2[cut+1:]-f2[cut]+f1[cut])
    f_new = np.array(f_new)
    if f_new[-1] > last_f1_paste_frame:
        f_new[-1] = last_f1_paste_frame
    return d_new, f_new

def optimal_displacement_max_envelope(s1, s2, lam):
    len_s1 = len(s1)
    len_s2 = len(s2)
    if len_s1 > len_s2:
        # Zero pad s2 to the length of s1
        s2_padded = np.pad(s2, (0, len_s1 - len_s2), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        # Iterate over all possible displacements
        for displacement in range(len_s1 - len_s2 + 1):
            # Create the current shifted version of s2
            current_s2_shifted = np.roll(s2_padded, displacement)
            # Calculate the sum of maximum values at each index
            current_sum = np.sum(s1[:displacement]) + \
                          np.sum(np.maximum(s1[displacement:displacement+len_s2],current_s2_shifted[displacement:displacement+len_s2])) + \
                          np.sum(s1[displacement+len_s2:])
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
    else: #len_s1 < len_s2
        # Zero pad s1 to the length of s2
        s1_padded = np.pad(s1, (0, len_s2 - len_s1), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        # Iterate over all possible displacements
        for displacement in range(len_s2 - len_s1 + 1):
            # Create the current shifted version of s2
            current_s1_shifted = np.roll(s1_padded, displacement)
            # Calculate the sum of maximum values at each index of the cropped instances
            current_sum = np.sum(np.maximum(s2[displacement:displacement+len_s1], current_s1_shifted[displacement:displacement+len_s1]))
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
    return opt_displacement

def optimal_displacement_max_sum(s1, s2, lam):
    len_s1 = len(s1)
    len_s2 = len(s2)
    if len_s1 > len_s2:
        # Zero pad s2 to the length of s1
        s2_padded = np.pad(s2, (0, len_s1 - len_s2), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        # Iterate over all possible displacements
        for displacement in range(len_s1 - len_s2 + 1):
            # Create the current shifted version of s2
            current_s2_shifted = np.roll(s2_padded, displacement)
            # Calculate the sum of maximum values at each index
            current_sum = np.sum(s1[:displacement]) + \
                          np.sum(s1[displacement:displacement+len_s2]*lam+current_s2_shifted[displacement:displacement+len_s2]*(1-lam)) + \
                          np.sum(s1[displacement+len_s2:])
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
    else: #len_s1 < len_s2
        # Zero pad s1 to the length of s2
        s1_padded = np.pad(s1, (0, len_s2 - len_s1), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        # Iterate over all possible displacements
        for displacement in range(len_s2 - len_s1 + 1):
            # Create the current shifted version of s2
            current_s1_shifted = np.roll(s1_padded, displacement)
            # Calculate the sum of maximum values at each index of the cropped instances
            current_sum = np.sum(current_s1_shifted[displacement:displacement+len_s1]*lam + s2[displacement:displacement+len_s1]*(1-lam))
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
    return opt_displacement

def normalize_saliency_state(s1, s2, norm):
    if norm == 'both':
        global_min = np.min([np.min(s1), np.min(s2)])
        s1 = s1-global_min
        s2 = s2-global_min
        global_max = np.max([np.max(s1), np.max(s2)])
        s1 = s1/global_max
        s2 = s2/global_max
    elif norm == 'single':
        s1 = s1-np.min(s1)
        s1 = s1/np.max(s1)
        s2 = s2-np.min(s2)
        s2 = s2/np.max(s2)
    return s1, s2

def optimal_displacement_puzzle(s1, s2, lam, opt_with_lam):
    len_s1 = len(s1)
    len_s2 = len(s2)
    if len_s1 > len_s2:
        # Zero pad s2 to the length of s1
        s2_padded = np.pad(s2, (0, len_s1 - len_s2), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        # Iterate over all possible displacements
        for displacement in range(len_s1 - len_s2 + 1):
            # Create the current shifted version of s2
            current_s2_shifted = np.roll(s2_padded, displacement)
            # Calculate the sum of maximum values at each index
            if not opt_with_lam:
                mask = (current_s2_shifted[displacement:displacement+len_s2] > s1[displacement:displacement+len_s2]).astype(int)
            else: 
                mask = (current_s2_shifted[displacement:displacement+len_s2] > lam[0]).astype(int)
            mask_r = 1-mask
            current_sum = np.sum(s1[:displacement]) + \
                            np.sum(s1[displacement:displacement+len_s2]*mask_r + current_s2_shifted[displacement:displacement+len_s2]*mask) + \
                            np.sum(s1[displacement+len_s2:])
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
                opt_mask = mask
    else: #len_s1 < len_s2
        # Zero pad s1 to the length of s2
        s1_padded = np.pad(s1, (0, len_s2 - len_s1), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        # Iterate over all possible displacements
        for displacement in range(len_s2 - len_s1 + 1):
            # Create the current shifted version of s2
            current_s1_shifted = np.roll(s1_padded, displacement)
            # Calculate the sum of maximum values at each index of the cropped instances
            if not opt_with_lam:
                mask = (s2[displacement:displacement+len_s1] > current_s1_shifted[displacement:displacement+len_s1]).astype(int)
            else: 
                mask = (s2[displacement:displacement+len_s1] > lam[0]).astype(int)
            mask_r = 1-mask
            current_sum = np.sum(current_s1_shifted[displacement:displacement+len_s1]*mask_r + s2[displacement:displacement+len_s1]*mask)
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
                opt_mask = mask
    return opt_displacement, opt_mask

def smooth_mask_with_k2(mask, mask_kernel):
    mask = np.convolve(mask, mask_kernel, mode='same')
    mask = mask/np.max(mask)
    mask = np.nan_to_num(mask, nan=0) # when mask is full zero, normalization will turn it to full nan
    mask = np.clip(mask, a_min=0, a_max=1) # not sure if needed
    return mask

def plot_heartbeat_mixing(lam, m, m_r, x1, x2):
    print(f'{lam=}')
    plt.figure(figsize=(10, 1.5))
    plt.axhline(y=lam, linestyle='--')
    plt.plot(m, color='k', linestyle='--')
    plt.plot(x1, color='b')
    plt.plot(x2, color='r')
    plt.plot(x1*m_r + x2*m, color='k')
    plt.show()
    plt.close()

def mixup_keepdur_multidim_tensors_salopt(d1, d2, f1, f2, sal1, sal2, lam, method, random_seed):
    d_new = d1.clone()
    sal_new = copy.deepcopy(sal1)
    lam_np = lam.detach().cpu().numpy()
    if '(saloptenv' in method:
        optimal_displacement_approach = optimal_displacement_max_envelope
    elif '(saloptsum' in method:
        optimal_displacement_approach = optimal_displacement_max_sum
    ### this is the approach: data = data*lams_out + data[mix_indices]*(1-lams_out)
    # mix S1 of d1 with S1 of d2
    len_s1_d1 = f1[1]-f1[0]
    len_s1_d2 = f2[1]-f2[0]
    sal1_s1 = sal1[f1[0]:f1[1]]
    sal2_s1 = sal2[f2[0]:f2[1]]
    if len_s1_d1 == len_s1_d2:
        disp_s1 = 0
        d_new[ :, f1[0]:f1[1]] = d_new[ :, f1[0]:f1[1]]*lam + d2[ :, f2[0]:f2[1]]*(1-lam)
        sal_new[f1[0]:f1[1]] = sal_new[f1[0]:f1[1]]*lam_np + sal2[f2[0]:f2[1]]*(1-lam_np)
    elif len_s1_d1 > len_s1_d2:
        disp_s1 = optimal_displacement_approach(sal1_s1, sal2_s1, lam_np)
        d_new[ :, f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2] = d_new[ :, f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2]*lam + d2[ :, f2[0]:f2[1]]*(1-lam)
        sal_new[f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2] = sal_new[f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2]*lam_np + sal2[f2[0]:f2[1]]*(1-lam_np)
    else: #(len_s1_d1 < len_s1_d2)
        disp_s1 = optimal_displacement_approach(sal1_s1, sal2_s1, lam_np)
        d_new[ :, f1[0]:f1[1]] = d_new[ :, f1[0]:f1[1]]*lam + d2[ :, f2[0]+disp_s1:f2[0]+disp_s1+len_s1_d1]*(1-lam)
        sal_new[f1[0]:f1[1]] = sal_new[f1[0]:f1[1]]*lam_np + sal2[f2[0]+disp_s1:f2[0]+disp_s1+len_s1_d1]*(1-lam_np)
    # mix systole of d1 with systole of d2
    len_sys_d1 = f1[2]-f1[1]
    len_sys_d2 = f2[2]-f2[1]
    sal1_sys = sal1[f1[1]:f1[2]]
    sal2_sys = sal2[f2[1]:f2[2]]
    if len_sys_d1 == len_sys_d2:
        disp_sys = 0
        d_new[ :, f1[1]:f1[2]] = d_new[ :, f1[1]:f1[2]]*lam + d2[ :, f2[1]:f2[2]]*(1-lam)
        sal_new[f1[1]:f1[2]] = sal_new[f1[1]:f1[2]]*lam_np + sal2[f2[1]:f2[2]]*(1-lam_np)
    elif len_sys_d1 > len_sys_d2:
        disp_sys = optimal_displacement_approach(sal1_sys, sal2_sys, lam_np)
        d_new[ :, f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2] = d_new[ :, f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2]*lam + d2[ :, f2[1]:f2[2]]*(1-lam)
        sal_new[f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2] = sal_new[f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2]*lam_np + sal2[f2[1]:f2[2]]*(1-lam_np)
    else: #(len_sys_d1 < len_sys_d2)
        disp_sys = optimal_displacement_approach(sal1_sys, sal2_sys, lam_np)
        d_new[ :, f1[1]:f1[2]] = d_new[ :, f1[1]:f1[2]]*lam + d2[ :, f2[1]+disp_sys:f2[1]+disp_sys+len_sys_d1]*(1-lam)
        sal_new[f1[1]:f1[2]] = sal_new[f1[1]:f1[2]]*lam_np + sal2[f2[1]+disp_sys:f2[1]+disp_sys+len_sys_d1]*(1-lam_np)
    # mix S2 of d1 with S2 of d2
    len_s2_d1 = f1[3]-f1[2]
    len_s2_d2 = f2[3]-f2[2]
    sal1_s2 = sal1[f1[2]:f1[3]]
    sal2_s2 = sal2[f2[2]:f2[3]]
    if len_s2_d1 == len_s2_d2:
        disp_s2 = 0
        d_new[ :, f1[2]:f1[3]] = d_new[ :, f1[2]:f1[3]]*lam + d2[ :, f2[2]:f2[3]]*(1-lam)
        sal_new[f1[2]:f1[3]] = sal_new[f1[2]:f1[3]]*lam_np + sal2[f2[2]:f2[3]]*(1-lam_np)
    elif len_s2_d1 > len_s2_d2:
        disp_s2 = optimal_displacement_approach(sal1_s2, sal2_s2, lam_np)
        d_new[ :, f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2] = d_new[ :, f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2]*lam + d2[ :, f2[2]:f2[3]]*(1-lam)
        sal_new[f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2] = sal_new[f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2]*lam_np + sal2[f2[2]:f2[3]]*(1-lam_np)
    else: #(len_s2_d1 < len_s2_d2)
        disp_s2 = optimal_displacement_approach(sal1_s2, sal2_s2, lam_np)
        d_new[ :, f1[2]:f1[3]] = d_new[ :, f1[2]:f1[3]]*lam + d2[ :, f2[2]+disp_s2:f2[2]+disp_s2+len_s2_d1]*(1-lam)
        sal_new[f1[2]:f1[3]] = sal_new[f1[2]:f1[3]]*lam_np + sal2[f2[2]+disp_s2:f2[2]+disp_s2+len_s2_d1]*(1-lam_np)
    # mix diastole of d1 with systole of d2
    len_dia_d1 = f1[4]-f1[3]
    len_dia_d2 = f2[4]-f2[3]
    sal1_dia = sal1[f1[3]:f1[4]]
    sal2_dia = sal2[f2[3]:f2[4]]
    if len_dia_d1 == len_dia_d2:
        disp_dia = 0
        d_new[ :, f1[3]:f1[4]] = d_new[ :, f1[3]:f1[4]]*lam + d2[ :, f2[3]:f2[4]]*(1-lam)
        sal_new[f1[3]:f1[4]] = sal_new[f1[3]:f1[4]]*lam_np + sal2[f2[3]:f2[4]]*(1-lam_np)
    elif len_dia_d1 > len_dia_d2:
        disp_dia = optimal_displacement_approach(sal1_dia, sal2_dia, lam_np)
        d_new[ :, f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2] = d_new[ :, f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2]*lam + d2[ :, f2[3]:f2[4]]*(1-lam)
        sal_new[f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2] = sal_new[f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2]*lam_np + sal2[f2[3]:f2[4]]*(1-lam_np)
    else: #(len_dia_d1 < len_dia_d2)
        disp_dia = optimal_displacement_approach(sal1_dia, sal2_dia, lam_np)
        d_new[ :, f1[3]:f1[4]] = d_new[ :, f1[3]:f1[4]]*lam + d2[ :, f2[3]+disp_dia:f2[3]+disp_dia+len_dia_d1]*(1-lam)
        sal_new[f1[3]:f1[4]] = sal_new[f1[3]:f1[4]]*lam_np + sal2[f2[3]+disp_dia:f2[3]+disp_dia+len_dia_d1]*(1-lam_np)
    return d_new

def mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lam, method, random_seed):
    d_new = d1.clone()
    ### this is the approach: data = data*lams_out + data[mix_indices]*(1-lams_out)
    if '(rand)' not in method:
        # mix S1 of d1 with S1 of d2
        len_s1_min = min(f1[1]-f1[0], f2[1]-f2[0])
        d_new[ :, f1[0]:f1[0]+len_s1_min] = d_new[ :, f1[0]:f1[0]+len_s1_min]*lam + d2[ :, f2[0]:f2[0]+len_s1_min]*(1-lam)
        # mix systole of d1 with systole of d2
        len_sys_min = min(f1[2]-f1[1], f2[2]-f2[1])
        d_new[ :, f1[1]:f1[1]+len_sys_min] = d_new[ :, f1[1]:f1[1]+len_sys_min]*lam + d2[ :, f2[1]:f2[1]+len_sys_min]*(1-lam)
        # mix S2 of d1 with S2 of d2
        len_s2_min = min(f1[3]-f1[2], f2[3]-f2[2])
        d_new[ :, f1[2]:f1[2]+len_s2_min] = d_new[ :, f1[2]:f1[2]+len_s2_min]*lam + d2[ :, f2[2]:f2[2]+len_s2_min]*(1-lam)
        # mix diastole of d1 with systole of d2
        len_dia_min = min(f1[4]-f1[3], f2[4]-f2[3])
        d_new[ :, f1[3]:f1[3]+len_dia_min] = d_new[ :, f1[3]:f1[3]+len_dia_min]*lam + d2[ :, f2[3]:f2[3]+len_dia_min]*(1-lam)
    else:
        # mix S1 of d1 with S1 of d2
        len_s1_min = min(f1[1]-f1[0], f2[1]-f2[0])
        s1_gap = (f2[1]-f2[0]) - (f1[1]-f1[0])
        s1_disp_rand = random.Random(random_seed).randint(0, np.abs(s1_gap))
        if s1_gap>=0:
            d_new[ :, f1[0]:f1[0]+len_s1_min] = d_new[ :, f1[0]:f1[0]+len_s1_min]*lam + d2[ :, f2[0]+s1_disp_rand:f2[0]+s1_disp_rand+len_s1_min]*(1-lam)
        else:
            d_new[ :, f1[0]+s1_disp_rand:f1[0]+s1_disp_rand+len_s1_min] = d_new[ :, f1[0]+s1_disp_rand:f1[0]+s1_disp_rand+len_s1_min]*lam + d2[ :, f2[0]:f2[0]+len_s1_min]*(1-lam)
        # mix systole of d1 with systole of d2
        len_sys_min = min(f1[2]-f1[1], f2[2]-f2[1])
        sys_gap = (f2[2]-f2[1]) - (f1[2]-f1[1])
        sys_disp_rand = random.Random(random_seed).randint(0, np.abs(sys_gap))
        if sys_gap>=0:
            d_new[ :, f1[1]:f1[1]+len_sys_min] = d_new[ :, f1[1]:f1[1]+len_sys_min]*lam + d2[ :, f2[1]+sys_disp_rand:f2[1]+sys_disp_rand+len_sys_min]*(1-lam)
        else:
            d_new[ :, f1[1]+sys_disp_rand:f1[1]+sys_disp_rand+len_sys_min] = d_new[ :, f1[1]+sys_disp_rand:f1[1]+sys_disp_rand+len_sys_min]*lam + d2[ :, f2[1]:f2[1]+len_sys_min]*(1-lam)
        # mix S2 of d1 with S2 of d2
        len_s2_min = min(f1[3]-f1[2], f2[3]-f2[2])
        s2_gap = (f2[3]-f2[2]) - (f1[3]-f1[2])
        s2_disp_rand = random.Random(random_seed).randint(0, np.abs(s2_gap))
        if s2_gap>=0:
            d_new[ :, f1[2]:f1[2]+len_s2_min] = d_new[ :, f1[2]:f1[2]+len_s2_min]*lam + d2[ :, f2[2]+s2_disp_rand:f2[2]+s2_disp_rand+len_s2_min]*(1-lam)
        else:
            d_new[ :, f1[2]+s2_disp_rand:f1[2]+s2_disp_rand+len_s2_min] = d_new[ :, f1[2]+s2_disp_rand:f1[2]+s2_disp_rand+len_s2_min]*lam + d2[ :, f2[2]:f2[2]+len_s2_min]*(1-lam)
        # mix diastole of d1 with systole of d2
        len_dia_min = min(f1[4]-f1[3], f2[4]-f2[3])
        dia_gap = (f2[4]-f2[3]) - (f1[4]-f1[3])
        dia_disp_rand = random.Random(random_seed).randint(0, np.abs(dia_gap))
        if dia_gap>=0:
            d_new[ :, f1[3]:f1[3]+len_dia_min] = d_new[ :, f1[3]:f1[3]+len_dia_min]*lam + d2[ :, f2[3]+dia_disp_rand:f2[3]+dia_disp_rand+len_dia_min]*(1-lam)
        else:
            d_new[ :, f1[3]+dia_disp_rand:f1[3]+dia_disp_rand+len_dia_min] = d_new[ :, f1[3]+dia_disp_rand:f1[3]+dia_disp_rand+len_dia_min]*lam + d2[ :, f2[3]:f2[3]+len_dia_min]*(1-lam)
    return d_new

def cutmix_keepdur_multidim_tensors(d1, d2, f1, f2, method, random_seed):
    d_new = d1.clone()
    if '(rand)' not in method:
        # swap systole of d1 with systole of d2
        len_sys_min = min(f1[2]-f1[1], f2[2]-f2[1])
        d_new[:, f1[1]:f1[1]+len_sys_min] = d2[:, f2[1]:f2[1]+len_sys_min]
        # swap diastole of d1 with systole of d2
        len_dia_min = min(f1[4]-f1[3], f2[4]-f2[3])
        d_new[:, f1[3]:f1[3]+len_dia_min] = d2[:, f2[3]:f2[3]+len_dia_min]
    else:
        # swap systole of d1 with systole of d2
        len_sys_min = min(f1[2]-f1[1], f2[2]-f2[1])
        sys_gap = (f2[2]-f2[1]) - (f1[2]-f1[1])
        sys_start = random.Random(random_seed).randint(0, np.abs(sys_gap))
        if sys_gap >= 0:
            d_new[:, f1[1]:f1[2]] = d2[:, f2[1]+sys_start:f2[1]+sys_start+len_sys_min]
        else:
            d_new[:, f1[1]+sys_start:f1[1]+sys_start+len_sys_min] = d2[:, f2[1]:f2[2]]
        # swap diastole of d1 with systole of d2
        len_dia_min = min(f1[4]-f1[3], f2[4]-f2[3])
        dia_gap = (f2[4]-f2[3]) - (f1[4]-f1[3])
        dia_start = random.Random(random_seed).randint(0, np.abs(dia_gap))
        if dia_gap >= 0:
            d_new[:, f1[3]:f1[4]] = d2[:, f2[3]+dia_start:f2[3]+dia_start+len_dia_min]
        else:
            d_new[:, f1[3]+dia_start:f1[3]+dia_start+len_dia_min] = d2[:, f2[3]:f2[4]]
    return d_new

### ========================================================================================================================
# FIND PAIRS FOR MIXING METHODS
### ========================================================================================================================

def distances_to_rankings(distance_matrix, k_num):
    m = distance_matrix.shape[0]
    rankings = np.zeros_like(distance_matrix, dtype=int)
    # Iterate over each row to compute rankings
    for i in range(m):
        sorted_indices = np.argsort(distance_matrix[i])
        # Assign ranks starting from 0 for the closest (self-distance)
        rankings[i, sorted_indices] = np.arange(m)
        # Closest k_num instances (excluding self-distance) get rank 1
        rankings[i, sorted_indices[1:k_num + 1]] = 1
        # Adjust ranks for the rest
        rankings[i, sorted_indices[k_num + 1:]] -= (k_num - 1)
    return rankings

def get_same_label_closestknn(target_ohe, data, k_num, random_seed, RESULTS_ARGS, step_number, batch_size):
    SAVEDIR = utils.check_folder(os.path.join(RESULTS_ARGS, 'closestknn'))
    target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
    size = len(target)
    wavs_dict = {}
    for i, t_i in enumerate(target):
        t_i = t_i[0]
        if t_i not in wavs_dict:
            wavs_dict[t_i] = list([i])
        else:
            wavs_dict[t_i].append(i)
    # get latent space features
    fts_latent = latent_space.generate_latent_space(data.clone())
    fts0 = fts_latent[wavs_dict[0]]
    fts1 = fts_latent[wavs_dict[1]]
    # if k_num is at least as large as the batch size, the solution is trivial
    if k_num >=  batch_size:
        # Reshuffle the indices for each wav
        mix_indices = np.arange(0, size, 1)
        for key in wavs_dict:
            mix_indices[wavs_dict[key]] = random.Random(random_seed).sample(list(mix_indices[wavs_dict[key]]), len(mix_indices[wavs_dict[key]]))
        # save the distance
        total_distance = np.sum(np.linalg.norm(fts_latent - fts_latent[mix_indices], axis=1))
        FILENAME = os.path.join(SAVEDIR, f'totaldistance_{step_number}.txt')
        np.savetxt(FILENAME, [total_distance])
        return mix_indices
    # compute pairwise distances
    pairwise_distances0 = pdist(fts0, metric='euclidean')
    pairwise_distances1 = pdist(fts1, metric='euclidean')
    # compute distance matrices based on pairwise distances
    distance_matrix0 = squareform(pairwise_distances0)
    distance_matrix1 = squareform(pairwise_distances1)
    # swap distances with rankings
    distance_matrix0 = distances_to_rankings(distance_matrix0, k_num)
    distance_matrix1 = distances_to_rankings(distance_matrix1, k_num)
    # solve TSP for each label separately
    mix_indices = np.arange(0, size, 1) 
    for label, dist_matrix in zip([0, 1], [distance_matrix0, distance_matrix1]):
        path = solve_tsp(dist_matrix, endpoints = (0,0)) # greedy solver (symmetric)
        path, _ = solve_tsp_local_search(dist_matrix, x0=path[:-1]) # (random, gives different solutions) heuristic solver (asymmetric): use symmetric solution as initial guess
        path.append(path[0])
        indices_label_first=np.array(path[:-1])
        indices_label_second=np.roll(path[:-1], -1)
        # reorder mix indices to correspond to the solution of the current label
        wavs_dict_label = np.array(wavs_dict[label])
        wavs_dict_label_indices_label_first = wavs_dict_label[indices_label_first]
        wavs_dict_label_indices_label_second = wavs_dict_label[indices_label_second]
        mix_indices[wavs_dict_label_indices_label_first] = mix_indices[wavs_dict_label_indices_label_second]
    # save the distance
    total_distance = np.sum(np.linalg.norm(fts_latent - fts_latent[mix_indices], axis=1))
    FILENAME = os.path.join(SAVEDIR, f'totaldistance_{step_number}.txt')
    np.savetxt(FILENAME, [total_distance])
    return mix_indices

def get_same_label_closestbins(target_ohe, data, num_bins, random_seed, RESULTS_ARGS, step_number):
    SAVEDIR = utils.check_folder(os.path.join(RESULTS_ARGS, 'closestbins'))
    target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
    size = len(target)
    wavs_dict = {}
    for i, t_i in enumerate(target):
        t_i = t_i[0]
        if t_i not in wavs_dict:
            wavs_dict[t_i] = list([i])
        else:
            wavs_dict[t_i].append(i)
    # get latent space features
    fts_latent = latent_space.generate_latent_space(data.clone())
    fts0 = fts_latent[wavs_dict[0]]
    fts1 = fts_latent[wavs_dict[1]]
    # if there is only one bin, the solution is trivial
    if num_bins == 1:
        # Reshuffle the indices for each wav
        mix_indices = np.arange(0, size, 1)
        for key in wavs_dict:
            mix_indices[wavs_dict[key]] = random.Random(random_seed).sample(list(mix_indices[wavs_dict[key]]), len(mix_indices[wavs_dict[key]]))
        # save the distance
        total_distance = np.sum(np.linalg.norm(fts_latent - fts_latent[mix_indices], axis=1))
        FILENAME = os.path.join(SAVEDIR, f'totaldistance_{step_number}.txt')
        np.savetxt(FILENAME, [total_distance])
        return mix_indices
    # compute pairwise distances
    pairwise_distances0 = pdist(fts0, metric='euclidean')
    pairwise_distances1 = pdist(fts1, metric='euclidean')
    # compute the bins
    all_values_max = np.max([np.max(pairwise_distances0), np.max(pairwise_distances1)])
    all_values_min = np.min([np.min(pairwise_distances0), np.min(pairwise_distances1)])
    # compute bin edges
    bin_edges = np.linspace(all_values_min, all_values_max, num_bins + 1)
    # digize the distances to get bin indices
    bin_indices0 = np.digitize(pairwise_distances0, bin_edges, right=True)
    bin_indices1 = np.digitize(pairwise_distances1, bin_edges, right=True)
    # adjust indices to ensure they are between 1 and num_bins
    bin_indices0 = np.clip(bin_indices0, 1, num_bins)
    bin_indices1 = np.clip(bin_indices1, 1, num_bins)
    # compute distance matrices based on bins
    distance_matrix0 = squareform(bin_indices0)
    distance_matrix1 = squareform(bin_indices1)
    # solve TSP for each label separately
    mix_indices = np.arange(0, size, 1) 
    for label, dist_matrix in zip([0, 1], [distance_matrix0, distance_matrix1]):
        path = solve_tsp(dist_matrix, endpoints = (0,0))
        indices_label_first=np.array(path[:-1])
        indices_label_second=np.roll(path[:-1], -1)
        # reorder mix indices to correspond to the solution of the current label
        wavs_dict_label = np.array(wavs_dict[label])
        wavs_dict_label_indices_label_first = wavs_dict_label[indices_label_first]
        wavs_dict_label_indices_label_second = wavs_dict_label[indices_label_second]
        mix_indices[wavs_dict_label_indices_label_first] = mix_indices[wavs_dict_label_indices_label_second]
    # save the distance
    total_distance = np.sum(np.linalg.norm(fts_latent - fts_latent[mix_indices], axis=1))
    FILENAME = os.path.join(SAVEDIR, f'totaldistance_{step_number}.txt')
    np.savetxt(FILENAME, [total_distance])
    return mix_indices

def get_same_label_mix_indices(target_ohe, random_seed):
    target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
    size = len(target)
    wavs_dict = {}
    for i, t_i in enumerate(target):
        t_i = t_i[0]
        if t_i not in wavs_dict:
            wavs_dict[t_i] = list([i])
        else:
            wavs_dict[t_i].append(i)
    # Reshuffle the indices for each wav
    mix_indices = np.arange(0, size, 1)
    for key in wavs_dict:
        mix_indices[wavs_dict[key]] = random.Random(random_seed).sample(list(mix_indices[wavs_dict[key]]), len(mix_indices[wavs_dict[key]]))
    return mix_indices

def get_same_cvd_mix_indices(wav, random_seed):
    size=len(wav)
    cvds = [cvds_map[cvds_map['wav']==w].iloc[0]['diagnosis'] for w in wav]
    cvds_dict = {c:[] for c in list(set(cvds))}
    for i, c_i in enumerate(cvds):
        cvds_dict[c_i].append(i)
    # Reshuffle the indices for each cvd
    mix_indices = np.arange(0, size, 1)
    for key in cvds_dict:
        mix_indices[cvds_dict[key]] = random.Random(random_seed).sample(list(mix_indices[cvds_dict[key]]), len(mix_indices[cvds_dict[key]]))
    return mix_indices

def get_same_wav_mix_indices(wav, random_seed):
    size = len(wav)
    wavs_dict = {}
    for i, w_i in enumerate(wav):
        if w_i not in wavs_dict:
            wavs_dict[w_i] = list([i])
        else:
            wavs_dict[w_i].append(i)
    # Reshuffle the indices for each wav
    mix_indices = np.arange(0, size, 1)
    for key in wavs_dict:
        mix_indices[wavs_dict[key]] = random.Random(random_seed).sample(list(mix_indices[wavs_dict[key]]), len(mix_indices[wavs_dict[key]]))  
    return mix_indices

def get_same_dataset_mix_indices(target_ohe, wav, random_seed):
    target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
    size = len(wav)
    wavs_dict = {}
    for i, (w_i, t_i) in enumerate(zip(wav, target)):
        key = f'{w_i[0]}_{t_i[0]}'
        if key not in wavs_dict:
            wavs_dict[key] = list([i])
        else:
            wavs_dict[key].append(i)
    # Reshuffle the indices for each wav
    mix_indices = np.arange(0, size, 1)
    for key in wavs_dict:
        mix_indices[wavs_dict[key]] = random.Random(random_seed).sample(list(mix_indices[wavs_dict[key]]), len(mix_indices[wavs_dict[key]])) 
    return mix_indices

def get_same_length_mix_indices(target_ohe, frames, random_seed, batch_size, method):
    target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
    size = len(target)
    lengths = [f[-1].item() for f in frames]
    lengths_min = np.min(lengths)
    lengths_max = np.max(lengths)
    num_bins = batch_size//100 # approx 25 per (bin, class) for batch_size=500
    if '(5bins)' in method:
        num_bins = 5
    if '(10bins)' in method:
        num_bins = 10
    bins = np.linspace(lengths_min-1, lengths_max+1, num_bins+1)
    bins_inds = np.digitize(lengths, bins)
    wavs_dict = {}
    for i, (b_i, t_i) in enumerate(zip(bins_inds, target)):
        key = f'{t_i[0]}_{b_i}'
        if key not in wavs_dict:
            wavs_dict[key] = list([i])
        else:
            wavs_dict[key].append(i)
    # Reshuffle the indices for each wav
    mix_indices = np.arange(0, size, 1)
    for key in wavs_dict:
        mix_indices[wavs_dict[key]] = random.Random(random_seed).sample(list(mix_indices[wavs_dict[key]]), len(mix_indices[wavs_dict[key]]))
    return mix_indices

def get_optimal_sal_mix_indices(saliency_map, target_ohe, frames, device, bin_width):
    # find indices of the same label
    target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
    size = len(target)
    wavs_dict = {}
    for i, t_i in enumerate(target):
        t_i = t_i[0]
        if t_i not in wavs_dict:
            wavs_dict[t_i] = list([i])
        else:
            wavs_dict[t_i].append(i)
    # find mix indices for all labels in loop 
    labels = list(set(target[:, 0])) # unique labels in this batch
    mix_indices = np.arange(0, size, 1) 
    path_total_cost = 0
    for label in labels:
        saliency_map_label, frames_label = saliency_map[wavs_dict[label]], frames[wavs_dict[label]]
        label_size = saliency_map_label.shape[0]
        dist_matrix = [[0 for col in range(label_size)] for row in range(label_size)]
        for i, (s1, f1) in enumerate(zip(saliency_map_label, frames_label)):
            for j, (s2, f2) in enumerate(zip(saliency_map_label, frames_label)):
                if j <= i:
                    continue
                sys_sal1 = s1[:, f1[1]:f1[2]]
                sys_sal2 = s2[:, f2[1]:f2[2]]
                _, sys_max_sum, _ = opt_sal_overlap(sys_sal1, sys_sal2, bin_width, device)
                dia_sal1 = s1[:, f1[3]:f1[4]]
                dia_sal2 = s2[:, f2[3]:f2[4]]
                _, dia_max_sum, _ = opt_sal_overlap(dia_sal1, dia_sal2, bin_width, device)
                max_sum = sys_max_sum + dia_max_sum
                dist_matrix[i][j] = max_sum
        dist_matrix = np.array(dist_matrix) + np.array(dist_matrix).T
        # flip distance matrix from max to min as traveling salesman is a minimization problem
        dmmax = np.array([[np.max(dist_matrix) for col in range(label_size)] for row in range(label_size)])
        dist_matrix = dmmax - dist_matrix
        np.fill_diagonal(dist_matrix, 0) # only for better clarity, not needed tho
        # solve traveling salesman problem
        path = solve_tsp(dist_matrix, endpoints = (0,0))
        indices_label_first=np.array(path[:-1])
        indices_label_second=np.roll(path[:-1], -1)
        path_total_cost += path_cost(dist_matrix, path)
        # reorder mix indices to correspond to the solution of the current label
        wavs_dict_label = np.array(wavs_dict[label])
        wavs_dict_label_indices_label_first = wavs_dict_label[indices_label_first]
        wavs_dict_label_indices_label_second = wavs_dict_label[indices_label_second]
        mix_indices[wavs_dict_label_indices_label_first] = mix_indices[wavs_dict_label_indices_label_second]
    return mix_indices

def get_same_umc_subset_mix_indices(target_ohe, wav, random_seed):
    target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
    size = len(wav)
    wavs_dict = {}
    for i, (w_i, t_i) in enumerate(zip(wav, target)):
        if len(w_i.split('_')[0]) == 3:
            key = f'new_{t_i[0]}'
            if key not in wavs_dict:
                wavs_dict[key] = list([i])
            else:
                wavs_dict[key].append(i)
        else:
            key = f'old_{t_i[0]}'
            if key not in wavs_dict:
                wavs_dict[key] = list([i])
            else:
                wavs_dict[key].append(i)
    # Reshuffle the indices for each wav
    mix_indices = np.arange(0, size, 1)
    for key in wavs_dict:
        mix_indices[wavs_dict[key]] = random.Random(random_seed).sample(list(mix_indices[wavs_dict[key]]), len(mix_indices[wavs_dict[key]])) 
    return mix_indices

### ========================================================================================================================
# AUGMENTATION METHODS
### ========================================================================================================================

def get_lambda(alpha=1.0, random_seed = 4):
    '''Returns lambda'''
    if alpha > 0.:
        np.random.seed(random_seed)
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam  
    
def sigmoid(overlap):
    sigmoid_array = np.array([1.0 / (1.0 + np.exp(-x)) for x in np.linspace(-8, 8, overlap*2)])
    sigmoid_array[0] = 0
    sigmoid_array[-1] = 1
    return sigmoid_array

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return ret

def time_warp(x, sigma=0.05, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def augment(args, data, target_ohe, frames, wav, step_counter, model, device, RESULTS_ARGS):
    # Implemented methods
    methods_implemented = ['durratiocutmix',
                           'lengthcutmix',
                           'datasetcutmix',
                           'wav-durratiocutmix',
                           'wavcutmix',
                           'lc-nointrusion',
                           'labelcutmix',
                           'swapsysdia',
                           's1s2mask',
                           'cont-cutmix',
                           'saliency-cutmix',
                           'latentmixup', 
                           'manifold-cutmix(ch)', 
                           'manifold-cutmix',
                           'manifold-cutout(ch)',
                           'manifold-cutout', 
                           'cutmix(ch)', 
                           'cutmix',
                           'cutout(ch)', 
                           'cutout',
                           'gaussiannoise',
                           'magnitudewarp',
                           'timewarp',
                           'mixup',
                           'timemask',
                           'durratiomixup',
                           'durmixmagwarp',
                           'respiratoryscale',
                           'durmixrespscale',
                           ]

    if not any(map(args.method.__contains__, methods_implemented)):
        return data, target_ohe, [], None
    
    if 'durmixrespscale' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        if '(sameCVD)' in args.method:
            mix_indices = get_same_cvd_mix_indices(wav, random_seed)
        # first durratiomuxip
        lam = get_lambda(alpha=1, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_out = lams[:, None, None]
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lams_out[0], args.method, random_seed)
            data_new[i, :, :] = d_new
        # now resporatory scale also
        respscale_rate_min = 12/60
        respscale_rate_max = 20/60
        if len(args.method.split('durmixrespscale(')) > 1:
            respscale_rate_min = float(args.method.split('durmixrespscale(')[1].split(',')[0])/60
            respscale_rate_max = int(args.method.split(',')[1].split(')')[0])/60
        respiration_rate = random.Random(random_seed).uniform(respscale_rate_min, respscale_rate_max)
        random_phase = random.Random(random_seed).uniform(0, 2 * np.pi)
        t = np.linspace(0, sig_len/args.sample_rate, sig_len)
        sinusoid = np.sin(2 * np.pi * respiration_rate * t + random_phase)
        sinusoid = sinusoid.reshape(1, 1, -1) # shape [1, 1, sig_len]
        sinusoid = np.tile(sinusoid, (size, num_channels, 1)) # same shape as batch
        sinusoid = torch.from_numpy(sinusoid).to(device)
        # perform respiratory scaling
        data_new = data_new * sinusoid
        data_new = data_new.float() # the multiplication makes data Double format instead of Float
        return data_new, target_ohe, [], None
    
    if 'respiratoryscale' in args.method:
        aug_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            aug_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= aug_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        respscale_rate_min = 12/60
        respscale_rate_max = 20/60
        if len(args.method.split('respiratoryscale(')) > 1:
            respscale_rate_min = float(args.method.split('respiratoryscale(')[1].split(',')[0])/60
            respscale_rate_max = int(args.method.split(',')[1].split(')')[0])/60
        respiration_rate = random.Random(random_seed).uniform(respscale_rate_min, respscale_rate_max)
        random_phase = random.Random(random_seed).uniform(0, 2 * np.pi)
        t = np.linspace(0, sig_len/args.sample_rate, sig_len)
        sinusoid = np.sin(2 * np.pi * respiration_rate * t + random_phase)
        sinusoid = sinusoid.reshape(1, 1, -1) # shape [1, 1, sig_len]
        sinusoid = np.tile(sinusoid, (size, num_channels, 1)) # same shape as batch
        sinusoid = torch.from_numpy(sinusoid).to(device)
        # perform respiratory scaling
        data = data * sinusoid
        data = data.float() # the multiplication makes data Double format instead of Float
        return data, target_ohe, [], None

    
    if 'timemask' in args.method:
        cutout_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutout_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutout_proba:
            return data, target_ohe, [], None
        mask_region_max = 0.2
        if len(args.method.split('timemask(')) > 1:
            mask_region_max = float(args.method.split('timemask(')[1].split(')')[0])
            mask_region_max = min(max(mask_region_max, 0), 1)
        mask_region_gap = random.Random(step_counter.count+131071).uniform(0, mask_region_max)
        mask_frac1 = random.Random(step_counter.count+13119).uniform(0, 1-mask_region_gap)
        mask_frac2 = mask_frac1+mask_region_gap
        for i, (d, f) in enumerate(zip(data, frames)):
            beat_len = f[-1].numpy()
            bb = [int(mask_frac1*beat_len), int(mask_frac2*beat_len)]
            d[:, bb[0]:bb[1]] = 0
        return data, target_ohe, [], None

    if 'mixup' in args.method and 'latentmixup' not in args.method and 'durratiomixup' not in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        if '(same)' in args.method:
            # Mixup data same label
            mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
            lam = get_lambda(alpha=1, random_seed=random_seed)
            lams = np.ones(size)*lam
            lams_np = np.array(lams).astype('float32')
            lams = torch.from_numpy(lams_np).to(device)
            lams_out = lams[:, None, None]
            data = data*lams_out + data[mix_indices]*(1-lams_out)
            return data, target_ohe, mix_indices, None
        elif '(mix)' in args.method:
            # Mixup data with mixed labels
            mix_indices = random.Random(random_seed).sample(list(np.arange(0, size, 1)), size)
            lam = get_lambda(alpha=1, random_seed=random_seed)
            lams = np.ones(size)*lam
            lams_np = np.array(lams).astype('float32')
            lams = torch.from_numpy(lams_np).to(device)
            lams_out = lams[:, None, None]
            lams_target = lams[:, None]
            data = data*lams_out + data[mix_indices]*(1-lams_out)
            target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
            return data, target_ohe, mix_indices, None
        
    if 'durmixmagwarp' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        if '(sameCVD)' in args.method: # same CVD
            mix_indices = get_same_cvd_mix_indices(wav, random_seed)
        if '(samePCG)' in args.method: # same patient
            mix_indices = get_same_wav_mix_indices(wav, random_seed)
        if '(sameDataset)' in args.method: # same recording conditions
            mix_indices = get_same_dataset_mix_indices(target_ohe, wav, random_seed)
        if '(mixAll)' in args.method: # disregard all constraints
            mix_indices = random.Random(random_seed).sample(list(np.arange(0, size, 1)), size)
        if '(closestbins=' in args.method: # only mix closest in latent space
            bin_num = int(args.method.split('(closestbins=')[1].split(')')[0])
            mix_indices = get_same_label_closestbins(target_ohe, data.clone(), bin_num, random_seed, RESULTS_ARGS, step_counter.count)
        if '(closestknn=' in args.method: # only mix closest in latent space
            k_num = int(args.method.split('(closestknn=')[1].split(')')[0])
            mix_indices = get_same_label_closestknn(target_ohe, data.clone(), k_num, random_seed, RESULTS_ARGS, step_counter.count, args.batch_size)
        if '(closestknn=' in args.method: # only mix closest in latent space
            k_num = int(args.method.split('(closestknn=')[1].split(')')[0])
            mix_indices = get_same_label_closestknn(target_ohe, data.clone(), k_num, random_seed, RESULTS_ARGS, step_counter.count, args.batch_size)
        # first durratiomuxip
        alpha=1
        if len(args.method.split('(alpha=')) > 1:
            alpha = float(args.method.split('(alpha=')[1].split(')durmixmagwarp')[0])
        lam = get_lambda(alpha=alpha, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_out = lams[:, None, None]
        if '(salopt' in args.method:
            gauss_k_n = 101
            saliency_maps = saliency.get_saliency_maps(args, device, data.clone(), target_ohe.clone(), frames.clone(), dim=1, gauss_k_n=gauss_k_n)
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            if '(salopt' in args.method:
                s1 = saliency_maps[i]
                s2 = saliency_maps[mix_indices][i]
                d_new = mixup_keepdur_multidim_tensors_salopt(d1, d2, f1, f2, s1, s2, lams_out[0], args.method, random_seed)
            else:    
                d_new = mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lams_out[0], args.method, random_seed)
            data_new[i, :, :] = d_new
        if '(mixAll)' in args.method: # disregard all constraints
            lams_target = lams[:, None]
            target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
        # now mag warping also
        magwarp_sigma = 0.2
        magwarp_knot = 4
        if len(args.method.split('durmixmagwarp(')) > 1:
            magwarp_sigma = float(args.method.split('durmixmagwarp(')[1].split(',')[0])
            magwarp_knot = int(args.method.split(',')[1].split(')')[0])
        data_new = data_new.detach().cpu().numpy()
        data_new = np.transpose(data_new, (0, 2, 1))   # mag warp accepts (batch_size, sig_len, num_channeles)
        data_new = magnitude_warp(data_new, magwarp_sigma, magwarp_knot)
        data_new = np.transpose(data_new, (0, 2, 1)) # transpose back to (batch_size, num_channels, sig_len)
        data_new = torch.from_numpy(data_new).to(device)
        return data_new, target_ohe, mix_indices, None
        
    if 'durratiomixup' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        if '(sameCVD)' in args.method: # same CVD
            mix_indices = get_same_cvd_mix_indices(wav, random_seed)
        if '(samePCG)' in args.method: # same patient
            mix_indices = get_same_wav_mix_indices(wav, random_seed)
        if '(sameDataset)' in args.method: # same recording conditions
            mix_indices = get_same_dataset_mix_indices(target_ohe, wav, random_seed)
        if '(mixAll)' in args.method: # disregard all constraints
            mix_indices = random.Random(random_seed).sample(list(np.arange(0, size, 1)), size)
        if '(closestbins=' in args.method: # only mix closest in latent space
            bin_num = int(args.method.split('(closestbins=')[1].split(')')[0])
            mix_indices = get_same_label_closestbins(target_ohe, data.clone(), bin_num, random_seed, RESULTS_ARGS, step_counter.count)
        if '(closestknn=' in args.method: # only mix closest in latent space
            k_num = int(args.method.split('(closestknn=')[1].split(')')[0])
            mix_indices = get_same_label_closestknn(target_ohe, data.clone(), k_num, random_seed, RESULTS_ARGS, step_counter.count, args.batch_size)
        alpha=1
        if len(args.method.split('(alpha=')) > 1:
            alpha = float(args.method.split('(alpha=')[1].split(')durratiomixup')[0])
        lam = get_lambda(alpha=alpha, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_out = lams[:, None, None]
        if '(salopt' in args.method:
            gauss_k_n = 101
            saliency_maps = saliency.get_saliency_maps(args, device, data.clone(), target_ohe.clone(), frames.clone(), dim=1, gauss_k_n=gauss_k_n)
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            if '(salopt' in args.method:
                s1 = saliency_maps[i]
                s2 = saliency_maps[mix_indices][i]
                d_new = mixup_keepdur_multidim_tensors_salopt(d1, d2, f1, f2, s1, s2, lams_out[0], args.method, random_seed)
            else:    
                d_new = mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lams_out[0], args.method, random_seed)
            data_new[i, :, :] = d_new
        if '(mixAll)' in args.method: # disregard all constraints
            lams_target = lams[:, None]
            target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
        return data_new, target_ohe, mix_indices, None

    if 'wav-durratiocutmix' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices =  get_same_wav_mix_indices(wav, random_seed)          
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = cutmix_keepdur_multidim_tensors(d1, d2, f1, f2, args.method, random_seed)
            data_new[i, :, :] = d_new
        return data_new, target_ohe, mix_indices, None
    
    if 'timewarp' in args.method:
        aug_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            aug_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= aug_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        timewarp_sigma = 0.05
        timewarp_knot = 2
        if len(args.method.split('timewarp(')) > 1:
            timewarp_sigma = float(args.method.split('timewarp(')[1].split(',')[0])
            timewarp_knot = int(args.method.split(',')[1].split(')')[0])
        data = data.detach().cpu().numpy()
        data = np.transpose(data, (0, 2, 1))   # time warp accepts (batch_size, sig_len, num_channeles)
        data = time_warp(data, timewarp_sigma, timewarp_knot)
        data = np.transpose(data, (0, 2, 1)) # transpose back to (batch_size, num_channels, sig_len)
        data = torch.from_numpy(data).to(device)
        return data, target_ohe, [], None

    if 'magnitudewarp' in args.method:
        aug_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            aug_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= aug_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        magwarp_sigma = 0.2
        magwarp_knot = 4
        if len(args.method.split('magnitudewarp(')) > 1:
            magwarp_sigma = float(args.method.split('magnitudewarp(')[1].split(',')[0])
            magwarp_knot = int(args.method.split(',')[1].split(')')[0])
        data = data.detach().cpu().numpy()
        data = np.transpose(data, (0, 2, 1))   # mag warp accepts (batch_size, sig_len, num_channeles)
        data = magnitude_warp(data, magwarp_sigma, magwarp_knot)
        data = np.transpose(data, (0, 2, 1)) # transpose back to (batch_size, num_channels, sig_len)
        data = torch.from_numpy(data).to(device)
        return data, target_ohe, [], None
    
    if 'gaussiannoise' in args.method:
        aug_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            aug_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= aug_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        snr_min = 25
        snr_max = 40
        if len(args.method.split('gaussiannoise(')) > 1:
            snr_min = float(args.method.split('gaussiannoise(')[1].split(',')[0])
            snr_max = int(args.method.split(',')[1].split(')')[0])
        transform = AddGaussianSNR(min_snr_in_db=snr_min,max_snr_in_db=snr_max,p=1.0)
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d, f) in enumerate(zip(data, frames.numpy())):
            d = d.detach().cpu().numpy()
            max_sig_val = np.max(d)
            d = d/max_sig_val # audimentatons expect inputs between -1 and 1
            d_new = transform(samples=d, sample_rate=args.sample_rate)
            d_new = d_new*max_sig_val # resize to the original size
            d_new = torch.from_numpy(d_new).to(device)
            d_new[:, f[-1]:] = 0 # everything after heart beat must be 0
            data_new[i, :, :] = d_new
        return data_new, target_ohe, [], None
    
    if '(UMC-subset)durratiocutmix' in args.method and '(plus)' not in args.method and '(plusplus)' not in args.method:
        # only mix instances from the same subset (old/new)
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        #print(f'Method "labelcutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_umc_subset_mix_indices(target_ohe, wav, random_seed)            
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = cutmix_keepdur_multidim_tensors(d1, d2, f1, f2, args.method, random_seed)
            data_new[i, :, :] = d_new
        return data_new, target_ohe, mix_indices, None

    if 'durratiocutmix' in args.method and '(plus)' not in args.method and '(plusplus)' not in args.method and '(UMC' not in args.method and 'wav-durratiocutmix' not in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        #print(f'Method "labelcutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)            
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = cutmix_keepdur_multidim_tensors(d1, d2, f1, f2, args.method, random_seed)
            data_new[i, :, :] = d_new
        return data_new, target_ohe, mix_indices, None
    
    if 'lengthcutmix' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        #print(f'Method "labelcutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]

        mix_indices = get_same_length_mix_indices(target_ohe, frames, random_seed, args.batch_size, args.method)
        # always cut in the middle
        cut = 2
        if '(rand)' in args.method:
            cut = random.Random(random_seed).randint(1, 3)
        if 'cutout' in args.method:
            cut_frac = sorted([random.Random(step_counter.count+i*131071).uniform(0, 1) for i in range(2)])
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        frames_new = []
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new, f_new = cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, sig_len, args.method, device)
            if 'cutout' in args.method:
                bb = [int(cf*f_new[-1]) for cf in cut_frac]
                d_new[:, bb[0]:bb[1]] = 0
            data_new[i, :, :] = d_new
            frames_new.append(f_new)
        return data_new, target_ohe, mix_indices, cut
    
    if 'datasetcutmix' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        #print(f'Method "labelcutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_dataset_mix_indices(target_ohe, wav, random_seed)
        # always cut in the middle
        cut = 2
        if '(rand)' in args.method:
            cut = random.Random(random_seed).randint(1, 3)
        if 'cutout' in args.method:
            cut_frac = sorted([random.Random(step_counter.count+i*131071).uniform(0, 1) for i in range(2)])
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        frames_new = []
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new, f_new = cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, sig_len, args.method, device)
            if 'cutout' in args.method:
                bb = [int(cf*f_new[-1]) for cf in cut_frac]
                d_new[:, bb[0]:bb[1]] = 0
            data_new[i, :, :] = d_new
            frames_new.append(f_new)
        return data_new, target_ohe, mix_indices, cut
    
    if 'wavcutmix' in args.method and 'durratiowavcutmix' not in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        #print(f'Method "labelcutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_wav_mix_indices(wav, random_seed)
        # always cut in the middle
        cut = 2
        if '(rand)' in args.method:
            cut = random.Random(random_seed).randint(1, 3)
        if 'cutout' in args.method:
            cut_frac = sorted([random.Random(step_counter.count+i*131071).uniform(0, 1) for i in range(2)])
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        frames_new = []
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new, f_new = cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, sig_len, args.method, device)
            if 'cutout' in args.method:
                bb = [int(cf*f_new[-1]) for cf in cut_frac]
                d_new[:, bb[0]:bb[1]] = 0
            data_new[i, :, :] = d_new
            frames_new.append(f_new)
        return data_new, target_ohe, mix_indices, cut
    
    if 'lc-nointrusion' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe
        print(f'Method "labelcutmix-nointrusion" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
        label_indices1 = [[] for i in range(args.num_classes)]
        for i in range(args.num_classes):
            label_indices1[i] = [idx for idx, t in enumerate(target) if t == i]
        label_indices2 = copy.deepcopy(label_indices1)
        multiplication_factor = 4
        label_indices_0_len = len(label_indices1[0])
        label_indices_1_len = len(label_indices1[1])
        for i in range(args.num_classes):
            label_indices1[i] = random.Random(random_seed*131071+178397654).choices(label_indices1[i], k=len(label_indices1[i])*multiplication_factor)
            label_indices2[i] = random.Random(random_seed*8191+99999).choices(label_indices2[i], k=len(label_indices1[i])*multiplication_factor)
        label_indices1_flat = [item for sublist in label_indices1 for item in sublist]
        label_indices2_flat = [item for sublist in label_indices2 for item in sublist]
        label_indices_both = list(zip(label_indices1_flat, label_indices2_flat))
        random.Random(random_seed).shuffle(label_indices_both)
        mix_indices1, mix_indices2 = zip(*label_indices_both)
        mix_indices1, mix_indices2 = np.array(mix_indices1), np.array(mix_indices2)
        # always cut in the middle
        cut = 2
        if '(rand)' in args.method:
            cut = random.Random(step_counter.count*131071).randint(1, 3)
        if 'cutout' in args.method:
            cut_frac = sorted([random.Random(step_counter.count+i*131071).uniform(0, 1) for i in range(2)])
        data_new = torch.zeros((size*multiplication_factor, num_channels, sig_len)).to(device)
        frames_new = []
        for i, (d1, f1, d2, f2) in enumerate(zip(data[mix_indices1], frames[mix_indices1].numpy(), data[mix_indices2], frames[mix_indices2].numpy())):
            d_new, f_new = cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, sig_len, args.method, device)
            if 'cutout' in args.method:
                bb = [int(cf*f_new[-1]) for cf in cut_frac]
                d_new[:, bb[0]:bb[1]] = 0
            data_new[i, :, :] = d_new
            frames_new.append(f_new)
        frames_new = torch.tensor(np.array(frames_new))
        target_ohe = target_ohe[mix_indices1]
        
        # get predictions of the synthetic instances
        _, _, out, _, _ = saliency.saliency_map(data_new.clone(), target_ohe.clone(), frames_new.clone(), copy.deepcopy(model), device)
        # only keep use instances with the lowest losses
        _, _, _, losses_1d  = train_model.custom_loss(out, target_ohe)
        target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy()
        new_indices_0 = np.array([idx for idx, t in enumerate(target) if t==0])
        new_indices_1 = np.array([idx for idx, t in enumerate(target) if t==1])
        losses_0 = losses_1d[new_indices_0]
        losses_1 = losses_1d[new_indices_1]
        new_indices_0_sort = np.array([x for _,x in sorted(zip(losses_0, new_indices_0))])
        new_indices_1_sort = np.array([x for _,x in sorted(zip(losses_1, new_indices_1))])
        new_indices_0_sort = new_indices_0_sort[:label_indices_0_len]
        new_indices_1_sort = new_indices_1_sort[:label_indices_1_len]
        new_indices_sort = list(new_indices_0_sort) + list(new_indices_1_sort)
        new_indices_sort = np.array(sorted(new_indices_sort))
        # select the new new instances
        data_new = data_new[new_indices_sort]
        target_ohe = target_ohe[new_indices_sort]
        frames_new = torch.tensor(frames_new.numpy()[new_indices_sort])
        
        return data_new, target_ohe
    
    if 'labelcutmix' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        """ if step_counter.count < args.cutmix_step:
            return data, target_ohe, [], None """
        if r >= cutmix_proba:
            return data, target_ohe, [], None
        #print(f'Method "labelcutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        # always cut in the middle
        cut = 2
        if '(rand)' in args.method:
            cut = random.Random(step_counter.count*131071).randint(1, 3)
        if 'cutout' in args.method:
            cut_frac = sorted([random.Random(step_counter.count+i*131071).uniform(0, 1) for i in range(2)])
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        frames_new = []
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new, f_new = cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, sig_len, args.method, device)
            if 'cutout' in args.method:
                bb = [int(cf*f_new[-1]) for cf in cut_frac]
                d_new[:, bb[0]:bb[1]] = 0
            data_new[i, :, :] = d_new
            frames_new.append(f_new)
        return data_new, target_ohe, mix_indices, cut
    
    if 'swapsysdia' in args.method:
        swap_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            swap_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= swap_proba:
            return data, target_ohe
        print(f'Method "swapsysdia" runs with probability {swap_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = np.arange(0, size, 1)
        mix_indices = random.Random(random_seed).sample(list(mix_indices), size)
        lams = []
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            # swap systole and diastole of d1 with those of d2
            d1_s1_len = f1[1]-f1[0]
            d1_s2_len = f1[3]-f1[2]
            d2_sys_len = f2[2]-f2[1]
            d2_dia_len = f2[4]-f2[3]
            d_new = torch.zeros((num_channels, sig_len*2)).to(device)
            d_new[:, 0:d1_s1_len] = d1[:, 0:f1[1]]
            d_new[:, d1_s1_len:d1_s1_len+d2_sys_len] = d2[:, f2[1]:f2[2]]
            d_new[:, d1_s1_len+d2_sys_len:d1_s1_len+d2_sys_len+d1_s2_len] = d1[:, f1[2]:f1[3]]
            d_new[:, d1_s1_len+d2_sys_len+d1_s2_len:d1_s1_len+d2_sys_len+d1_s2_len+d2_dia_len] = d2[:, f2[3]:f2[4]]
            data_new[i, :, :] = d_new[:, 0:sig_len]
            lam = (d1_s1_len+d1_s2_len)/(d1_s1_len+d2_sys_len+d1_s2_len+d2_dia_len)
            lams.append(lam)

        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_target = lams[:, None]
        target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
        return data_new, target_ohe

    if 'cont-cutmix' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe
        print(f'Method "cont-cutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = np.arange(0, size, 1)
        mix_indices = random.Random(random_seed).sample(list(mix_indices), size)
        cut_frac = sorted([random.Random(step_counter.count+i*131071).uniform(0, 1) for i in range(2)])
        print(f'{cut_frac=}')
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        target_ohe_new = torch.empty((size, args.num_classes)).to(device)
        for i, (d1, f1, t1, d2, f2, t2) in enumerate(zip(data, frames.numpy(), target_ohe, data[mix_indices], frames[mix_indices].numpy(), target_ohe[mix_indices])):
            # perform cutout
            d1_len = f1[-1]
            d2_len = f2[-1]
            bb1 = [int(cf*d1_len) for cf in cut_frac]
            bb2 = [int(cf*d2_len) for cf in cut_frac]
            data_new[i, :, 0:bb1[0]] = d1[:, 0:bb1[0]]
            data_new[i, :, bb1[0]:bb1[0]+bb2[1]-bb2[0]] = d2[:, bb2[0]:bb2[1]]
            data_new[i, :, bb1[0]+bb2[1]-bb2[0]:bb1[0]+bb2[1]-bb2[0]+d1_len-bb1[1]] = d1[:, bb1[1]:d1_len]
            if np.array_equal(t1.detach().cpu().numpy(), np.array([0, 1])) or np.array_equal(t2.detach().cpu().numpy(), np.array([0, 1])):
                target_ohe_new[i] = torch.from_numpy(np.array([0, 1])).to(device)
            else:
                target_ohe_new[i] = torch.from_numpy(np.array([1, 0])).to(device)
        lam = 1-(cut_frac[1]-cut_frac[0])
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_target = lams[:, None]
        target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
        return data_new, target_ohe
    
    if 'saliency-cutmix' in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe
        print(f'Method "saliency-cutmix" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = np.arange(0, size, 1)
        mix_indices = random.Random(random_seed).sample(list(mix_indices), size)
        # Get saliency maps
        saliency_map, saliency_states_map, _, bin_values, bin_frames = saliency.saliency_map(data.clone(), target_ohe.clone(), frames.clone(), copy.deepcopy(model), device)
        # Saliency cutmix
        quasi_lam = get_lambda(alpha=1.0, random_seed = random_seed)
        print(f'{quasi_lam=}')
        lams = []
        frames_new = []
        d_new_lens = []
        data_new = torch.zeros((size, num_channels, sig_len)).to(device)
        for i, (d1, d2) in enumerate(zip(data, data[mix_indices])):
            binval1 = bin_values[i]
            binval2 = bin_values[mix_indices[i]]
            binfr1 = bin_frames[i]
            binfr2 = bin_frames[mix_indices[i]]
            d1_np = d1.detach().cpu().numpy()
            d2_np = d2.detach().cpu().numpy()
            d_new = np.array([], dtype=np.int64).reshape(num_channels,0)
            binval2_threshold = sorted(binval2, reverse=True)[int(quasi_lam*len(binval2))]
            if i ==0:
                print(f'{binval2_threshold=}')
            f_new = [0]
            samples_per_label = [0, 0]
            for j, (bv1, bv2) in enumerate(zip(binval1, binval2)):
                if j in [0, 5]: # if s1 or s2
                    if bv1 > bv2:
                        d_new = np.append(d_new, d1_np[:, binfr1[j]:binfr1[j+1]], axis=1)
                        samples_per_label[0] += binfr1[j+1]-binfr1[j]
                        f_new.append(f_new[j]+binfr1[j+1]-binfr1[j])
                    else:
                        d_new = np.append(d_new, d2_np[:, binfr2[j]:binfr2[j+1]], axis=1)
                        samples_per_label[1] += binfr2[j+1]-binfr2[j]
                        f_new.append(f_new[j]+binfr2[j+1]-binfr2[j])
                else:
                    if bv2 >= binval2_threshold:
                        d_new = np.append(d_new, d2_np[:, binfr2[j]:binfr2[j+1]], axis=1)
                        samples_per_label[1] += binfr2[j+1]-binfr2[j]
                        f_new.append(f_new[j]+binfr2[j+1]-binfr2[j])
                    else:
                        d_new = np.append(d_new, d1_np[:, binfr1[j]:binfr1[j+1]], axis=1)
                        samples_per_label[0] += binfr1[j+1]-binfr1[j]
                        f_new.append(f_new[j]+binfr1[j+1]-binfr1[j])
            # calculate length of new data instance           
            d_new_len = min(sig_len, d_new.shape[1])
            d_new_lens.append(d_new_len)
            # turn to tensor
            d_new_tensor = torch.zeros((num_channels, sig_len)).to(device)
            d_new_tensor[:, 0:d_new_len] = torch.tensor(d_new).to(device)
            # add to the new data set batch
            data_new[i, :, :] =  d_new_tensor
            # calculate lambda
            lam = samples_per_label[0] / (samples_per_label[0]+samples_per_label[1])
            lams.append(lam)
            frames_new.append(f_new)
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_target = lams[:, None]

        # mixup target
        target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
        return data_new, target_ohe
        
    if 'latentmixup' in args.method:
        mixup_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            mixup_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= mixup_proba:
            return data, target_ohe, [], None
        size = data.shape[0]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        # Get latent space features
        if args.model == 'FCN':
            args.depth = 4 
        elif args.model == 'Potes':
            max_model_depth = 1
        elif args.model == 'ResCNN':
            args.depth = 5
        elif args.model == 'resnet9':
            max_model_depth = 3
        elif args.model == 'Singstad':
            max_model_depth = 3
        args.depth = random.Random(random_seed).randint(1, max_model_depth) # always map data to hidden representation at some layer
        data = model(data, depth=args.depth, pass_part='first')
        # Mixup data
        lam = get_lambda(alpha=1, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        if len(data.shape) == 2:
            lams_out = lams[:, None]
        elif len(data.shape) == 3:
            lams_out = lams[:, None, None]  
        data = data*lams_out + data[mix_indices]*(1-lams_out)
        return data, target_ohe, mix_indices, None
            
    elif 'cutmix' in args.method and 'saliency' not in args.method and 'label' not in args.method:
        cutmix_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutmix_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutmix_proba:
            return data, target_ohe
        #print(f'Method "{args.method}" runs with probability {cutmix_proba}')
        size = data.shape[0]
        num_channels = data.shape[1]
        sig_len = data.shape[2]
        mix_indices = np.arange(0, size, 1)
        mix_indices = random.Random(random_seed).sample(list(mix_indices), size)
        if 'manifold' in args.method:
            # Get data from one of the inner layers of the model
            model.eval()
            with torch.no_grad():
                if args.model == 'FCN':
                    max_depth = 3
                r_depth = random.Random(random_seed).randint(0, max_depth)
                args.depth = r_depth
                data = model(data, depth=args.depth, pass_part='first')
                print(f'{args.depth=}')
                print(f'{data.shape=}')
            model.train()
        lams = []
        if '(ch)' in args.method:
            cuts = [random.Random(step_counter.count*131071+c*524287).randint(1, 3) for c in range(num_channels)]
            for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
                lams_sig = []
                for c in range(num_channels):
                    cut = cuts[c]
                    last_f1_paste_frame = min(f1[cut]+f2[-1]-f2[cut], sig_len)
                    d1[c, f1[cut]:last_f1_paste_frame] = d2[c, f2[cut]:f2[cut]+last_f1_paste_frame-f1[cut]]
                    d1[c, last_f1_paste_frame:] = 0
                    lam = f1[cut]/last_f1_paste_frame
                    lams_sig.append(lam)
                lams.append(np.mean(lams_sig))
        else:
            cut = random.Random(step_counter.count*131071).randint(1, 3)
            target_ohe_new = torch.empty((size, args.num_classes)).to(device)
            data_new = torch.zeros((size, num_channels, sig_len)).to(device)
            for i, (d1, f1, t1, d2, f2, t2) in enumerate(zip(data, frames.numpy(), target_ohe, data[mix_indices], frames[mix_indices].numpy(), target_ohe[mix_indices])):
                last_f1_paste_frame = min(f1[cut]+f2[-1]-f2[cut], sig_len)
                data_new[i, :, 0:f1[cut]] = d1[:, 0:f1[cut]]
                data_new[i, :, f1[cut]:last_f1_paste_frame] = d2[:, f2[cut]:f2[cut]+last_f1_paste_frame-f1[cut]]
                if np.array_equal(t1.detach().cpu().numpy(), np.array([0, 1])) or np.array_equal(t2.detach().cpu().numpy(), np.array([0, 1])):
                    target_ohe_new[i] = torch.from_numpy(np.array([0, 1])).to(device)
                else:
                    target_ohe_new[i] = torch.from_numpy(np.array([1, 0])).to(device)
                lam = f1[cut]/last_f1_paste_frame
                lams.append(lam)
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_target = lams[:, None]
        target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
        
        return data_new, target_ohe
    
    if 'cutout' in args.method and 'saliency' not in args.method:
        cutout_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            cutout_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= cutout_proba:
            return data, target_ohe, [], None
        #print(f'Method "{args.method}" runs with probability {cutout_proba}')
        if 'manifold' in args.method:
            # Get data from one of the inner layers of the model
            model.eval()
            with torch.no_grad():
                if args.model == 'FCN':
                    max_depth = 3
                r_depth = random.Random(random_seed).randint(0, max_depth)
                args.depth = r_depth
                data = model(data, depth=args.depth, pass_part='first')
                print(f'{args.depth=}')
                print(f'{data.shape=}')
            model.train()
        if '(ch)' in args.method:
            num_channels = data.shape[1]
            cut_fracs = [sorted([random.Random(step_counter.count+i*131071+c*524287).uniform(0, 1) for i in range(2)]) for c in range(num_channels)]
            print(f'{cut_fracs=}')
            for i, (d, f) in enumerate(zip(data, frames)):
                # perform cutout
                beat_len = f[-1].numpy()
                for c in range(num_channels):
                    cut_frac = cut_fracs[c]
                    bb = [int(cf*beat_len) for cf in cut_frac]
                    d[c, bb[0]:bb[1]] = 0
        else:
            #cut_frac = sorted([random.Random(step_counter.count+i*131071).uniform(0, 1) for i in range(2)])
            cutout_region_max = 0.05
            cutout_region_gap = random.Random(step_counter.count+131071).uniform(0, cutout_region_max)
            cut_frac1 = random.Random(step_counter.count+13119).uniform(0, 1-cutout_region_gap)
            cut_frac2 = cut_frac1+cutout_region_gap
            #print(f'{cut_frac=}')
            for i, (d, f) in enumerate(zip(data, frames)):
                # perform cutout
                beat_len = f[-1].numpy()
                #bb = [int(cf*beat_len) for cf in cut_frac]
                bb = [int(cut_frac1*beat_len), int(cut_frac2*beat_len)]
                d[:, bb[0]:bb[1]] = 0
        #utils.show_sig(data[0], frames[0])
        return data, target_ohe, [], None
    
    if args.method == 's1s2mask':
        mask_proba = 1.0 #TWEAK
        method_str_split = args.method.split('+')
        if len(method_str_split) > 1:
            mask_proba = float(method_str_split[-1])
        random_seed = step_counter.count
        r = random.Random(random_seed).uniform(0, 1)
        if r >= mask_proba:
            return data, target_ohe
        print(f'Method "s1s2mask" runs with probability {mask_proba}')
        for i, (d, f) in enumerate(zip(data, frames)):
                # perform cutout
                d[:, f[0]:f[1]] = 0 # S1 region
                d[:, f[2]:f[3]] = 0 # S2 region
        #utils.show_sig(data[0], frames=frames[0])
        return data, target_ohe