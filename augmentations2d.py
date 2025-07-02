from tkinter import wantobjects
import numpy as np
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

import saliency
import train_model
import utils

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
                     
def cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, spec_dim1, spec_dim2, method, device):
    # check how long the connected instance will be and limit it to spec_dim2
    d1_cut_len = f1[cut]
    d2_cut_len = f2[-1]-f2[cut]
    d_connected_len = d1_cut_len + d2_cut_len
    # define a new instance
    d_new = torch.zeros((num_channels, spec_dim1, spec_dim2)).to(device)
    # connect the first part of d1 and second part of d2
    last_f1_paste_frame = min(d_connected_len, spec_dim2) 
    d_new[:, :, 0:f1[cut]] = d1[:, :, 0:f1[cut]]
    d_new[:, :, f1[cut]:last_f1_paste_frame] = d2[:, :, f2[cut]:f2[cut]+last_f1_paste_frame-f1[cut]]
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
            #new_s1 = np.concatenate((s1[:displacement], np.maximum(s1[displacement:displacement+len_s2], current_s2_shifted[displacement:displacement+len_s2]), s1[displacement+len_s2:]))
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
            #new_s1 = np.concatenate((s1[:displacement], (s1[displacement:displacement+len_s2]*lam+current_s2_shifted[displacement:displacement+len_s2]*(1-lam))[0], s1[displacement+len_s2:]))
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
    #""" 
    len_s1_d1 = f1[1]-f1[0]
    len_s1_d2 = f2[1]-f2[0]
    sal1_s1 = sal1[f1[0]:f1[1]]
    sal2_s1 = sal2[f2[0]:f2[1]]
    if len_s1_d1 == len_s1_d2:
        disp_s1 = 0
        d_new[ :, :, f1[0]:f1[1]] = d_new[ :, :, f1[0]:f1[1]]*lam + d2[ :, :, f2[0]:f2[1]]*(1-lam)
        sal_new[f1[0]:f1[1]] = sal_new[f1[0]:f1[1]]*lam_np + sal2[f2[0]:f2[1]]*(1-lam_np)
    elif len_s1_d1 > len_s1_d2:
        disp_s1 = optimal_displacement_approach(sal1_s1, sal2_s1, lam_np)
        d_new[ :, :, f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2] = d_new[ :, :, f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2]*lam + d2[ :, :, f2[0]:f2[1]]*(1-lam)
        sal_new[f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2] = sal_new[f1[0]+disp_s1:f1[0]+disp_s1+len_s1_d2]*lam_np + sal2[f2[0]:f2[1]]*(1-lam_np)
    else: #(len_s1_d1 < len_s1_d2)
        disp_s1 = optimal_displacement_approach(sal1_s1, sal2_s1, lam_np)
        d_new[ :, :, f1[0]:f1[1]] = d_new[ :, :, f1[0]:f1[1]]*lam + d2[ :, :, f2[0]+disp_s1:f2[0]+disp_s1+len_s1_d1]*(1-lam)
        sal_new[f1[0]:f1[1]] = sal_new[f1[0]:f1[1]]*lam_np + sal2[f2[0]+disp_s1:f2[0]+disp_s1+len_s1_d1]*(1-lam_np)
    # mix systole of d1 with systole of d2
    len_sys_d1 = f1[2]-f1[1]
    len_sys_d2 = f2[2]-f2[1]
    sal1_sys = sal1[f1[1]:f1[2]]
    sal2_sys = sal2[f2[1]:f2[2]]
    if len_sys_d1 == len_sys_d2:
        disp_sys = 0
        d_new[ :, :, f1[1]:f1[2]] = d_new[ :, :, f1[1]:f1[2]]*lam + d2[ :, :, f2[1]:f2[2]]*(1-lam)
        sal_new[f1[1]:f1[2]] = sal_new[f1[1]:f1[2]]*lam_np + sal2[f2[1]:f2[2]]*(1-lam_np)
    elif len_sys_d1 > len_sys_d2:
        disp_sys = optimal_displacement_approach(sal1_sys, sal2_sys, lam_np)
        d_new[ :, :, f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2] = d_new[ :, :, f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2]*lam + d2[ :, :, f2[1]:f2[2]]*(1-lam)
        sal_new[f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2] = sal_new[f1[1]+disp_sys:f1[1]+disp_sys+len_sys_d2]*lam_np + sal2[f2[1]:f2[2]]*(1-lam_np)
    else: #(len_sys_d1 < len_sys_d2)
        disp_sys = optimal_displacement_approach(sal1_sys, sal2_sys, lam_np)
        d_new[ :, :, f1[1]:f1[2]] = d_new[ :, :, f1[1]:f1[2]]*lam + d2[ :, :, f2[1]+disp_sys:f2[1]+disp_sys+len_sys_d1]*(1-lam)
        sal_new[f1[1]:f1[2]] = sal_new[f1[1]:f1[2]]*lam_np + sal2[f2[1]+disp_sys:f2[1]+disp_sys+len_sys_d1]*(1-lam_np)
    # mix S2 of d1 with S2 of d2
    len_s2_d1 = f1[3]-f1[2]
    len_s2_d2 = f2[3]-f2[2]
    sal1_s2 = sal1[f1[2]:f1[3]]
    sal2_s2 = sal2[f2[2]:f2[3]]
    if len_s2_d1 == len_s2_d2:
        disp_s2 = 0
        d_new[ :, :, f1[2]:f1[3]] = d_new[ :, :, f1[2]:f1[3]]*lam + d2[ :, :, f2[2]:f2[3]]*(1-lam)
        sal_new[f1[2]:f1[3]] = sal_new[f1[2]:f1[3]]*lam_np + sal2[f2[2]:f2[3]]*(1-lam_np)
    elif len_s2_d1 > len_s2_d2:
        disp_s2 = optimal_displacement_approach(sal1_s2, sal2_s2, lam_np)
        d_new[ :, :, f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2] = d_new[ :, :, f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2]*lam + d2[ :, :, f2[2]:f2[3]]*(1-lam)
        sal_new[f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2] = sal_new[f1[2]+disp_s2:f1[2]+disp_s2+len_s2_d2]*lam_np + sal2[f2[2]:f2[3]]*(1-lam_np)
    else: #(len_s2_d1 < len_s2_d2)
        disp_s2 = optimal_displacement_approach(sal1_s2, sal2_s2, lam_np)
        d_new[ :, :, f1[2]:f1[3]] = d_new[ :, :, f1[2]:f1[3]]*lam + d2[ :, :, f2[2]+disp_s2:f2[2]+disp_s2+len_s2_d1]*(1-lam)
        sal_new[f1[2]:f1[3]] = sal_new[f1[2]:f1[3]]*lam_np + sal2[f2[2]+disp_s2:f2[2]+disp_s2+len_s2_d1]*(1-lam_np)
    # mix diastole of d1 with systole of d2
    len_dia_d1 = f1[4]-f1[3]
    len_dia_d2 = f2[4]-f2[3]
    sal1_dia = sal1[f1[3]:f1[4]]
    sal2_dia = sal2[f2[3]:f2[4]]
    #print(f'{len_dia_d1=} {len_dia_d2=}')
    if len_dia_d1 == len_dia_d2:
        disp_dia = 0
        d_new[ :, :, f1[3]:f1[4]] = d_new[ :, :, f1[3]:f1[4]]*lam + d2[ :, :, f2[3]:f2[4]]*(1-lam)
        sal_new[f1[3]:f1[4]] = sal_new[f1[3]:f1[4]]*lam_np + sal2[f2[3]:f2[4]]*(1-lam_np)
    elif len_dia_d1 > len_dia_d2:
        disp_dia = optimal_displacement_approach(sal1_dia, sal2_dia, lam_np)
        d_new[ :, :, f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2] = d_new[ :, :, f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2]*lam + d2[ :, :, f2[3]:f2[4]]*(1-lam)
        sal_new[f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2] = sal_new[f1[3]+disp_dia:f1[3]+disp_dia+len_dia_d2]*lam_np + sal2[f2[3]:f2[4]]*(1-lam_np)
    else: #(len_dia_d1 < len_dia_d2)
        disp_dia = optimal_displacement_approach(sal1_dia, sal2_dia, lam_np)
        d_new[ :, :, f1[3]:f1[4]] = d_new[ :, :, f1[3]:f1[4]]*lam + d2[ :, :, f2[3]+disp_dia:f2[3]+disp_dia+len_dia_d1]*(1-lam)
        sal_new[f1[3]:f1[4]] = sal_new[f1[3]:f1[4]]*lam_np + sal2[f2[3]+disp_dia:f2[3]+disp_dia+len_dia_d1]*(1-lam_np)
    return d_new

def mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lam, method, random_seed):
    d_new = d1.clone()
    ### this is the approach: data = data*lams_out + data[mix_indices]*(1-lams_out)
    # mix S1 of d1 with S1 of d2
    len_s1_min = min(f1[1]-f1[0], f2[1]-f2[0])
    d_new[:, :, f1[0]:f1[0]+len_s1_min] = d_new[:, :, f1[0]:f1[0]+len_s1_min]*lam + d2[:, :, f2[0]:f2[0]+len_s1_min]*(1-lam)
    # mix systole of d1 with systole of d2
    len_sys_min = min(f1[2]-f1[1], f2[2]-f2[1])
    d_new[:, :, f1[1]:f1[1]+len_sys_min] = d_new[:, :, f1[1]:f1[1]+len_sys_min]*lam + d2[:, :, f2[1]:f2[1]+len_sys_min]*(1-lam)
    # mix S2 of d1 with S2 of d2
    len_s2_min = min(f1[3]-f1[2], f2[3]-f2[2])
    d_new[:, :, f1[2]:f1[2]+len_s2_min] = d_new[:, :, f1[2]:f1[2]+len_s2_min]*lam + d2[:, :, f2[2]:f2[2]+len_s2_min]*(1-lam)
    # mix diastole of d1 with systole of d2
    len_dia_min = min(f1[4]-f1[3], f2[4]-f2[3])
    d_new[:, :, f1[3]:f1[3]+len_dia_min] = d_new[:, :, f1[3]:f1[3]+len_dia_min]*lam + d2[:, :, f2[3]:f2[3]+len_dia_min]*(1-lam)
    return d_new

def cutmix_keepdur_multidim_tensors(d1, d2, f1, f2, method, random_seed):
    d_new = d1.clone()
    if '(rand)' not in method:
        # swap systole of d1 with systole of d2
        len_sys_min = min(f1[2]-f1[1], f2[2]-f2[1])
        d_new[:, :, f1[1]:f1[1]+len_sys_min] = d2[:, :, f2[1]:f2[1]+len_sys_min]
        # swap diastole of d1 with systole of d2
        len_dia_min = min(f1[4]-f1[3], f2[4]-f2[3])
        d_new[:, :, f1[3]:f1[3]+len_dia_min] = d2[:, :, f2[3]:f2[3]+len_dia_min]
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

def augment(args, data, target_ohe, frames, wav, step_counter, model, device, RESULTS_ARGS):
    # Implemented methods
    methods_implemented = [
                            'durratiocutmix',
                            'cutmix',
                            'mixup',
                            'latentmixup',
                            'freqmask',
                            'timemask',
                            'cutout',
                            'durratiomixup',
                            'durmixfreqmask',
                            'durmixtimemask',
                            'durmixcutout',
                           ]

    if not any(map(args.method.__contains__, methods_implemented)):
        return data, target_ohe, [], None
    
    if 'durmixcutout' in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        lam = get_lambda(alpha=1, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_out = lams[:, None, None, None]
        data_new = torch.zeros((size, num_channels, spec_dim1, spec_dim2)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lams_out[0], args.method, random_seed)
            data_new[i, :, :] = d_new
        timemask_region_max = 0.2
        freqmask_region_max = 0.2
        if len(args.method.split('cutout(')) > 1:
            timemask_region_max = float(args.method.split('cutout(')[1].split(',')[0])
            timemask_region_max = min(max(timemask_region_max, 0), 1)
            freqmask_region_max = float(args.method.split(',')[1].split(')')[0])
            freqmask_region_max = min(max(freqmask_region_max, 0), 1)
        timemask_region_gap = random.Random(step_counter.count+131071).uniform(0, timemask_region_max)
        timemask_frac1 = random.Random(step_counter.count+13119).uniform(0, 1-timemask_region_gap)
        timemask_frac2 = timemask_frac1+timemask_region_gap
        freqmask_region_gap = random.Random(step_counter.count+131071).uniform(0, freqmask_region_max)
        freqmask_h1 = int(spec_dim1 * random.Random(step_counter.count+13119).uniform(0, 1-freqmask_region_gap))
        freqmask_h2 = min(spec_dim1, freqmask_h1+int(freqmask_region_gap * spec_dim1))
        for i, (d, f) in enumerate(zip(data_new, frames)):
            beat_len = f[-1].numpy()
            bb = [int(timemask_frac1*beat_len), int(timemask_frac2*beat_len)]
            d[:, freqmask_h1:freqmask_h2, bb[0]:bb[1]] = 0
        return data_new, target_ohe, mix_indices, None
    
    if 'durmixtimemask' in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        lam = get_lambda(alpha=1, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_out = lams[:, None, None, None]
        data_new = torch.zeros((size, num_channels, spec_dim1, spec_dim2)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lams_out[0], args.method, random_seed)
            data_new[i, :, :] = d_new
        mask_region_max = 0.2
        if len(args.method.split('timemask(')) > 1:
            mask_region_max = float(args.method.split('timemask(')[1].split(')')[0])
            mask_region_max = min(max(mask_region_max, 0), 1)
        mask_region_gap = random.Random(step_counter.count+131071).uniform(0, mask_region_max)
        mask_frac1 = random.Random(step_counter.count+13119).uniform(0, 1-mask_region_gap)
        mask_frac2 = mask_frac1+mask_region_gap
        for i, (d, f) in enumerate(zip(data_new, frames)):
            beat_len = f[-1].numpy()
            bb = [int(mask_frac1*beat_len), int(mask_frac2*beat_len)]
            d[:, :, bb[0]:bb[1]] = 0
        return data_new, target_ohe, mix_indices, None
    
    if 'durmixfreqmask' in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        lam = get_lambda(alpha=1, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_out = lams[:, None, None, None]
        data_new = torch.zeros((size, num_channels, spec_dim1, spec_dim2)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lams_out[0], args.method, random_seed)
            data_new[i, :, :] = d_new
        mask_region_max = 0.2
        if len(args.method.split('freqmask(')) > 1:
            mask_region_max = float(args.method.split('freqmask(')[1].split(')')[0])
            mask_region_max = min(max(mask_region_max, 0), 1)
        mask_region_gap = random.Random(step_counter.count+131071).uniform(0, mask_region_max)
        mask_h1 = int(spec_dim1 * random.Random(step_counter.count+13119).uniform(0, 1-mask_region_gap))
        mask_h2 = min(spec_dim1, mask_h1+int(mask_region_gap * spec_dim1))
        data_new[:, :, mask_h1:mask_h2, :] = 0
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
        lam = get_lambda(alpha=1, random_seed=random_seed)
        lams = np.ones(size)*lam
        lams_np = np.array(lams).astype('float32')
        lams = torch.from_numpy(lams_np).to(device)
        lams_out = lams[:, None, None, None]
        if '(salopt' in args.method:
            saliency_maps = saliency.get_saliency_maps(args, device, data.clone(), target_ohe.clone(), frames.clone(), dim=2)
        data_new = torch.zeros((size, num_channels, spec_dim1, spec_dim2)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            if '(salopt' in args.method:
                s1 = saliency_maps[i]
                s2 = saliency_maps[mix_indices][i]
                d_new = mixup_keepdur_multidim_tensors_salopt(d1, d2, f1, f2, s1, s2, lams_out[0], args.method, random_seed)
            else:
                d_new = mixup_keepdur_multidim_tensors(d1, d2, f1, f2, lams_out[0], args.method, random_seed)
            data_new[i, :, :] = d_new
        return data_new, target_ohe, mix_indices, None

    if 'cutout' in args.method and 'durmixcutout' not in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        timemask_region_max = 0.2
        freqmask_region_max = 0.2
        if len(args.method.split('cutout(')) > 1:
            timemask_region_max = float(args.method.split('cutout(')[1].split(',')[0])
            timemask_region_max = min(max(timemask_region_max, 0), 1)
            freqmask_region_max = float(args.method.split(',')[1].split(')')[0])
            freqmask_region_max = min(max(freqmask_region_max, 0), 1)
        timemask_region_gap = random.Random(step_counter.count+131071).uniform(0, timemask_region_max)
        timemask_frac1 = random.Random(step_counter.count+13119).uniform(0, 1-timemask_region_gap)
        timemask_frac2 = timemask_frac1+timemask_region_gap
        freqmask_region_gap = random.Random(step_counter.count+131071).uniform(0, freqmask_region_max)
        freqmask_h1 = int(spec_dim1 * random.Random(step_counter.count+13119).uniform(0, 1-freqmask_region_gap))
        freqmask_h2 = min(spec_dim1, freqmask_h1+int(freqmask_region_gap * spec_dim1))
        for i, (d, f) in enumerate(zip(data, frames)):
            beat_len = f[-1].numpy()
            bb = [int(timemask_frac1*beat_len), int(timemask_frac2*beat_len)]
            d[:, freqmask_h1:freqmask_h2, bb[0]:bb[1]] = 0
        return data, target_ohe, [], None

    if 'timemask' in args.method and 'durmixtimemask' not in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
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
            d[:, :, bb[0]:bb[1]] = 0
        return data, target_ohe, [], None
    
    if 'freqmask' in args.method and 'durmixfreqmask' not in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        mask_region_max = 0.2
        if len(args.method.split('freqmask(')) > 1:
            mask_region_max = float(args.method.split('freqmask(')[1].split(')')[0])
            mask_region_max = min(max(mask_region_max, 0), 1)
        mask_region_gap = random.Random(step_counter.count+131071).uniform(0, mask_region_max)
        mask_h1 = int(spec_dim1 * random.Random(step_counter.count+13119).uniform(0, 1-mask_region_gap))
        mask_h2 = min(spec_dim1, mask_h1+int(mask_region_gap * spec_dim1))
        data[:, :, mask_h1:mask_h2, :] = 0
        return data, target_ohe, [], None
    
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
        if args.model == 'resnet9':
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
        elif len(data.shape) == 4:
            lams_out = lams[:, None, None, None]    
        data = data*lams_out + data[mix_indices]*(1-lams_out)
        return data, target_ohe, mix_indices, None

    if 'mixup' in args.method and 'durratiomixup' not in args.method and 'latentmixup' not in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        if '(same)' in args.method:
            # Mixup data same label
            mix_indices = get_same_label_mix_indices(target_ohe, random_seed)
            lam = get_lambda(alpha=1, random_seed=random_seed)
            lams = np.ones(size)*lam
            lams_np = np.array(lams).astype('float32')
            lams = torch.from_numpy(lams_np).to(device)
            lams_out = lams[:, None, None, None]
            data = data*lams_out + data[mix_indices]*(1-lams_out)
            return data, target_ohe, mix_indices, None
        elif '(mix)' in args.method:
            # Mixup data with mixed labels
            mix_indices = random.Random(random_seed).sample(list(np.arange(0, size, 1)), size)
            lam = get_lambda(alpha=1, random_seed=random_seed)
            lams = np.ones(size)*lam
            lams_np = np.array(lams).astype('float32')
            lams = torch.from_numpy(lams_np).to(device)
            lams_out = lams[:, None, None, None]
            lams_target = lams[:, None]
            data = data*lams_out + data[mix_indices]*(1-lams_out)
            target_ohe = target_ohe*lams_target + target_ohe[mix_indices]*(1-lams_target)
            return data, target_ohe, mix_indices, None

    if 'cutmix' in args.method and 'durratiocutmix' not in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)    
        cut = 2 # cut in the middle
        if '(rand)' in args.method:
            cut = random.Random(step_counter.count*131071).randint(1, 3) # cut after S1, sys, or S2
        data_new = torch.zeros((size, num_channels, spec_dim1, spec_dim2)).to(device)
        frames_new = []
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new, f_new = cutmix_multidim_tensors(d1, d2, f1, f2, cut, num_channels, spec_dim1, spec_dim2, args.method, device)
            data_new[i, :, :, :] = d_new
            frames_new.append(f_new)
        return data_new, target_ohe, mix_indices, cut

    if 'durratiocutmix' in args.method:
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
        spec_dim1 = data.shape[2]
        spec_dim2 = data.shape[2]
        mix_indices = get_same_label_mix_indices(target_ohe, random_seed)            
        data_new = torch.zeros((size, num_channels, spec_dim1, spec_dim2)).to(device)
        for i, (d1, f1, d2, f2) in enumerate(zip(data, frames.numpy(), data[mix_indices], frames[mix_indices].numpy())):
            d_new = cutmix_keepdur_multidim_tensors(d1, d2, f1, f2, args.method, random_seed)
            data_new[i, :, :] = d_new
        return data_new, target_ohe, mix_indices, None
    