import torch
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import copy
import os
import torch.nn as nn

import results_new
import models
import models2d
import utils

def gaussian_kernel(n=11,sigma=1):
    ### Creates Gaussian kernel of lenght n
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]

def get_saliency_maps(args, device, data, target_ohe, frames, dim=1, gauss_k_n=101):
    # gaussian kernel
    #gauss_k = {19: 2.54, 49:4.54, 57:7.54, 101:12}
    #gauss_k_sigma = gauss_k[gauss_k_n]
    gauss_k_sigma = (12/101)*gauss_k_n
    # saliency map extractor model
    method_save = args.method
    args.method = 'base'
    if '-1' in method_save:
        args.method = 'durratiomixup'
        args = results_new.hyperparameters_robust(args)
    if '-2' in method_save:
        args.method = 'durmixmagwarp(0.2,4)'
        args = results_new.hyperparameters_robust(args)
    EXPERIMENT_ARGS = utils.experiment_dir(args)
    #print('Saliency model will be loaded from:', EXPERIMENT_ARGS)
    args.method = method_save
    MODEL_SAL = os.path.join(EXPERIMENT_ARGS, 'model.pth') # Load baseline model (no augmentation)
    if dim==1:
        if args.model == 'resnet9':
            model_sal = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'Potes':
            model_sal = models.CNN_potes_TS(num_channels=args.num_channels, num_classes=args.num_classes, dataset=args.dataset)
    elif dim==2:
        if args.model == 'resnet9':
            model_sal = models2d.ResNet9(num_classes = 2)
    else: 
        print('Error: Set dimension to either 1 or 2')
    model_sal = nn.DataParallel(model_sal) # sets to all available cuda devices
    model_sal.to(device)
    model_sal.load_state_dict(torch.load(MODEL_SAL))
    model_sal.eval()
    # Calculate saliencies
    target = target_ohe.max(1, keepdim=True)[1] # reverse ohe
    #data2 = copy.deepcopy(data)
    data.requires_grad_()
    # Forward pass data
    out = model_sal(data)
    label_view = target.view(-1, 1) # for correct class
    scores = (out.gather(1, label_view).squeeze())  # print(scores.shape[0]) # 500
    # Calculate gradients
    scores.backward(torch.FloatTensor([1.0]*scores.shape[0]).to(device))
    # Get saliency map
    saliency = data.grad.data.abs() 
    if dim==1:
        # Saliency is often non-zero in the zero-padded regions of the signals (this is brefore smoothing)
        for s, f in zip(saliency, frames.numpy()):
            s[:, f[-1]:] = 0
        # Sum the channels (the 200-400Hz channel is by far the most importnat)
        saliency = torch.sum(saliency, dim=1)[:, None, :]
        # Smooth the saliency maps
    #     kernel = gaussian_kernel(n=19, sigma=2.54) # Gaussian kernel
    #     kernel = gaussian_kernel(n=49, sigma=4.54) # Gaussian kernel
    #     kernel = gaussian_kernel(n=57, sigma=7.54) # Gaussian kernel
        kernel = gaussian_kernel(n=gauss_k_n, sigma=gauss_k_sigma) # Gaussian kernel
        kernel = torch.FloatTensor([[kernel]]).to(device)
        saliency = F.conv1d(saliency, kernel, padding='same') # luckily, the kernel can never be too large as is the case in np.convolve
        # Saliency is often non-zero in the zero-padded regions of the signals (this is after smoothing)
        for s, f in zip(saliency, frames.numpy()):
            s[:, f[-1]:] = 0
        # Instance-level normalization: normalize each instance between 0 and 1
        saliency_size = saliency.size()
        saliency = saliency.view(saliency.size(0), -1)
        saliency -= saliency.min(1, keepdim=True)[0]
        saliency /= saliency.max(1, keepdim=True)[0]
        saliency = saliency.view(saliency_size)
        # Fill the missing values as some saliencies that were full-zero are now "nan" after normalization
        saliency = torch.nan_to_num(saliency, nan=0.0)
        # To numpy
        saliency = saliency.detach().cpu().numpy()
        # Remove channel dimension
        saliency = np.squeeze(saliency)
    elif dim==2:
         # Saliency is often non-zero in the zero-padded regions of the signals
        for s, f in zip(saliency, frames.numpy()):
            s[:, :, f[-1]:] = 0
        # Sum the channels (the aug method only works in temporal direction so the frequency channels can be merged)
        saliency = torch.sum(saliency, dim=2)[:, None, :]
        # Smooth the saliency maps
        saliency = torch.squeeze(saliency, 2)
        kernel = gaussian_kernel(n=11, sigma=1) # Gaussian kernel
        kernel = torch.FloatTensor([[kernel]]).to(device)
        saliency = F.conv1d(saliency, kernel, padding='same')    
        # Saliency is often non-zero in the zero-padded regions of the signals (this is after smoothing)
        # Also, instance-level normalization
        for s, f in zip(saliency, frames.numpy()):
            s[:, f[-1]:] = 0 # set to zero after heartbeat
            # normalize only the heartbeat region between 0 and 1
            s[:, :f[-1]] -= s[:, :f[-1]].min()
            s[:, :f[-1]] /= s[:, :f[-1]].max()
        # Fill the missing values as some saliencies that were full-zero are now "nan" after normalization
        saliency = torch.nan_to_num(saliency, nan=0.0)
        # To numpy
        saliency = saliency.detach().cpu().numpy()
        # Remove channel dimension
        saliency = np.squeeze(saliency)
    return saliency

def bin_tensor(tensor, bins, device):
    num_channels = tensor.shape[0]
    sig_len = tensor.shape[1]
    samples_per_bin = int(np.ceil(sig_len/bins))
    # Downsample the input tensor to the same length as the bin number
    tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(bins,), mode='linear').squeeze(0)
    # Upsample the downsampled tensor to the original length of the input tensor
    tensor_new = torch.empty((num_channels, sig_len)).to(device)
    for i in range(num_channels):
        tensor_new[i] = tensor[i].repeat_interleave(samples_per_bin)[:sig_len]
    bin_values = tensor.cpu().detach().numpy().tolist()[0]
    bin_frames = list(np.arange(0, bins, 1)*samples_per_bin)
    return tensor_new, bin_values, bin_frames

def saliency_map(data, target_ohe, frames, model, device):
    size = data.shape[0]
    target = target_ohe.max(1, keepdim=True)[1] # reverse ohe
    data.requires_grad_()
    # Forward pass data
    model.eval()
    out = model(data)
    # returns scores of the correct (or selected if uncommented) class and puts them all into 1d tensor
    #selected_label = label
    #label_view = (torch.ones((size, 1)).type(torch.int64)*selected_label).to(device)
    label_view = target.view(-1, 1) # for correct class
    scores = (out.gather(1, label_view).squeeze())  # print(scores.shape[0]) # 500
    # Calculate gradients
    scores.backward(torch.FloatTensor([1.0]*scores.shape[0]).to(device))
    # Get saliency map
    saliency = data.grad.data.abs() 
    # Saliency is often non-zero in the zero-padded regions of the signals
    for d, f in zip(saliency, frames.numpy()):
        d[:, f[-1]:] = 0
    # Sum the channels (the 200-400Hz channel is by far the most importnat)
    saliency = torch.sum(saliency, dim=1)[:, None, :]
    # Smooth the saliency maps
    kernel = gaussian_kernel(n=19, sigma=2.54) # Gaussian kernel
    kernel = gaussian_kernel(n=49, sigma=4.54) # Gaussian kernel
    kernel = gaussian_kernel(n=57, sigma=7.54) # Gaussian kernel
    kernel = torch.FloatTensor([[kernel]]).to(device)
    saliency = F.conv1d(saliency, kernel, padding='same')
    # Instance-level normalization: normalize between 0 and 1 
    saliency_size = saliency.size()
    saliency = saliency.view(saliency.size(0), -1)
    saliency -= saliency.min(1, keepdim=True)[0]
    saliency /= saliency.max(1, keepdim=True)[0]
    saliency = saliency.view(saliency_size)
    # Fill the missing values as some saliencies that were full-zero are now "nan" after normalization
    saliency = torch.nan_to_num(saliency, nan=0.0)
    # Split systole and diastole in 6 and 12 bins, respectively
    num_channels = saliency.shape[1]
    sig_len =  saliency.shape[2]
    saliency_bins = torch.zeros((size, num_channels, sig_len))
    bin_values_batch = []
    bin_frames_batch = []
    for i, (d, f) in enumerate(zip(saliency, frames.numpy())):
        bin_values_d = []
        bin_frames_d = []
        # Copy S1
        tensor_new, bin_values, bin_frames = bin_tensor(d[:, f[0]:f[1]], 1, device)
        saliency_bins[i, :, f[0]:f[1]] = tensor_new #torch.mean(d[:, 0:f[1]], dim=-1) #d[:, f[0]:f[1]]
        bin_values_d = bin_values_d + bin_values
        bin_frames_d = bin_frames_d + bin_frames
        # Systole into 6 bins
        tensor_new, bin_values, bin_frames = bin_tensor(d[:, f[1]:f[2]], 4, device)
        saliency_bins[i, :, f[1]:f[2]] = tensor_new
        bin_values_d = bin_values_d + bin_values
        bin_frames_d = bin_frames_d + [bf + f[1] for bf in bin_frames]
        # Copy S2
        tensor_new, bin_values, bin_frames = bin_tensor(d[:, f[2]:f[3]], 1, device)
        saliency_bins[i, :, f[2]:f[3]] = tensor_new  #torch.mean(d[:, f[2]:f[3]], dim=-1) #d[:, f[2]:f[3]]
        bin_values_d = bin_values_d + bin_values
        bin_frames_d = bin_frames_d + [bf + f[2] for bf in bin_frames]
        # Systole into 6 bins
        tensor_new, bin_values, bin_frames = bin_tensor(d[:, f[3]:f[4]], 8, device)
        saliency_bins[i, :, f[3]:f[4]] = tensor_new
        bin_values_d = bin_values_d + bin_values
        bin_frames_d = bin_frames_d + [bf + f[3] for bf in bin_frames]
        bin_frames_d = bin_frames_d + [f[4]]
        # Append to the lists
        bin_values_d = np.array(bin_values_d)
        bin_frames_d = np.array(bin_frames_d)
        bin_values_batch.append(bin_values_d)
        bin_frames_batch.append(bin_frames_d)
    return saliency, saliency_bins, out, bin_values_batch, bin_frames_batch
