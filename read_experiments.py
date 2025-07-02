import os
from unittest import result
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils

def read_experiments_all_dataseeds(args, n_fractions, metric='Accuracy'):
    args.valid = False # just to make sure
    method = args.method
    upper = []
    mean = []
    lower = []
    std = []
    n_fracs_method = []
    #print(args.method)
    for n_frac in n_fractions:
        if n_frac == 0.015:
            seed_datas = np.arange(1001001, 1001334, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1001001, 1001201, 1)
        if n_frac == 0.052:
            seed_datas = np.arange(1005001, 1005101, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1005001, 1005061, 1)
        if n_frac == 0.1:
            seed_datas = np.arange(1010001, 1010051, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1010001, 1010031, 1)
        if n_frac == 0.2:
            seed_datas = np.arange(1020001, 1020026, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1020001, 1020016, 1)
        if n_frac == 0.3:
            seed_datas = np.arange(1030001, 1030017, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1030001, 1030011, 1)
        if n_frac == 0.4:
            seed_datas = np.arange(1040001, 1040013, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1040001, 1040009, 1)
        if n_frac == 0.6:
            seed_datas = np.arange(1060001, 1060009, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1060001, 1060006, 1)
        if n_frac == 0.8:
            seed_datas = np.arange(1080001, 1080007, 1)
            if args.dataset == 'PhysioNet(spec128)':
                seed_datas = np.arange(1080001, 1080005, 1)
        if n_frac == 1.0:
            seed_datas = [1100001]
        if n_frac == 1.0:
            seeds_test = [1, 2, 3, 4, 5]
            if args.dataset == 'PhysioNet(spec128)':
                seeds_test = [1, 2, 3]
        else:
            seeds_test = [1]
        args.method = method
        args.n_fraction = n_frac
        accs = []
        for seed_data in seed_datas:
            args.seed_data = seed_data
            args.method = method
            args = hyperparameters_robust(args)
            for seed in seeds_test:
                args.seed = seed
                RESULTS_ARGS = utils.results_dir(args)
                #print(RESULTS_ARGS)
                if utils.experiment_already_done(args):
                    RESULTS_ARGS = utils.results_dir(args)
                    perf = utils.read_pkl_perf(RESULTS_ARGS)
                    if metric == 'Accuracy':
                        acc_test = perf['test_accuracy']
                    elif metric == 'ROC AUC':
                        acc_test = perf['test_rocauc']
                        acc_test = [x*100 for x in acc_test]
                    elif metric == 'F1 score':
                        acc_test = perf['test_f1']
                        acc_test = [x*100 for x in acc_test]
                    elif metric == 'Specificity':
                        acc_test = perf['test_specificity']
                    elif metric == 'Sensitivity':
                        acc_test = perf['test_sensitivity']
                    elif metric == 'Precision':
                        acc_test = perf['test_precision']
                        acc_test = [x*100 for x in acc_test]
                    elif metric == 'Recall':
                        acc_test = perf['test_recall']
                        acc_test = [x*100 for x in acc_test]
                    accs.append(acc_test[-1])
        '''
        if accs == [] and args.n_fraction == 0.1:
            print(f'(n_fraction = 0.1): No results for method {args.method} {args.seed_data=}')
        '''
        if accs != []:
            #print(f'Method: {args.method}:', accs, np.mean(accs), '+-', np.std(accs), '#:', len(accs))
            print(f'Method: {args.method}:', np.round(np.mean(accs), 2), '+-', np.round(np.std(accs), 2), '#:', len(accs))
            upper.append(np.max(accs))
            mean.append(np.mean(accs))
            lower.append(np.min(accs))
            std.append(np.std(accs))
            n_fracs_method.append(n_frac)
    num_exp = len(accs)
    args.seed_data = seed_data # set back to the previous seed_data, orelse the next method will use the updated one (seed_data=3)
    return mean, lower, upper, std, n_fracs_method, num_exp 

def read_experiments(args, approach, n_fractions):
    #args.valid = False # just to make sure
    ### approach: 'finetune' or 'max'
    seeds = [1, 2, 3]
    seed_data = args.seed_data
    method = args.method
    upper = []
    mean = []
    lower = []
    std = []
    n_fracs_method = []
    methods_aug = ['polarityinv', 'addnoise', 'gain', 'basicaugments']
    if any(map(args.method.__contains__, methods_aug)):
        args.aug = True
    else: 
        args.aug = False
    for n_frac in n_fractions:
        args.method = method
        args.n_fraction = n_frac
        if args.n_fraction == 1.0:
            args.seed_data = 3
        else:
            args.seed_data = seed_data
        args = hyperparameters_robust(args, approach)
        accs = []
        for seed in seeds:
            args.seed = seed
            RESULTS_ARGS = utils.results_dir(args)
            #print(RESULTS_ARGS)
            if utils.experiment_already_done(args):
                acc_test, _ = utils.read_pkl_acc(RESULTS_ARGS)
                accs.append(acc_test[-1])
        if accs != []:
            upper.append(np.max(accs))
            mean.append(np.mean(accs))
            lower.append(np.min(accs))
            std.append(np.std(accs))
            n_fracs_method.append(n_frac)
    num_exp = len(accs)
    args.seed_data = seed_data # set back to the previous seed_data, orelse the next method will use the updated one (seed_data=3)
    return mean, lower, upper, std, n_fracs_method, num_exp 

def hyperparameters_robust(args):
    if args.dataset in ['PhysioNet', 'PhysioNet(spec128)']:
        if args.model == 'resnet9' or args.model == 'Potes' or args.model == 'Singstad_d10':
            if args.model == 'resnet9' or args.model == 'Potes':
                args.num_epochs = 50
                args.lr_max = 0.01
            elif args.model == 'Singstad_d10':
                args.num_epochs = 30
                args.lr_max = 0.00001
            n_fractions = [0.015, 0.052, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
            if args.method == 'durmixmagwarp(0.2,4)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == 'durratiomixup':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == 'mixup(same)': 
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.2]
            if args.method == 'latentmixup': 
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.2, 0.2, 0.2]
            if args.method == 'magnitudewarp(0.2,4)': 
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.4, 0.4]
            if args.method == 'timewarp(0.05,4)': #CHANGE THIS ONE
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.2, 0.2, 0.2]
            if args.method == 'respiratoryscale(12,20)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.2, 0.2, 0.2]
            if args.method == 'timemask(0.2)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == 'gaussiannoise(25,40)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.2]
            if args.method == '(sameCVD)durmixmagwarp(0.2,4)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == '(samePCG)durmixmagwarp(0.2,4)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == '(sameDataset)durmixmagwarp(0.2,4)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == '(mixAll)durmixmagwarp(0.2,4)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == '(sameCVD)durratiomixup':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == 'freqmask(0.1)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == 'timemask(0.1)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == 'cutout(0.25,0.25)':
                # p = [1.5, 5.2, 10,  20,  30,  40,  60,  80,  100]
                cps = [1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2]
            if args.method == 'base':
                return args
            else:
                cp = cps[n_fractions.index(args.n_fraction)]
                args.method = f'{args.method}+{cp}'
                return args
    return args




