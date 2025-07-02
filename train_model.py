from warnings import filters
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import copy

import classical
import latent_space
import augmentations2d
import augmentations
import dataloader_umc2d
import dataloader_physionet2d
import dataloader_umc
import dataloader_physionet
import models2d
import models
import plotters
import utils

import tsai
from tsai.models.InceptionTime import *
from tsai.models.InceptionTimePlus import *
from tsai.models.XceptionTime import *
from tsai.models.XceptionTimePlus import *
from tsai.models.ResNetPlus import *
from tsai.models.gMLP import *
from tsai.models.XCM import *
from tsai.models.XCMPlus import *
from tsai.models.FCN import *
from tsai.models.FCNPlus import *
from tsai.models.RNN import *
from tsai.models.mWDN import *
from tsai.models.XResNet1d import *
from tsai.models.XResNet1dPlus import *
from tsai.models.OmniScaleCNN import *

class CELoss(torch.nn.Module):
    '''Cross-entropy loss that works with soft targets'''
    def __init__(self, num_classes):
        super(CELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, target_ohe):
        pred = F.log_softmax(logits, dim=1)
        ce = -torch.sum(pred * target_ohe, dim = 1)
        return ce.mean()
        
class SELCLoss(torch.nn.Module):
    def __init__(self, labels, num_classes, es=10, momentum=0.9):
        super(SELCLoss, self).__init__()
        self.num_classes = num_classes
        self.soft_labels = torch.zeros(len(labels), num_classes, dtype=torch.float).cuda()
        self.soft_labels[torch.arange(len(labels)), labels] = 1
        self.es = es
        self.momentum = momentum
        self.CEloss = CELoss(num_classes)

    def forward(self, logits, labels, index, epoch, mode):
        pred = F.softmax(logits, dim=1)
        if mode == 'test':
            ce = self.CEloss(logits, labels)
            return ce
        elif mode == 'train':
            if epoch <= self.es:
                ce = self.CEloss(logits, labels)
                return ce
            else:
                pred_detach = F.softmax(logits.detach(), dim=1)
                self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * pred_detach

                selc_loss = -torch.sum(torch.log(pred) * self.soft_labels[index], dim=1)
                return selc_loss.mean()

def custom_loss(output, target_ohe, num_classes = 2):
    num_samples = output.shape[0]
    output_lsm = F.log_softmax(output, dim=1)
    #target_ohe = F.one_hot(target, num_classes)

    losses = target_ohe*output_lsm
    losses_1d = torch.sum(losses, dim=1)
    losses_1d = losses_1d.cpu().detach().numpy()*-1 
    
    loss = losses_1d.sum()/num_samples
    loss = loss.item() # this is what nn.CrossEntropyLoss returns

    target = target_ohe.max(1, keepdim=True)[1]
    target = [x.item() for x in target]
    target = torch.tensor(target)
    output_classes = torch.argmax(output, dim=1)
    output_classes_correct = output_classes.cpu().detach().numpy() == target.cpu().detach().numpy()

    losses_1d_correct = losses_1d[output_classes_correct]
    losses_1d_incorrect = losses_1d[~output_classes_correct]
    
    return losses_1d_correct, losses_1d_incorrect, loss, losses_1d   
    
class step_counter_class():
    def __init__(self):
        self.count = 0
    def add(self):
        self.count +=1

class variability_counter_class():
    '''
    Count 3 types of variability:
    base: original segments
    pairs: (sample1,sample2)=(sample1,sample1), where sample1!=sample2
    unique combinations: (sample1, sample2, cut), where sample1!=sample2
    '''
    def __init__(self):
        # Number of original samples in the experiment
        self.base_original = 0
        # Lists to save the samples during the experiment
        self.base = []
        self.pairs = []
        self.unique = []
        # For plotting
        self.steps = []
        self.lens_base = []
        self.lens_pairs = []
        self.lens_unique = []
        
    def add(self, indices, mix_indices, cut, step):
        if mix_indices == []:
            batch_indices = list(indices.numpy().astype(str))
        else:
            batch_indices = []
            batch_indices = batch_indices + [f'{sorted([p1, p2])[0]}_{sorted([p1, p2])[1]}' for p1, p2 in zip(indices.numpy(), indices.numpy()[mix_indices])]
            batch_indices = batch_indices + [f'{p1}_{p2}_{cut}' for p1, p2 in zip(indices.numpy(), indices.numpy()[mix_indices])]
        for w in batch_indices :
            w_parts = w.split('_')
            if len(w_parts) == 1:
                self.base.append(w) 
            elif len(w_parts) == 2:
                if w_parts[0]==w_parts[1]:
                   self.base.append(w_parts[0])
                else:
                    self.pairs.append(w) 
            elif len(w_parts) == 3:
                if w_parts[0]==w_parts[1]:
                   self.base.append(w_parts[0])
                else:
                    self.unique.append(w)
            else:
                raise Exception(f"Sorry, {w} is not of correct string shape")
        self.base = list(set(self.base))
        self.pairs = list(set(self.pairs))
        self.unique = list(set(self.unique))
        self.steps.append(step)
        self.lens_base.append(len(self.base))
        self.lens_pairs.append(len(self.pairs))
        self.lens_unique.append(len(self.unique))

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_gradients_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def set_seed(seed):
    torch.manual_seed(seed)  # CPU seed and all GPUs //Also affects dropout in the models
    random.seed(seed)  # python seed
    np.random.seed(seed) # numpy seed

class performance_metrics_class():
    def __init__(self):
        self.dict = {'steps':[],
                    'epochs':[],
                    'times':[],
                    'train_loss':[],
                    'train_accuracy':[],
                    'test_loss':[],
                    'test_accuracy':[],
                    'test_specificity':[],
                    'test_sensitivity':[],
                    'test_precision':[],
                    'test_recall':[],
                    'test_f1':[],
                    'test_rocauc':[],
                        }  
    def add(self, string, value):
        self.dict[string].append(value)
    
def train_model(args, dataset, device):
    print(f'TRAINING MODEL {args.model}')
    print(f'\tDataset: {args.dataset}')
    print(f'\tSeed(data): {args.seed_data}')
    print(f'\tNumber of channels: {args.num_channels}')
    print(f'\tMethod: {args.method}')
    print(f'\tNumber of epochs: {args.num_epochs}')
    print(f'\tBatch size: {args.batch_size}')
    print(f'\tLearning rate: {args.lr}')
    print(f'\tMax learning rate: {args.lr_max}')
    print(f'\tUse learning rate scheduler: {args.use_sched}')
    print(f'\tOptimizer: {args.op}')
    print(f'\tDepth: {args.depth}')
    print(f'\tSeed: {args.seed}')
    print(f'\tn_fraction: {args.n_fraction}')
    print(f'\tTrain_balance: {args.train_balance}')
    print(f'\tValidation: {args.valid}')

    # Fix the random seeds for reproduction purposes
    args.seed_fix = 4
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed_fix)  # CPU seed and all GPUs //Also affects dropout in the models
    #torch.cuda.manual_seed_all(seed_fix)  # GPU seed (_all for multi-GPU)
    random.seed(args.seed_fix)  # python seed
    np.random.seed(args.seed_fix) # numpy seed
    os.environ["PYTHONHASHSEED"] = str(args.seed_fix)
    
    EXPERIMENT_ARGS = utils.check_folder(utils.experiment_dir(args))

    # Initialize train and test data loaders
    if args.dataset == 'PhysioNet(spec128)': # for spectrograms
        data_loader = dataloader_physionet2d.physionet_dataloader(args, dataset)
        train_loader, train_labels = data_loader.run(mode='train', transform_seed=args.seed_fix)
        if args.valid==True:
            test_loader = data_loader.run(mode='valid', transform_seed=None)
        else:
            test_loader = data_loader.run(mode='test', transform_seed=None)
    elif args.dataset in ['UMC(spec128)', 'UMC(spec64)']:
        data_loader = dataloader_umc2d.umc_dataloader(args, dataset)
        train_loader, train_labels = data_loader.run(mode='train', transform_seed=args.seed_fix)
        if args.valid==True:
            test_loader = data_loader.run(mode='valid', transform_seed=None)
        else:
            test_loader = data_loader.run(mode='test', transform_seed=None)
    else: #for timeseries
        if args.dataset == 'PhysioNet':
            data_loader = dataloader_physionet.physionet_dataloader(args, dataset)
            train_loader, train_labels = data_loader.run(mode='train', transform_seed=args.seed_fix)
            if args.valid==True:
                test_loader = data_loader.run(mode='valid', transform_seed=None)
            else:
                test_loader = data_loader.run(mode='test', transform_seed=None)
        elif args.dataset == 'UMC':
            data_loader = dataloader_umc.umc_dataloader(args, dataset)
            train_loader, train_labels = data_loader.run(mode='train', transform_seed=args.seed_fix)
            if args.valid==True:
                test_loader = data_loader.run(mode='valid', transform_seed=None)
            else:
                test_loader = data_loader.run(mode='test', transform_seed=None)

    # Print out the train and test data loaders properties
    print(f'\ttrain data size:')
    print(f'\t\twavs: {len(list(set(train_loader.dataset.train_wav)))}')
    print(f'\t\twavs: {sorted(list(set(train_loader.dataset.train_wav)))}')
    wavs_label = [0]*args.num_classes
    wavs_flat = []
    segments_label = [0]*args.num_classes
    for i, (wav, label) in enumerate(zip(train_loader.dataset.train_wav, train_loader.dataset.train_label)):
        segments_label[label] += 1
        if wav not in wavs_flat:
            wavs_label[label] += 1
            wavs_flat.append(wav)
    for i, count in enumerate(wavs_label):
        print(f'\t\t\tlabel {i} :', count)
    print(f'\t\tsegments: {len(train_loader.dataset)}')
    for i, count in enumerate(segments_label):
        print(f'\t\t\tlabel {i} :', count)
    print(f'\ttest data size:')  
    print(f'\t\twavs: {len(list(set(test_loader.dataset.test_wav)))}')
    print(f'\t\twavs: {sorted(list(set(test_loader.dataset.test_wav)))}')
    wavs_label = [0]*args.num_classes
    wavs_flat = []
    segments_label = [0]*args.num_classes
    for i, (wav, label) in enumerate(zip(test_loader.dataset.test_wav, test_loader.dataset.test_label)):
        segments_label[label] += 1
        if wav not in wavs_flat:
            wavs_label[label] += 1
            wavs_flat.append(wav)
    for i, count in enumerate(wavs_label):
        print(f'\t\t\tlabel {i} :', count)
    print(f'\t\tsegments: {len(test_loader.dataset)}')
    for i, count in enumerate(segments_label):
        print(f'\t\t\tlabel {i} :', count)

    # initialize the model
    torch.manual_seed(args.seed_fix) # set the initial weights with fixed seed
    if args.dataset == 'PhysioNet(spec128)': # for spectrogram models (2d convolutions)
        if args.model == 'resnet9':
            model = models2d.ResNet9(num_classes = 2) #linear=8192
    elif args.dataset == 'UMC(spec128)': # for spectrogram models (2d convolutions)
        if args.model == 'resnet9':
            model = models2d.ResNet9(num_classes = 2)
    elif args.dataset == 'UMC(spec64)': # for spectrogram models (2d convolutions)
        if args.model == 'resnet9':
            model = models2d.ResNet9(num_classes = 2, linear=2048)
    else: # timeseries models (1d convolutions)
        if args.model == 'ResNet':
            model = models.ResNet_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'ResNetPlus':
            model = tsai.models.ResNetPlus.ResNetPlus(4, 2)
        elif args.model == 'XResNet1d18':
            model = tsai.models.XResNet1d.xresnet1d18(4, 2)
        elif args.model == 'XResNet1d18Plus':
            model = tsai.models.XResNet1dPlus.xresnet1d18plus(4, 2)
        elif args.model == 'ResCNN':
            model = models.ResCNN_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'InceptionTime':
            model = tsai.models.InceptionTime.InceptionTime(4, 2)
        elif args.model == 'InceptionTimePlus':
            model = tsai.models.InceptionTimePlus.InceptionTimePlus(4, 2)
        elif args.model == 'XceptionTime':
            model = tsai.models.XceptionTime.XceptionTime(4, 2)
        elif args.model == 'XceptionTimePlus':
            model = tsai.models.XceptionTimePlus.XceptionTimePlus(4, 2)
        elif args.model == 'gMLP':
            model = tsai.models.gMLP.gMLP(4, 2, args.sig_len)
        elif args.model == 'XCM':
            model = tsai.models.XCM.XCM(4, 2, args.sig_len)
        elif args.model == 'XCMPlus':
            model = tsai.models.XCMPlus.XCMPlus(4, 2, args.sig_len)
        elif args.model == 'FCN':
            #model = tsai.models.FCN.FCN(4, 2)
            model = models.FCN_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'Singstad_d3':
            model = models.inceptiontime_singstad_d3_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'Singstad_d6':
            model = models.inceptiontime_singstad_d6_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'Singstad_d10':
            model = models.inceptiontime_singstad_d10_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'resnet9':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes)
            if args.dataset == 'UMC':
                model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, linear=31744)
        elif args.model == 'resnet9-5k':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[2, 4, 8, 16], linear=1248)
        elif args.model == 'resnet9-15k':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[4, 8, 16, 32], linear=2496)
        elif args.model == 'resnet9-50k':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[8, 16, 32, 64], linear=4992)
        elif args.model == 'resnet9-150k':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[16, 32, 64, 128], linear=9984)
        elif args.model == 'resnet9-600k':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[32, 64, 128, 256], linear=19968)
        elif args.model == 'resnet9-1.4m':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[64, 128, 192, 384], linear=29952)
        elif args.model == 'resnet9-2.3m':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[64, 128, 256, 512], linear=39936)
        elif args.model == 'resnet9-5m':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[96, 192, 384, 768], linear=59904)
        elif args.model == 'resnet9-9m':
            model = models.ResNet9(in_channels=args.num_channels, num_classes=args.num_classes, filters=[128, 256, 512, 1024], linear=79872)
        elif args.model =='Potes':
            model = models.CNN_potes_TS(num_channels=args.num_channels, num_classes=args.num_classes, dataset=args.dataset)
        elif args.model =='PotesBig128and64':
            model = models.CNN_potes_big128and64_TS(num_channels=args.num_channels, num_classes=args.num_classes, dataset=args.dataset)
        elif args.model =='PotesBig64and32':
            model = models.CNN_potes_big64and32_TS(num_channels=args.num_channels, num_classes=args.num_classes, dataset=args.dataset)
        elif args.model =='Potes(noDropout)':
            model = models.CNN_potes_TS(num_channels=args.num_channels, num_classes=args.num_classes, dataset=args.dataset, dropout=0.0)
        elif args.model =='Potes0.1':
            model = models.CNN_potes_tenpercent_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model =='Potes0.02':
            model = models.CNN_potes_twopercent_TS(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'FCN(custom)':
            model = models.FCN_TS_custom(num_channels=args.num_channels, num_classes=args.num_classes)
        elif args.model == 'FCNPlus':
            model = tsai.models.FCNPlus.FCNPlus(4, 2)
        elif args.model == 'RNN':
            model = tsai.models.RNN.RNN(4, 2)
        elif args.model == 'LSTM':
            model = tsai.models.RNN.LSTM(4, 2)
        elif args.model == 'GRU':
            model = tsai.models.RNN.GRU(4, 2)
        elif args.model == 'mWDN':
            model = tsai.models.mWDN.mWDN(4, 2, 2500)
        elif args.model == 'OmniScaleCNN':
            model = tsai.models.OmniScaleCNN.OmniScaleCNN(4, 2, 2500)
    model = nn.DataParallel(model) # sets to all available cuda devices
    model.to(device)
    print(f'\tModel parameters count: {count_model_parameters(model)}')
    
    # calculate the number of epochs
    args.num_steps = args.num_epochs*(len(train_loader.dataset)//args.batch_size)
    print(f'\tNumber of steps (calculated): {args.num_steps}')
    
    # initialize the criterion
    if 'SELC' in args.method:
        if 'mixup' in args.method:
            es = int(args.num_epochs*0.4) # when using mixup, turn point is at 40% of training
        elif 'base' in args.method:
            es = int(args.num_epochs*0.4) # if using base, turn point is at 20% of training
    else:
        es = args.num_epochs+1 # otherwise, set turn point so high that SELC is never activated
    print(f'\tTurning point epoch: {es}')
    criterion = SELCLoss(labels=train_labels, num_classes=args.num_classes, es=es)
    # initialize the optimizer
    if args.op == 'SGD':
        optimizer = torch.optim.SGD(params = model.parameters(), lr = args.lr_max, weight_decay = args.weight_decay)
    elif args.op == 'adam':
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr_max, weight_decay = args.weight_decay)
    # initialize the scheduler
    if args.use_sched:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer, max_lr = args.lr_max, total_steps = args.num_steps) #COMMENT
    else: 
        scheduler = None #COMMENT
    # initialize the step counter
    step_counter = step_counter_class()
    # initialize the variability counter
    variability_counter = variability_counter_class()
    variability_counter.base_original = len(train_loader.dataset)
    # initialize performance metric tracker
    performance = performance_metrics_class()
                    
    times = []
    lr_per_step = []
    step_saver = [] # used for variables that are saved every epoch
    epoch_plot = np.linspace(1, args.num_epochs, 11).astype('int') #11 #51 #args.num_epochs+1
    epoch_plot = np.array(list(set(epoch_plot)))
    print(f'Number of plot epochs:', len(epoch_plot))
    print('->Training START<-')
    args.depth = 0
    for epoch in range(1, args.num_epochs + 1):
        time_start_epoch = time.time()
        
        # Train the model for one epoch
        loss_train_epoch, acc_train_epoch, lr_per_step_epoch = train_epoch(args, 
                                                                            model, 
                                                                            train_loader,
                                                                            device,
                                                                            optimizer, 
                                                                            scheduler, 
                                                                            criterion, 
                                                                            epoch, 
                                                                            step_counter,
                                                                            variability_counter,
                                                                            EXPERIMENT_ARGS)
        step_saver.append(step_counter.count)
        lr_per_step = lr_per_step + lr_per_step_epoch
        
        if epoch in epoch_plot:
            performance.add('epochs', epoch)
            performance.add('steps', step_counter.count)
            print(f'Plotting @ Epoch={epoch}, Step={step_counter.count}')
            # Train data accuracy
            performance.add('train_loss', loss_train_epoch)
            performance.add('train_accuracy', acc_train_epoch)
            # Test data accuracy
            test_data_accuracy(args, model, test_loader, device, criterion, epoch, performance)
            print('\tMean test loss per epoch: {:.6f}'.format(performance.dict['test_loss'][-1]))
            ### Plots:
            # train and test accuracies vs epoch
            plotters.plot_train_test_acc(performance.dict['train_accuracy'], performance.dict['test_accuracy'], args.valid, performance.dict['steps'], EXPERIMENT_ARGS)
            # train and test losses vs epoch
            plotters.plot_train_test_loss(performance.dict['train_loss'], performance.dict['test_loss'], args.valid, performance.dict['steps'], EXPERIMENT_ARGS)
            # learning rate vs batch
            plotters.plot_lr_per_step(lr_per_step, EXPERIMENT_ARGS, show = False)
            # Latent space
            #if args.latent_space:
                #latent_space.save_latent_space(latent_space_dict_train, 'train', step_counter.count, EXPERIMENT_ARGS)
            # Variability
            #plotters.plot_variability(variability_counter, EXPERIMENT_ARGS)
            
        times.append(time.time()-time_start_epoch)
        if epoch in epoch_plot:
            # times vs epoch
            performance.add('times', np.sum(times))
            plotters.plot_times(times, step_saver, EXPERIMENT_ARGS, show = False)
        # Save performance dict
        if epoch in epoch_plot:
            DICT = os.path.join(EXPERIMENT_ARGS, f'performance.pkl')
            utils.save_dict(performance.dict, DICT)
    
    print('Finished Training')
    MODEL = os.path.join(EXPERIMENT_ARGS, 'model.pth')
    torch.save(model.state_dict(), MODEL)
    
    # Release cuda memory
    for variable in [criterion, optimizer, scheduler, model]:
        del variable

    return
    
def train_epoch(args, model, train_loader, device, optimizer, scheduler, criterion, epoch, step_counter, variability_counter, EXPERIMENT_ARGS):
    model.train()

    print(f'Step [{step_counter.count+1}/{args.num_steps}]')
    loss_per_batch = []
    lr_per_step = []
    pred_dict = {}
    torch.manual_seed(args.seed*635410 + step_counter.count) # this is here as some other methods (measure_test_accuracy()...) call into the pseudo-RNG and changes the order of training data of the next step       
    for batch_idx, (data, target, frames, wav, sig_qual, indices) in enumerate(train_loader):
        data = data.to(device)
        target_ohe = F.one_hot(target, args.num_classes)
        target_ohe = target_ohe.to(device)

        # Augment data 
        if args.dataset in ['PhysioNet(spec128)', 'UMC(spec128)', 'UMC(spec64)']: # spectrograms
            data, target_ohe, _, _ = augmentations2d.augment(args, data, target_ohe, frames, wav, step_counter, model, device, EXPERIMENT_ARGS)
        else: #timeseries
            data, target_ohe, _, _ = augmentations.augment(args, data, target_ohe, frames, wav, step_counter, model, device, EXPERIMENT_ARGS)
        if args.latent_space:
            # Save latent space features
            if 'latent' in args.method: # works very bad with latent mixup as the model in .train() mode produces very different outputs as the one in the .eval() mode
                fts_latent = data.cpu().detach().numpy()
            else:
                if args.classical_space:
                    fts_latent = latent_space.generate_latent_space(data[:, :4, :].clone()) # if classical is also activated, the data has 5 channels instead of 4
                else:
                    fts_latent = latent_space.generate_latent_space(data.clone())
            latent_space_dict = {'fts': fts_latent, 'target':target.numpy()}
            latent_space.save_latent_space(latent_space_dict, 'train', step_counter.count, EXPERIMENT_ARGS)
        if args.classical_space:
            # Save classical space features
            fts_classical = pd.DataFrame(dtype=float)
            if 'latent' in args.method: continue
            else:
                for d, t, f, sq, w, in zip(data.cpu().detach().numpy(), target.numpy(), frames.numpy(), sig_qual.numpy(), wav):
                    # extract classical features from the [25, 400] channel only
                    fts_classical_vec = classical.feature_vector_seg(d[4], t, f, w, sq, 5, 'train')
                    fts_classical = pd.concat([fts_classical, fts_classical_vec], axis=1) 
            fts_classical = fts_classical.T.reset_index(drop=True)    
            CLASSICAL_SPACE = utils.check_folder(os.path.join(EXPERIMENT_ARGS, 'classical_space'))
            FN = os.path.join(CLASSICAL_SPACE, f'train_{step_counter.count}.csv')
            fts_classical.to_csv(FN, index = False)
            data = data[:, :4, :] # reduce to 4 channels

        if args.model in ['XceptionTime', 'InceptionTime', 'ResCNN', 'XResNet1d18', 'FCN', 'ResNet']:
            out = model(data)
        else:
            out = model(data, depth=args.depth, pass_part='second')
        args.depth = 0 # reset in case we use 'latent-cutmix'
        loss = criterion(out, target_ohe, indices, epoch, 'train')

        # Append loss
        loss_per_batch.append(loss.item())
        # Save predictions in dictionary
        pred = out.max(1, keepdim=True)[1]  # get the index of the max log-probability
        target = target_ohe.max(1, keepdim=True)[1] # reverse ohe
        for pred_i, target_i in zip(pred, target):
            pred_i = pred_i.item()
            target_i = target_i.item()
            if target_i not in pred_dict:
                pred_dict[target_i] = list([pred_i])
            else:
                pred_dict[target_i].append(pred_i)
        
        # Backward
        loss.backward()
        # Gradient clipping
        if args.grad_clip:
            nn.utils.clip_grad_value_(parameters = model.parameters(), clip_value = args.grad_clip)
            
        # Save some things
        learning_rate = optimizer.param_groups[0]['lr']
        lr_per_step.append(learning_rate)
        
        # Optimize
        torch.cuda.manual_seed_all(args.seed_fix) # SGD optimizer is non-deterministic, that's why we fix the seed
        optimizer.step()
        optimizer.zero_grad()
        if args.use_sched:
            scheduler.step() #COMMENT

        # Release cuda memory
        for var in [data, target_ohe, out, target]:
            var.cpu().detach()
            del var
        
        # Add a step to the counter
        step_counter.add()
        # Update variability counter
        # Stop
        if not step_counter.count < args.num_steps:
            print(f'Training loop was stopped: epoch {epoch}, step {step_counter.count}')
            break
            
    acc_network = calc_acc(args, pred_dict)        
    # Release cuda memory
    #gc.collect()
    #torch.cuda.empty_cache()

    return np.average(loss_per_batch), acc_network, lr_per_step
                
def test_data_accuracy(args, model, test_loader, device, criterion, epoch, performance):
    from collections import Counter

    model.eval()

    losses_all = 0
    pred_dict = {}
    wav_targets_dict = {}
    with torch.no_grad():
        for data, target, _, wav, _, _ in test_loader:
            data = data.to(device)
            target_ohe = F.one_hot(target, args.num_classes)
            target_ohe = target_ohe.to(device)
            size = data.shape[0]

            output = model(data)

            loss = criterion(output, target_ohe, None, None, 'test')
            losses_all += loss.item()*size   

            # calculate accuracy
            pred_proba = F.softmax(output, dim=1).cpu().detach().numpy()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
            for pred_proba_i, target_i, wav_i in zip(pred_proba, target, wav):
                if wav_i not in pred_dict:
                    pred_dict[wav_i] = list([pred_proba_i])
                    wav_targets_dict[wav_i] = target_i[0]
                else:
                    pred_dict[wav_i].append(pred_proba_i)
    # Majority vote from segments based on mean prediction probabilities
    wav_majority = []
    wav_majority_predproba = []
    wav_targets = []
    for key in pred_dict:
        arr = pred_dict[key]
        arr_mean = np.mean(arr, axis=0)
        wav_majority_predproba.append(arr_mean)
        pred = np.argmax(arr_mean)
        wav_majority.append(pred)
        wav_target = wav_targets_dict[key]
        wav_targets.append(wav_target)
    if '(class_majority)' in args.method:
        wav_majority = []
        wav_majority_predproba = []
        wav_targets = []
        for key in pred_dict:
            arr = pred_dict[key]
            pred = [np.argmax(x) for x in arr]
            counts = np.bincount(pred)
            majority_vote = np.argmax(counts)
            if (len(counts) == 2) and (counts[0] == counts[1]):
                majority_vote = 1
            wav_majority.append(majority_vote)
            wav_target = wav_targets_dict[key]
            wav_targets.append(wav_target)

    equal = np.equal(np.array(wav_targets), np.array(wav_majority))
    # Accuracy
    acc_network = np.sum(equal)/len(wav_targets)*100
    performance.add('test_accuracy', acc_network)
    # Loss
    loss_total = losses_all/len(test_loader.dataset)
    performance.add('test_loss', loss_total)
    # Specificity and sensitiviy
    tn, fp, fn, tp = confusion_matrix(wav_targets, wav_majority).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    performance.add('test_specificity', specificity*100)
    performance.add('test_sensitivity', sensitivity*100)
    # Precision, recall, f1, aucroc
    f1 = f1_score(wav_targets, wav_majority)
    performance.add('test_f1', f1)
    precision = precision_score(wav_targets, wav_majority)
    performance.add('test_precision', precision)
    recall = recall_score(wav_targets, wav_majority)
    performance.add('test_recall', recall)
    rocauc = roc_auc_score(wav_targets, np.array(wav_majority_predproba)[:, 1])
    performance.add('test_rocauc', rocauc)
    return 

def calc_acc(args, pred_dict, verbose=True):
    correct = 0
    samples = 0
    class_correct = [0 for i in range(args.num_classes)]
    class_samples = [0 for i in range(args.num_classes)]
    for target in pred_dict:
        pred = pred_dict[target]
        num_samples = len(pred)
        class_samples[target] += num_samples
        samples += num_samples
        num_correct = pred.count(target)
        class_correct[target] = num_correct
        correct += num_correct
    
    acc_network = 100.0 * correct / samples

    return acc_network

def plot_wav_predprobas_boxplot(pred_dict, wav_targets_dict, epoch, EXPERIMENT_ARGS):
    SAVE_DIR = utils.check_folder(os.path.join(EXPERIMENT_ARGS, 'test_wav_predprobas'))
    wav_sorted = sorted(wav_targets_dict, key=lambda k: wav_targets_dict[k])
    label_sorted = [wav_targets_dict[key] for key in wav_sorted]
    num_normal_wavs = label_sorted.count(0)
    #wav_mean_predproba = [np.mean(pred_dict[key], axis=0) for key in wav_sorted]
    wav_predproba_abnormal = []
    for key in wav_sorted:
        key_predprobas = pred_dict[key]
        key_predprobas_abnormal = [x[1] for x in key_predprobas]
        wav_predproba_abnormal.append(key_predprobas_abnormal)
    # data for bar plot
    wav_predproba_abnormal_mean = [np.mean(x) for x in wav_predproba_abnormal]
    x_label = [f'{w}_{l}' for w, l in zip(wav_sorted, label_sorted)]
    x_pos = [1*i for i in range(len(x_label))]
    abnormal_threshold = 0.5
    colors = []
    for ab_mean, lbl in zip(wav_predproba_abnormal_mean, label_sorted):
        if lbl == 1 and ab_mean >= abnormal_threshold:
            colors.append('green')
        elif lbl == 0 and ab_mean < abnormal_threshold:
            colors.append('green')
        else:
            colors.append('red')
    fig = plt.figure(figsize = (45, 5))
    plt.bar(x_pos, wav_predproba_abnormal_mean, width = 0.8, color=colors)
    plt.axhline(y=abnormal_threshold, color='k')
    plt.axvline(x=x_pos[num_normal_wavs-1]+0.5, color='k')
    plt.xticks(x_pos, x_label)
    plt.xlabel("Wav")
    plt.xticks(rotation=90)
    plt.ylabel("Mean abnormal prediction probability")
    plt.ylim((0,1))
    plt.xlim((x_pos[0]-4, x_pos[-1]+4))
    plt.title(f'Means for labels: {np.round(np.mean(wav_predproba_abnormal_mean[:num_normal_wavs]), 2)}, {np.round(np.mean(wav_predproba_abnormal_mean[num_normal_wavs:]), 2)}')
    plt.tight_layout()
    FILENAME = os.path.join(SAVE_DIR, f'test_wav_predprobas_{epoch}.jpg')
    plt.savefig(FILENAME, dpi=300)
    plt.close()
    return
