import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import random
import torch_audiomentations
import librosa

class physionet_dataset(Dataset):
    def __init__(self, arguments, dataset, dataset_name, seed_data, num_classes, n_fraction, mode, transform, sample_rate, num_channels, seed, train_balance, method, valid, classical_space):
        self.arguments = arguments
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.seed_data = seed_data
        self.num_classes = num_classes
        self.n_fraction = n_fraction
        self.mode = mode
        self.transform = transform
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.seed = seed
        self.train_balance = train_balance
        self.method = method
        self.valid = valid
        self.classical_space = classical_space

        if self.mode == 'test':
            if self.num_channels==1:
                self.test_data = np.array(dataset['test']['data']['25-400'])
            elif self.num_channels==4:
                test_bands_zip = zip(dataset['test']['data']['25-45'], 
                                dataset['test']['data']['45-80'], 
                                dataset['test']['data']['80-200'], 
                                dataset['test']['data']['200-400'])
                self.test_data = np.array([np.vstack((b1, b2, b3, b4)) for b1, b2, b3, b4 in test_bands_zip])
            self.test_label = np.array(dataset['test']['label'])
            self.test_frames = np.array(dataset['test']['frames'])
            self.test_wav = np.array(dataset['test']['wav'])
            self.test_sig_qual = np.array(dataset['test']['sig_qual'])
        elif self.mode == 'train' or self.mode == 'valid':
            if self.num_channels==1:
                self.train_data = np.array(dataset['train']['data']['25-400'])
            elif self.num_channels==4:
                train_bands_zip = zip(dataset['train']['data']['25-45'], 
                                    dataset['train']['data']['45-80'], 
                                    dataset['train']['data']['80-200'], 
                                    dataset['train']['data']['200-400'])
                self.train_data = np.array([np.vstack((b1, b2, b3, b4)) for b1, b2, b3, b4 in train_bands_zip])
            if self.classical_space:
                train_bands_zip = zip(dataset['train']['data']['25-45'], 
                                    dataset['train']['data']['45-80'], 
                                    dataset['train']['data']['80-200'], 
                                    dataset['train']['data']['200-400'],
                                    dataset['train']['data']['25-400'])
                self.train_data = np.array([np.vstack((b1, b2, b3, b4, b5)) for b1, b2, b3, b4, b5 in train_bands_zip])
            self.train_label = np.array(dataset['train']['label'])
            self.train_frames = np.array(dataset['train']['frames'])
            self.train_wav = np.array(dataset['train']['wav'])
            self.train_sig_qual = np.array(dataset['train']['sig_qual'])
            # remove the noisy recordings (signal quality is 0)
            indices = np.nonzero(self.train_sig_qual)[0]
            self.train_data = self.train_data[indices]
            self.train_label = self.train_label[indices]
            self.train_frames = self.train_frames[indices]
            self.train_wav = self.train_wav[indices]
            self.train_sig_qual = self.train_sig_qual[indices]
            # split the wavs into 12 categories: 2 classes per 6 subsets (a, b, c, d, e, f)
            dataset_map = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5}
            wavs = [[] for i in range(6*self.num_classes)] # 6 datasets with 2 classes each
            wavs_flat = []
            for wav, label in zip(self.train_wav, self.train_label):
                if wav not in wavs_flat:
                    dataset_letter = wav[0]
                    idx = dataset_map[dataset_letter] + 6*label #label*self.num_classes + (1-sig_qual)
                    wavs[idx].append(wav)
                    wavs_flat.append(wav)
            if self.train_balance: # only works for two classes
                # select a representative subset with respect to the 6 datasets and 2 class labels
                max_wavs = [min(len(wavs[i]), len(wavs[i+6])) for i in range(6)]
                max_wavs = max_wavs + max_wavs
                tbal_seed = 18 # hardcoded
                if hasattr(self.arguments, "true_seed"):
                    print(f'True seed for train balance data selection has been changed to {self.arguments.true_seed}')
                    tbal_seed = self.arguments.true_seed
                wavs = [random.Random(tbal_seed).sample(x, maxw) for x, maxw in zip(wavs, max_wavs)]
                wavs_bal_flat = [wav for sublist in wavs for wav in sublist]
                wavs_bal_flat = np.sort(wavs_bal_flat)
                indices = [i for i, wav in enumerate(self.train_wav) if wav in wavs_bal_flat]
                self.train_data = self.train_data[indices]
                self.train_label = self.train_label[indices]
                self.train_frames = self.train_frames[indices]
                self.train_wav = self.train_wav[indices]
                self.train_sig_qual = self.train_sig_qual[indices]
            if n_fraction < 1.0: # only works for two classes
                # balance each label separately
                wavs_nfrac_flat_0 = [wav for sublist in wavs[:6] for wav in sublist]
                wavs_nfrac_flat_1 = [wav for sublist in wavs[6:] for wav in sublist]
                wavs_nfrac_flat_0 = sorted(wavs_nfrac_flat_0)
                wavs_nfrac_flat_1 = sorted(wavs_nfrac_flat_1)
                random.Random(self.seed_data).shuffle(wavs_nfrac_flat_0)
                random.Random(self.seed_data).shuffle(wavs_nfrac_flat_1)
                n_frac_wavs_label = int(np.ceil(self.n_fraction*len(list(set(self.train_wav)))/2))
                wavs_nfrac_flat_0 = wavs_nfrac_flat_0[:n_frac_wavs_label]
                wavs_nfrac_flat_1 = wavs_nfrac_flat_1[:n_frac_wavs_label]
                wavs_nfrac_flat = wavs_nfrac_flat_0 + wavs_nfrac_flat_1
                wavs_nfrac_flat = np.sort(wavs_nfrac_flat)
                indices = [i for i, wav in enumerate(self.train_wav) if wav in wavs_nfrac_flat]
                self.train_data = self.train_data[indices]
                self.train_label = self.train_label[indices]
                self.train_frames = self.train_frames[indices]
                self.train_wav = self.train_wav[indices]
                self.train_sig_qual = self.train_sig_qual[indices]
            if self.valid == True:
                # Create k-folds for the cross-validation approch
                k_folds = 5
                if self.seed not in np.arange(1, k_folds+1, 1):
                    raise Exception(f"Parameter 'self.seed' (was set to {self.seed}) must be in {list(np.arange(1, k_folds+1, 1))} (we are applying {k_folds}-fold-CV)!")
                wavs_flat_0 = []
                wavs_flat_1 = []
                wavs_flat = []
                for wav, label in zip(self.train_wav, self.train_label):
                    if wav not in wavs_flat:
                        if label == 0:
                            wavs_flat_0.append(wav)
                        elif label == 1:
                            wavs_flat_1.append(wav)
                        wavs_flat.append(wav)
                partitions_0 = [wavs_flat_0[i::k_folds] for i in range(k_folds)]
                partitions_1 = [wavs_flat_1[i::k_folds] for i in range(k_folds)]
                folds_holder = []
                for i in range(k_folds):
                    folds_holder.append(partitions_0[i] + partitions_1[k_folds-i-1]) #i
                print(folds_holder)
                # create test data from validation indices
                wavs_valid = folds_holder[self.seed-1]
                indices_valid = [i for i, wav in enumerate(self.train_wav) if wav in wavs_valid]
                self.test_data = self.train_data[indices_valid]
                self.test_label = self.train_label[indices_valid]
                self.test_frames = self.train_frames[indices_valid]
                self.test_wav = self.train_wav[indices_valid]
                self.test_sig_qual = self.train_sig_qual[indices_valid]
                # take validation indices from train data
                wavs_train = [w for fold in folds_holder for w in fold if w not in wavs_valid]
                indices = [i for i, wav in enumerate(self.train_wav) if wav in wavs_train]
                self.train_data = self.train_data[indices]
                self.train_label = self.train_label[indices]
                self.train_frames = self.train_frames[indices]
                self.train_wav = self.train_wav[indices]
                self.train_sig_qual = self.train_sig_qual[indices]

    def __getitem__(self, index):
        if self.mode == 'train':
            signal, target, frames, wav, sig_qual = self.train_data[index], self.train_label[index], self.train_frames[index], self.train_wav[index], self.train_sig_qual[index]
            signal = torch.tensor(signal)
            if self.num_channels in [0, 1]:
                signal = signal[None, :]
            signal = signal[None, :] # add batch dimension for transformation
            signal = self.transform(signal, sample_rate=self.sample_rate)
            signal = torch.squeeze(signal, dim=0) # removes batch dimension
            return signal, target, frames, wav, sig_qual, index
        elif self.mode == 'test':
            signal, target, frames, wav, sig_qual = self.test_data[index], self.test_label[index], self.test_frames[index], self.test_wav[index], self.test_sig_qual[index]
            signal = torch.tensor(signal)
            if self.num_channels in [0, 1]:
                signal = signal[None, :]
            return signal, target, frames, wav, sig_qual, index
        elif self.mode == 'valid':
            signal, target, frames, wav, sig_qual = self.test_data[index], self.test_label[index], self.test_frames[index], self.test_wav[index], self.test_sig_qual[index]
            signal = torch.tensor(signal)
            if self.num_channels in [0, 1]:
                signal = signal[None, :]
            return signal, target, frames, wav, sig_qual, index

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        elif self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'valid':
            return len(self.test_data)

class physionet_dataloader():
    def __init__(self, args, dataset):
        self.arguments = args
        self.dataset = dataset
        self.dataset_name = args.dataset
        self.seed_data = args.seed_data
        self.n_fraction = args.n_fraction
        self.batch_size = args.batch_size  
        self.num_classes = args.num_classes
        self.sample_rate = args.sample_rate
        self.num_channels = args.num_channels
        self.seed = args.seed
        self.train_balance = args.train_balance
        self.method = args.method
        self.valid = args.valid
        self.classical_space = args.classical_space

        # Set for future updates, currently no pytorch augmentation
        self.transform_train = torch_audiomentations.Compose(
            transforms=[torch_audiomentations.Identity()
                        ])

    def run(self, mode, transform_seed):
        if mode == 'train':
            train_dataset = physionet_dataset(arguments = self.arguments,
                                              dataset = self.dataset,
                                              dataset_name = self.dataset_name,
                                              seed_data = self.seed_data,
                                              num_classes = self.num_classes,
                                              n_fraction = self.n_fraction, 
                                              mode = 'train',
                                              transform = self.transform_train,
                                              sample_rate = self.sample_rate,
                                              num_channels = self.num_channels,
                                              seed = self.seed,
                                              train_balance = self.train_balance,
                                              method = self.method,
                                              valid = self.valid,
                                              classical_space = self.classical_space)                       
            random.seed(transform_seed)
            torch.manual_seed(transform_seed)

            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last = True)
                                        
            return train_loader, np.asarray(train_dataset.train_label)

        elif mode == 'test':
            test_dataset = physionet_dataset(arguments = self.arguments,
                                             dataset = self.dataset,
                                             dataset_name = self.dataset_name,
                                             seed_data = self.seed_data,
                                             num_classes = self.num_classes,
                                             n_fraction = None, 
                                             mode = 'test',
                                             transform = None,
                                             sample_rate = None,
                                             num_channels = self.num_channels,
                                             seed = None,
                                             train_balance = None,
                                             method = self.method,
                                             valid = None,
                                             classical_space = False)   
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=1000, # manually set to higher value to make it go faster
                                     shuffle=False,
                                     drop_last = False)  
            return test_loader

        elif mode == 'valid':
            valid_dataset = physionet_dataset(arguments = self.arguments,
                                              dataset = self.dataset,
                                              dataset_name = self.dataset_name,
                                              seed_data = self.seed_data,
                                              num_classes = self.num_classes,
                                              n_fraction = self.n_fraction, 
                                              mode = 'valid',
                                              transform = None,
                                              sample_rate = None,
                                              num_channels = self.num_channels,
                                              seed = self.seed,
                                              train_balance = self.train_balance,
                                              method = self.method,
                                              valid = True,
                                              classical_space = False)   
            valid_loader = DataLoader(dataset=valid_dataset,
                                      batch_size=1000, # manually set to higher value to make it go faster
                                      shuffle=False,
                                      drop_last = False)
            return valid_loader


                
            
            