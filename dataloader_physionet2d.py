import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import random
import torch_audiomentations
import librosa

class physionet_dataset(Dataset):
    def __init__(self, dataset, dataset_name, seed_data, num_classes, n_fraction, mode, seed, method, valid):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.seed_data = seed_data
        self.num_classes = num_classes
        self.n_fraction = n_fraction
        self.mode = mode
        self.seed = seed
        self.method = method
        self.valid = valid

        if self.mode == 'test':
            self.test_data = np.array(dataset['test']['data'])
            self.test_label = np.array(dataset['test']['label'])
            self.test_frames = np.array(dataset['test']['frames'])
            self.test_wav = np.array(dataset['test']['wav'])
            self.test_sig_qual = np.array(dataset['test']['sig_qual'])
        elif self.mode == 'train' or self.mode == 'valid':
            self.train_data = np.array(dataset['train']['data'])
            self.train_label = np.array(dataset['train']['label'])
            self.train_frames = np.array(dataset['train']['frames'])
            self.train_wav = np.array(dataset['train']['wav'])
            self.train_sig_qual = np.array(dataset['train']['sig_qual'])
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
            spectrogram, target, frames, wav, sig_qual = self.train_data[index], self.train_label[index], self.train_frames[index], self.train_wav[index], self.train_sig_qual[index]
            spectrogram = torch.tensor(spectrogram)
            spectrogram = spectrogram[None, :] # add channel dimensions
            return spectrogram, target, frames, wav, sig_qual, index
        elif self.mode == 'test':
            spectrogram, target, frames, wav, sig_qual = self.test_data[index], self.test_label[index], self.test_frames[index], self.test_wav[index], self.test_sig_qual[index]
            spectrogram = torch.tensor(spectrogram)
            spectrogram = spectrogram[None, :] # add channel dimensions
            return spectrogram, target, frames, wav, sig_qual, index
        elif self.mode == 'valid':
            spectrogram, target, frames, wav, sig_qual = self.test_data[index], self.test_label[index], self.test_frames[index], self.test_wav[index], self.test_sig_qual[index]
            spectrogram = torch.tensor(spectrogram)
            spectrogram = spectrogram[None, :] # add channel dimensions
            return spectrogram, target, frames, wav, sig_qual, index

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        elif self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'valid':
            return len(self.test_data)

class physionet_dataloader():
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.dataset_name = args.dataset
        self.seed_data = args.seed_data
        self.n_fraction = args.n_fraction
        self.batch_size = args.batch_size  
        self.num_classes = args.num_classes
        self.num_channels = args.num_channels
        self.seed = args.seed
        self.method = args.method
        self.valid = args.valid

    def run(self, mode, transform_seed):
        if mode == 'train':
            train_dataset = physionet_dataset(dataset = self.dataset,
                                              dataset_name = self.dataset_name,
                                              seed_data = self.seed_data,
                                              num_classes = self.num_classes,
                                              n_fraction = self.n_fraction, 
                                              mode = 'train',
                                              seed = self.seed,
                                              method = self.method,
                                              valid = self.valid)                       
            random.seed(transform_seed)
            torch.manual_seed(transform_seed)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      #generator = torch.Generator(device='cpu'), # this line prevents changing the shuffle random when iterating
                                      drop_last = True)
                                        
            return train_loader, np.asarray(train_dataset.train_label)

        elif mode == 'test':
            test_dataset = physionet_dataset(dataset = self.dataset,
                                             dataset_name = self.dataset_name,
                                             seed_data = self.seed_data,
                                             num_classes = self.num_classes,
                                             n_fraction = None, 
                                             mode = 'test',
                                             seed = None,
                                             method = self.method,
                                             valid = None)   
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=256, # manually set to higher value to make it go faster
                                     shuffle=False,
                                     drop_last = False)  
            return test_loader

        elif mode == 'valid':
            valid_dataset = physionet_dataset(dataset = self.dataset,
                                              dataset_name = self.dataset_name,
                                              seed_data = self.seed_data,
                                              num_classes = self.num_classes,
                                              n_fraction = self.n_fraction, 
                                              mode = 'valid',
                                              seed = self.seed,
                                              method = self.method,
                                              valid = True)   
            valid_loader = DataLoader(dataset=valid_dataset,
                                      batch_size=256, # manually set to higher value to make it go faster
                                      shuffle=False,
                                      drop_last = False)
            return valid_loader


                
            
            