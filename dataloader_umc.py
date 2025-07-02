import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import random
import torch_audiomentations
import librosa

class umc_dataset(Dataset):
    def __init__(self, dataset, dataset_name, seed_data, num_classes, n_fraction, mode, transform, sample_rate, num_channels, seed, train_balance, method, valid, classical_space):
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
        
        if self.num_channels==1:
            self.data = np.array(dataset['data']['25-400'])
        elif self.num_channels==4:
            bands_zip = zip(dataset['data']['25-45'], 
                            dataset['data']['45-80'], 
                            dataset['data']['80-200'], 
                            dataset['data']['200-400'])
            self.data = np.array([np.vstack((b1, b2, b3, b4)) for b1, b2, b3, b4 in bands_zip])
        if self.classical_space:
            bands_zip = zip(dataset['data']['25-45'], 
                            dataset['data']['45-80'], 
                            dataset['data']['80-200'], 
                            dataset['data']['200-400'],
                            dataset['data']['25-400'])
            self.data = np.array([np.vstack((b1, b2, b3, b4, b5)) for b1, b2, b3, b4, b5 in bands_zip])
        self.label = np.array(dataset['label'])
        self.label = np.where((self.label==0)|(self.label==1), self.label^1, self.label) # swap classes to rekomp=0, dekomp=1
        self.frames = np.array(dataset['frames'])
        self.wav = np.array(dataset['wav'])
        self.sig_qual = np.array(dataset['sig_qual'])
        self.id = np.array(dataset['id'])
        self.excluded = np.array(dataset['excluded'])
        # only keep those that have excluded==1 as those are NOT bad (21 has same recs for dekomp and rekomp, 17 and 18 only have one class)
        indices_exc = [i for i, ex in enumerate(self.excluded) if ex==1]
        self.data = self.data[indices_exc]
        self.label = self.label[indices_exc]
        self.frames = self.frames[indices_exc]
        self.wav = self.wav[indices_exc]
        self.sig_qual = self.sig_qual[indices_exc]
        self.id = self.id[indices_exc]
        self.excluded = self.excluded[indices_exc]

        # folds are hardcoded as we want to have the same as in the classical method published in previous paper #note: 002 has 2 dekomp recordings
        # 10-fold cross-validation
        #""" 
        if self.seed_data not in np.arange(1, 11, 1):
            raise Exception(f"Parameter 'self.seed' (was set to {self.seed_data}) must be in {list(np.arange(1, 11, 1))} (we are applying {10}-fold-CV)!")
        self.folds = [['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_010', 'ID_015', 'ID_5', 'ID_20', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19'], 
                      ['ID_005', 'ID_006', 'ID_6', 'ID_13', 'ID_012', 'ID_011', 'ID_7', 'ID_24', 'ID_009', 'ID_001', 'ID_8', 'ID_4', 'ID_014', 'ID_004', 'ID_23', 'ID_14', 'ID_003', 'ID_007', 'ID_12', 'ID_11', 'ID_000', 'ID_15', 'ID_3', 'ID_008', 'ID_22', 'ID_10', 'ID_013', 'ID_9', 'ID_16', 'ID_002', 'ID_2', 'ID_1', 'ID_19']]
        """
        # 5-fold cross-validation
        if self.seed_data not in np.arange(1, 6, 1):
            raise Exception(f"Parameter 'self.seed' (was set to {self.seed_data}) must be in {list(np.arange(1, 6, 1))} (we are applying {5}-fold-CV)!")
        self.folds = [['ID_010', 'ID_003', 'ID_015', 'ID_007', 'ID_12', 'ID_5', 'ID_11', 'ID_20', 'ID_005', 'ID_000', 'ID_006', 'ID_15', 'ID_6', 'ID_3', 'ID_13', 'ID_012', 'ID_008', 'ID_011', 'ID_22', 'ID_7', 'ID_10', 'ID_24', 'ID_009', 'ID_013', 'ID_001', 'ID_9', 'ID_8', 'ID_16', 'ID_4'], 
                     ['ID_010', 'ID_003', 'ID_015', 'ID_007', 'ID_12', 'ID_5', 'ID_11', 'ID_20', 'ID_005', 'ID_000', 'ID_006', 'ID_15', 'ID_6', 'ID_3', 'ID_13', 'ID_012', 'ID_008', 'ID_011', 'ID_22', 'ID_7', 'ID_10', 'ID_24', 'ID_014', 'ID_002', 'ID_004', 'ID_2', 'ID_23', 'ID_1', 'ID_14', 'ID_19'], 
                     ['ID_010', 'ID_003', 'ID_015', 'ID_007', 'ID_12', 'ID_5', 'ID_11', 'ID_20', 'ID_005', 'ID_000', 'ID_006', 'ID_15', 'ID_6', 'ID_3', 'ID_13', 'ID_009', 'ID_013', 'ID_001', 'ID_9', 'ID_8', 'ID_16', 'ID_4', 'ID_014', 'ID_002', 'ID_004', 'ID_2', 'ID_23', 'ID_1', 'ID_14', 'ID_19'], 
                     ['ID_010', 'ID_003', 'ID_015', 'ID_007', 'ID_12', 'ID_5', 'ID_11', 'ID_20', 'ID_012', 'ID_008', 'ID_011', 'ID_22', 'ID_7', 'ID_10', 'ID_24', 'ID_009', 'ID_013', 'ID_001', 'ID_9', 'ID_8', 'ID_16', 'ID_4', 'ID_014', 'ID_002', 'ID_004', 'ID_2', 'ID_23', 'ID_1', 'ID_14', 'ID_19'], 
                     ['ID_005', 'ID_000', 'ID_006', 'ID_15', 'ID_6', 'ID_3', 'ID_13', 'ID_012', 'ID_008', 'ID_011', 'ID_22', 'ID_7', 'ID_10', 'ID_24', 'ID_009', 'ID_013', 'ID_001', 'ID_9', 'ID_8', 'ID_16', 'ID_4', 'ID_014', 'ID_002', 'ID_004', 'ID_2', 'ID_23', 'ID_1', 'ID_14', 'ID_19']]
        """""
        self.selected_fold = self.folds[self.seed_data-1]
        self.unique_ids = set(list(self.id))
        #print(f'{self.unique_ids=}')
        
        if self.mode == 'test':
            indices_test = [i for i, id in enumerate(self.id) if id not in self.selected_fold] 
            self.test_data = self.data[indices_test]
            #self.test_data = self.test_data[:4, :] # if classical is True, reduce test to 4 channels
            self.test_label = self.label[indices_test]
            self.test_frames = self.frames[indices_test]
            self.test_wav = self.wav[indices_test]
            self.test_sig_qual = self.sig_qual[indices_test]
        elif self.mode == 'train' or self.mode == 'valid':
            indices_train = [i for i, id in enumerate(self.id) if id in self.selected_fold] 
            self.train_data = self.data[indices_train]
            self.train_label = self.label[indices_train]
            self.train_frames = self.frames[indices_train]
            self.train_wav = self.wav[indices_train]
            self.train_sig_qual = self.sig_qual[indices_train]
            self.train_id = self.id[indices_train]
            # remove the noisy recordings (signal quality is 0) # badly segmented: 'ID_12', 'ID_14', 'ID_24', 'ID_004', 'ID_007', 'ID_013', 'ID_3'
            indices = np.nonzero(self.train_sig_qual)[0]
            self.train_data = self.train_data[indices]
            self.train_label = self.train_label[indices]
            self.train_frames = self.train_frames[indices]
            self.train_wav = self.train_wav[indices]
            self.train_sig_qual = self.train_sig_qual[indices]
            self.train_id = self.train_id[indices]
            # split the wavs into 2 categories: umc_new, umc_old
            dataset_map = {'old':0, 'new':1}
            ids = [[] for i in range(2)] # each patient is already class-balanced
            ids_flat = []
            for id in self.train_id:
                if id not in ids_flat:
                    if len(id) == 6: # new have 3 digit ID thus the whole name is 6 characters long
                        dataset_letter = 'new'
                    elif len(id) < 6: # old have either 2 or 1 digit ID
                        dataset_letter = 'old'
                    idx = dataset_map[dataset_letter]
                    ids[idx].append(id)
                    ids_flat.append(id)
            if self.valid == True:
                # Create three folds for the cross-validation approch
                k_folds = 3
                if self.seed not in np.arange(1, k_folds+1, 1):
                    raise Exception(f"Parameter 'self.seed' (was set to {self.seed}) must be in {list(np.arange(1, k_folds+1, 1))} (we are applying {k_folds}-fold-CV)!")
                partitions_old = [ids[0][i::k_folds] for i in range(k_folds)]
                partitions_new = [ids[1][i::k_folds] for i in range(k_folds)]
                folds_holder = []
                for i in range(k_folds):
                    folds_holder.append(partitions_old[i] + partitions_new[k_folds-i-1]) #i
                print(folds_holder)
                # create test data from validation indices
                ids_valid = folds_holder[self.seed-1]
                indices_valid = [i for i, id in enumerate(self.train_id) if id in ids_valid]
                self.test_data = self.train_data[indices_valid]
                self.test_label = self.train_label[indices_valid]
                self.test_frames = self.train_frames[indices_valid]
                self.test_wav = self.train_wav[indices_valid]
                self.test_sig_qual = self.train_sig_qual[indices_valid]
                # take validation indices from train data
                ids_train = [w for fold in folds_holder for w in fold if w not in ids_valid]
                indices = [i for i, id in enumerate(self.train_id) if id in ids_train]
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
        elif self.mode == 'test' or self.mode == 'valid':
            signal, target, frames, wav, sig_qual = self.test_data[index], self.test_label[index], self.test_frames[index], self.test_wav[index], self.test_sig_qual[index]
            signal = torch.tensor(signal)
            if self.num_channels in [0, 1]:
                signal = signal[None, :]
            return signal, target, frames, wav, sig_qual, index

    def __len__(self):
        if self.mode == 'test' or self.mode == 'valid':
            return len(self.test_data)
        elif self.mode == 'train':
            return len(self.train_data)

class umc_dataloader():
    def __init__(self, args, dataset):
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

        self.transform_train = torch_audiomentations.Compose(
            transforms=[torch_audiomentations.Identity()
                        ])

    def run(self, mode, transform_seed):
        if mode == 'train':
            train_dataset = umc_dataset(dataset = self.dataset,
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
                                      #generator = torch.Generator(device='cpu'), # this line prevents changing the shuffle random when iterating
                                      drop_last = True)
                                        
            return train_loader, np.asarray(train_dataset.train_label)

        elif mode == 'test':
            test_dataset = umc_dataset(dataset = self.dataset,
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
            valid_dataset = umc_dataset(dataset = self.dataset,
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