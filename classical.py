import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',size = 12)
import numpy as np
import scipy
from scipy import signal
from scipy import stats
from scipy.signal import hilbert, chirp
import random
import warnings
import glob
import librosa
import pywt
import soundfile
import shutil
import matplotlib.pyplot as plt
import antropy as ent
import librosa
from scipy.interpolate import interp1d
import pydub
import scipy.special
import itertools
import math
import collections
from functools import reduce

from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import average_precision_score, precision_recall_curve

import utils
    
def extract_features_python(dataset, SAVE_DIR, start_counter=0):
    # Used for extracting features using Python (Librosa...)
    print('EXTRACTING PYTHON FEATURES')        
    features = pd.DataFrame(dtype=float)
    
    import time
    import utils
    time_start = time.time()
    counter = 0
    problematic = [15762] # some instances cause errors because of very short cycle stages
    for split in ['train', 'test']:
        data = dataset[split]['data']['25-400']
        label = dataset[split]['label']
        frames = dataset[split]['frames']
        wav = dataset[split]['wav']
        sig_qual = dataset[split]['sig_qual']
        segment = 0
        w_last = ''
        for d, l, f, w, s in zip(data, label, frames, wav, sig_qual):
            counter += 1 
            if w == w_last:
                segment = segment + 1
            else:
                segment = 0
            w_last = w
            if counter in problematic or counter < start_counter:
                continue
            print(f'{counter}: {w} segment {segment} ({split} split); timer: {utils.timer(time_start, time.time())}')
            vec = feature_vector_seg(d, l, f, w, s, segment, split)
            features = pd.concat([features, vec], axis=1)
            # save every nth step
            if counter % 2000 == 0 and counter != start_counter:
                features_fin = features.T.reset_index(drop=True)    
                print(f'features shape: {features_fin.shape}')
                # Name of the resulting features file
                FEATURES = f'features_{start_counter}-{counter}.csv'
                FEATURES = os.path.join(SAVE_DIR, FEATURES)
                # Save the features
                features_fin.to_csv(FEATURES, index = False)
                print(f'Features saved to: {FEATURES}\n')  

    # Save the whole dataset      
    features_fin = features.T.reset_index(drop=True)    
    print(f'features shape: {features_fin.shape}')
    # Name of the resulting features file
    FEATURES = f'features.csv'
    FEATURES = os.path.join(SAVE_DIR, FEATURES)
    # Save the features
    features_fin.to_csv(FEATURES, index = False)
    print(f'Features saved to: {FEATURES}\n')
    
    return

def remove_segments_meanEnvelope(FEATURES, SAVE_DIR, std_factor = 1.4, print_steps = False):
    # Used for removing segments if the mean of their envelope is too high/low compared to other segments of the same recording
    print('REMOVING SEGMENTS BASED ON THEIR MEAN ENVELOPE AND THE MEAN ENVELOPE OF ALL SEGMENTS IN THE SAME RECORDING')
    print(f'\tSegments with mean envelope outside mean +- SD*{std_factor} region will be removed.')
    features = pd.read_csv(FEATURES)
    print(f'\tFeature file {FEATURES} has been loaded')
    new_features = pd.DataFrame()
    discarded_percentage_arr = []
    for i, recording in enumerate(features['wav'].unique()):
        print(f'{i+1}: {recording}')
        features_rec = features[(features['wav'] == recording)]
        mean_seg_envelope = features_rec['MeanEnv_RR'].mean()
        std_seg_envelope = features_rec['MeanEnv_RR'].std()
        mask = (features_rec['MeanEnv_RR'] < (mean_seg_envelope-std_seg_envelope*std_factor)) | (features_rec['MeanEnv_RR'] > (mean_seg_envelope+std_seg_envelope*std_factor))
        discarded = features_rec[mask]
        kept = features_rec[~mask]
        discarded_percentage = np.round(len(discarded)/len(features_rec)*100, 1)
        discarded_percentage_arr.append(discarded_percentage)
        new_features = pd.concat([new_features, kept])
        if print_steps:
            discarded_list = discarded['segment'].tolist()
            print(f'\t\t{recording}, discarded = {len(discarded)} ({discarded_percentage} %), Segments: {discarded_list}') 
    print(f'\tMean percentage of discarded segments: {np.round(np.mean(discarded_percentage_arr), 1)}+-{np.round(np.std(discarded_percentage_arr), 1)}%')
    
    # Save the new features
    FEATURES_NEW = os.path.basename(FEATURES).split('.csv')[0] + f'_sigma{std_factor}' + '.csv'
    FEATURES_NEW = os.path.join(SAVE_DIR, FEATURES_NEW)
    new_features.to_csv(FEATURES_NEW, index = False)
    print(f'\tFeatures saved to: {FEATURES_NEW}\n')
    print()
    
    return
    
def aggregate_features_into_rec_mean_and_sd(FEATURES, SAVE_DIR, window = 2):
    print('AGGREGATING FEATURES INTO MEAN AND SD OF EACH RECORDING')
    features = pd.read_csv(FEATURES)
    print(f'\tFeature file {FEATURES} has been loaded')
    # Aggregate the features into recordings' mean and standard deviation features
    features = get_mean_and_SD_of_seg_features_window(features, window)
        
    # Name of the resulting features file
    FEATURES_NEW = os.path.basename(FEATURES).split('.csv')[0] + f'_AggW{window}.csv'
        
    print(f'features shape: {features.shape}')
    # Save the features
    FEATURES_NEW = os.path.join(SAVE_DIR, FEATURES_NEW)
    features.to_csv(FEATURES_NEW, index = False)
    print(f'Features saved to: {FEATURES_NEW}\n')
    return

def get_mean_and_SD_of_seg_features_window(features, window):
    features_cols = [x for x in features.columns if x not in ['class', 'wav', 'segment', 'sig_qual', 'split']]
    new_features = pd.DataFrame()
    for i, recording in enumerate(features['wav'].unique()):
        print(f'{i+1}: {recording}')
        features_rec = features[features['wav']==recording]
        features_rec_len = len(features_rec)
        # Drop inf values 
        features_rec.replace([-np.inf, np.inf], np.nan, inplace=True)
        # Fill all missing values with mean from columns
        features_rec_mean = features_rec.mean()
        features_rec.fillna(features_rec_mean, inplace=True)
        if window >= features_rec_len:
            print(f'\tCUSTOM WARNING: window size {window} is not smaller than the number of segments {features_rec_len}!')
        features_rec = features_rec.astype({'segment': int})
        features_rec = features_rec.sort_values(by = 'segment')
        # multiply the features of each recording, so that the sliding window is cyclic
        req_rec_len = features_rec_len + window
        mult_f = int(np.ceil(req_rec_len/features_rec_len))
        features_rec_multiplied = pd.concat([features_rec for i in range(mult_f)])
        features_rec_multiplied_copy = features_rec_multiplied.copy() # This is here so that we avoid the notorius SettingWithCopyWarning 
        import warnings
        warnings.filterwarnings("ignore")
        for col in features_cols:
            features_rec_multiplied_copy.loc[:, f'm_{col}'] = features_rec_multiplied_copy[col].rolling(window = window).mean()
            features_rec_multiplied_copy.loc[:, f'sd_{col}'] = features_rec_multiplied_copy[col].rolling(window = window).std()
        features_rec_multiplied_copy = features_rec_multiplied_copy.iloc[-features_rec_len:]
        # Sanity check: print if anyting went wrong
        if (features_rec_multiplied_copy.isna().any().any()):
            print(f'Rolling produced zeros in recording: {recording}')
            display(features_rec_multiplied_copy)
            for col in features_rec_multiplied_copy:
                if features_rec_multiplied_copy[col].isna().sum() !=0:
                    print(col, features_rec_multiplied_copy[col].isna().sum())            
        new_features = pd.concat([new_features, features_rec_multiplied_copy])
    return new_features   

def aggregate_features_into_rec_mean_and_sd_single_vector(FEATURES, SAVE_DIR):
    print('AGGREGATING FEATURES INTO MEAN AND SD OF EACH RECORDING')
    features = pd.read_csv(FEATURES)
    print(f'\tFeature file {FEATURES} has been loaded')
    # Aggregate the features into recordings' mean and standard deviation features
    features_cols = [x for x in features.columns if x not in ['class', 'wav', 'segment', 'sig_qual', 'split']]
    new_features = pd.DataFrame()
    for i, recording in enumerate(features['wav'].unique()):
        print(f'{i+1}: {recording}')
        features_rec = features[features['wav']==recording]
        features_rec_len = len(features_rec)
        # Drop inf values 
        features_rec.replace([-np.inf, np.inf], np.nan, inplace=True)
        # Fill all missing values with mean from columns
        features_rec_mean = features_rec.mean()
        features_rec.fillna(features_rec_mean, inplace=True)
        features_rec = features_rec.astype({'segment': int})
        features_rec = features_rec.sort_values(by = 'segment')
        # multiply the features of each recording, so that the sliding window is cyclic
        features_rec_copy = features_rec.copy() # This is here so that we avoid the notorius SettingWithCopyWarning 
        import warnings
        warnings.filterwarnings("ignore")
        for col in features_cols:
            features_rec_copy.loc[:, f'm_{col}'] = features_rec_copy[col].mean()
            features_rec_copy.loc[:, f'sd_{col}'] = features_rec_copy[col].std()
        features_rec_copy = features_rec_copy.iloc[-features_rec_len:]
        # Sanity check: print if anyting went wrong
        if (features_rec_copy.isna().any().any()):
            print(f'Rolling produced zeros in recording: {recording}')
            display(features_rec_copy)
            for col in features_rec_copy:
                if features_rec_copy[col].isna().sum() !=0:
                    print(col, features_rec_copy[col].isna().sum())            
        new_features = pd.concat([new_features, features_rec_copy.head(1)])
    # Name of the resulting features file
    FEATURES_NEW = os.path.basename(FEATURES).split('.csv')[0] + f'_AggSingle.csv'
    print(f'features shape: {new_features.shape}')
    # Save the features
    FEATURES_NEW = os.path.join(SAVE_DIR, FEATURES_NEW)
    new_features.to_csv(FEATURES_NEW, index = False)
    print(f'Features saved to: {FEATURES_NEW}\n')
    return
    
def feature_vector_seg(data, label, frames, wav, sig_qual, segment, split):
    # Used for extracting a feature vector from a segment   
    Fs = 1000
    # Split the segment into 4 heart sounds
    RR = data[:frames[-1]]
    S1 = data[:frames[1]]
    systole = data[frames[1]:frames[2]]
    S2 = data[frames[2]:frames[3]]
    diastole = data[frames[3]:frames[4]]

    # CALCULATE THE FEATURES
    # RR interval in miliseconds based on only this heart beat
    duration_RR = int(len(RR)*1000/Fs)
    # BPM based on only this heart beat
    BPM = round(60000/duration_RR, 4)
    # Duration of S1 in miliseconds
    duration_S1 = int(len(S1)*1000/Fs)
    # Duration of systole in miliseconds
    duration_systole = int(len(systole)*1000/Fs)
    # Duration of S2 in miliseconds
    duration_S2 = int(len(S2)*1000/Fs)
    # Duration of diastole in miliseconds
    duration_diastole = int(len(diastole)*1000/Fs)
    # Ration between the duration of the S1 and the S2 period
    duration_ratio_S1_S2 = round(duration_S1/duration_S2, 4)
    # Ration between the duration of the systole and the diastole period
    duration_ratio_systole_diastole = round(duration_systole/duration_diastole, 4)
    # Ration between the duration of the S1 and the RR period
    duration_ratio_S1_RR = round(duration_S1/duration_RR, 4)
    # Ration between the duration of the systole and the RR period
    duration_ratio_systole_RR = round(duration_systole/duration_RR, 4)
    # Ration between the duration of the S2 and the RR period
    duration_ratio_S2_RR = round(duration_S2/duration_RR, 4)
    # Ration between the duration of the diastole and the RR period
    duration_ratio_diastole_RR = round(duration_diastole/duration_RR, 4)
    # Ration between the duration of the systole and the S1 period
    duration_ratio_systole_S1 = round(duration_systole/duration_S1, 4)
    # Ration between the duration of the diastole and the S2 period
    duration_ratio_diastole_S2 = round(duration_diastole/duration_S2, 4)

    # max amplitude of S1
    max_amplitude_S1 = np.max(S1)
    # max amplitude of systole
    max_amplitude_systole = np.max(systole)
    # max amplitude of S2
    max_amplitude_S2 = np.max(S2)
    # max amplitude of diastole
    max_amplitude_diastole = np.max(diastole)
    # Ratio between the max amplitude of the S1 and S2 
    max_amplitude_ratio_S1_S2 = round(max_amplitude_S1/max_amplitude_S2, 4)
    # Ratio between the max amplitude of the systole and diastole
    max_amplitude_ratio_systole_diastole = round(max_amplitude_systole/max_amplitude_diastole, 4)
    # Ratio between the max amplitude of the systole and S1 
    max_amplitude_ratio_systole_S1 = round(max_amplitude_systole/max_amplitude_S1, 4)
    # Ratio between the max amplitude of the systole and S2 
    max_amplitude_ratio_systole_S2 = round(max_amplitude_systole/max_amplitude_S2, 4)
    # Ratio between the max amplitude of the diastole and S1 
    max_amplitude_ratio_diastole_S1 = round(max_amplitude_diastole/max_amplitude_S1, 4)
    # Ratio between the max amplitude of the diastole and S2 
    max_amplitude_ratio_diastole_S2 = round(max_amplitude_diastole/max_amplitude_S2, 4)

    # Envelope integral of S1
    analytic_signal = hilbert(S1)
    amplitude_envelope_S1 = np.abs(analytic_signal)
    envelope_integral_S1 = np.trapz(amplitude_envelope_S1, dx=5)
    # Envelope integral of systole
    analytic_signal = hilbert(systole)
    amplitude_envelope_systole = np.abs(analytic_signal)
    envelope_integral_systole = np.trapz(amplitude_envelope_systole, dx=5)
    # Envelope integral of S2
    analytic_signal = hilbert(S2)
    amplitude_envelope_S2 = np.abs(analytic_signal)
    envelope_integral_S2 = np.trapz(amplitude_envelope_S2, dx=5)
    # Envelope integral of diastole
    analytic_signal = hilbert(diastole)
    amplitude_envelope_diastole = np.abs(analytic_signal)
    envelope_integral_diastole = np.trapz(amplitude_envelope_diastole, dx=5)
    # Envelope integral of RR interval
    analytic_signal = hilbert(RR)
    amplitude_envelope_RR = np.abs(analytic_signal)
    envelope_integral_RR = np.trapz(amplitude_envelope_RR, dx=5)
    # Ratio between the envelope integral of S1 and S2
    envelope_integral_ratio_S1_S2 = round(envelope_integral_S1/envelope_integral_S2, 4)
    # Ratio between the envelope integral of systole and diastole
    envelope_integral_ratio_systole_diastole = round(envelope_integral_systole/envelope_integral_diastole, 4)
    # Ratio between the envelope integral of S1 and RR
    envelope_integral_ratio_S1_RR = round(envelope_integral_S1/envelope_integral_RR, 4)
    # Ratio between the envelope integral of systole and RR
    envelope_integral_ratio_systole_RR = round(envelope_integral_systole/envelope_integral_RR, 4)
    # Ratio between the envelope integral of S2 and RR
    envelope_integral_ratio_S2_RR = round(envelope_integral_S2/envelope_integral_RR, 4)
    # Ratio between the envelope integral of diastole and RR
    envelope_integral_ratio_diastole_RR = round(envelope_integral_diastole/envelope_integral_RR, 4)
    # Ratio between the envelope integral of systole and S1
    envelope_integral_ratio_systole_S1 = round(envelope_integral_systole/envelope_integral_S1, 4)
    # Ratio between the envelope integral of diastole and S2
    envelope_integral_ratio_diastole_S2 = round(envelope_integral_diastole/envelope_integral_S2, 4)
    # Mean of the envelopes
    mean_envelope_S1 = np.mean(amplitude_envelope_S1)
    mean_envelope_systole = np.mean(amplitude_envelope_systole)
    mean_envelope_S2 = np.mean(amplitude_envelope_S2)
    mean_envelope_diastole = np.mean(amplitude_envelope_diastole)
    mean_envelope_RR = np.mean(amplitude_envelope_RR)
    # Mean envelope ratios
    mean_envelope_ratio_S1_RR = mean_envelope_S1/mean_envelope_RR
    mean_envelope_ratio_systole_RR = mean_envelope_systole/mean_envelope_RR
    mean_envelope_ratio_S2_RR = mean_envelope_S2/mean_envelope_RR
    mean_envelope_ratio_diastole_RR = mean_envelope_diastole/mean_envelope_RR
    mean_envelope_ratio_systole_diastole = mean_envelope_systole/mean_envelope_diastole
    mean_envelope_ratio_systole_S1 = mean_envelope_systole/mean_envelope_S1
    mean_envelope_ratio_diastole_S2 = mean_envelope_diastole/mean_envelope_S2
    mean_envelope_ratio_S1_S2 = mean_envelope_S1/mean_envelope_S2

    # Power spectral density for different lower- and upper-bounds frequency ranges (RR):
    freqs, psd_RR = signal.welch(RR, Fs)
    analytic_signal_RR = hilbert(psd_RR)
    amplitude_envelope_RR = np.abs(analytic_signal_RR)
    evelope_integral_psd_RR = np.trapz(amplitude_envelope_RR, dx=5)
    psd_RR_normalized = psd_RR/evelope_integral_psd_RR
    # Mean power spectral density for all frequencies
    mean_psd_RR = np.mean(psd_RR)
    mean_psd_RR_normalized = np.mean(psd_RR_normalized)
    # Mean power spectral density for 25-40 Hz frequency range
    psd_RR_25_40_hz = np.where( (25 <=freqs) & (freqs<= 40), psd_RR, np.nan)
    psd_RR_25_40_hz = psd_RR_25_40_hz[~np.isnan(psd_RR_25_40_hz)]
    mean_psd_RR_25_40_hz = np.mean(psd_RR_25_40_hz)
    psd_RR_normalized_25_40_hz = np.where( (25 <=freqs) & (freqs<= 40), psd_RR_normalized, np.nan)
    psd_RR_normalized_25_40_hz = psd_RR_normalized_25_40_hz[~np.isnan(psd_RR_normalized_25_40_hz)]
    mean_psd_RR_normalized_25_40_hz = np.mean(psd_RR_normalized_25_40_hz)
    # Mean power spectral density for 40-60 Hz frequency range
    psd_RR_40_60_hz = np.where( (40 <=freqs) & (freqs<= 60), psd_RR, np.nan)
    psd_RR_40_60_hz = psd_RR_40_60_hz[~np.isnan(psd_RR_40_60_hz)]
    mean_psd_RR_40_60_hz = np.mean(psd_RR_40_60_hz)
    psd_RR_normalized_40_60_hz = np.where( (40 <=freqs) & (freqs<= 60), psd_RR_normalized, np.nan)
    psd_RR_normalized_40_60_hz = psd_RR_normalized_40_60_hz[~np.isnan(psd_RR_normalized_40_60_hz)]
    mean_psd_RR_normalized_40_60_hz = np.mean(psd_RR_normalized_40_60_hz)
    # Mean power spectral density for 60-80 Hz frequency range
    psd_RR_60_80_hz = np.where( (60 <=freqs) & (freqs<= 80), psd_RR, np.nan)
    psd_RR_60_80_hz = psd_RR_60_80_hz[~np.isnan(psd_RR_60_80_hz)]
    mean_psd_RR_60_80_hz = np.mean(psd_RR_60_80_hz)
    psd_RR_normalized_60_80_hz = np.where( (60 <=freqs) & (freqs<= 80), psd_RR_normalized, np.nan)
    psd_RR_normalized_60_80_hz = psd_RR_normalized_60_80_hz[~np.isnan(psd_RR_normalized_60_80_hz)]
    mean_psd_RR_normalized_60_80_hz = np.mean(psd_RR_normalized_60_80_hz)
    # Mean power spectral density for 80-100 Hz frequency range
    psd_RR_80_100_hz = np.where( (80 <=freqs) & (freqs<= 100), psd_RR, np.nan)
    psd_RR_80_100_hz = psd_RR_80_100_hz[~np.isnan(psd_RR_80_100_hz)]
    mean_psd_RR_80_100_hz = np.mean(psd_RR_80_100_hz)
    psd_RR_normalized_80_100_hz = np.where( (80 <=freqs) & (freqs<= 100), psd_RR_normalized, np.nan)
    psd_RR_normalized_80_100_hz = psd_RR_normalized_80_100_hz[~np.isnan(psd_RR_normalized_80_100_hz)]
    mean_psd_RR_normalized_80_100_hz = np.mean(psd_RR_normalized_80_100_hz)
    # Mean power spectral density for 100-120 Hz frequency range
    psd_RR_100_120_hz = np.where( (100 <=freqs) & (freqs<= 120), psd_RR, np.nan)
    psd_RR_100_120_hz = psd_RR_100_120_hz[~np.isnan(psd_RR_100_120_hz)]
    mean_psd_RR_100_120_hz = np.mean(psd_RR_100_120_hz)
    psd_RR_normalized_100_120_hz = np.where( (100 <=freqs) & (freqs<= 120), psd_RR_normalized, np.nan)
    psd_RR_normalized_100_120_hz = psd_RR_normalized_100_120_hz[~np.isnan(psd_RR_normalized_100_120_hz)]
    mean_psd_RR_normalized_100_120_hz = np.mean(psd_RR_normalized_100_120_hz)
    # Mean power spectral density for 120-140 Hz frequency range
    psd_RR_120_140_hz = np.where( (120 <=freqs) & (freqs<= 140), psd_RR, np.nan)
    psd_RR_120_140_hz = psd_RR_120_140_hz[~np.isnan(psd_RR_120_140_hz)]
    mean_psd_RR_120_140_hz = np.mean(psd_RR_120_140_hz)
    psd_RR_normalized_120_140_hz = np.where( (120 <=freqs) & (freqs<= 140), psd_RR_normalized, np.nan)
    psd_RR_normalized_120_140_hz = psd_RR_normalized_120_140_hz[~np.isnan(psd_RR_normalized_120_140_hz)]
    mean_psd_RR_normalized_120_140_hz = np.mean(psd_RR_normalized_120_140_hz)
    # Mean power spectral density for 140-160 Hz frequency range
    psd_RR_140_160_hz = np.where( (140 <=freqs) & (freqs<= 160), psd_RR, np.nan)
    psd_RR_140_160_hz = psd_RR_140_160_hz[~np.isnan(psd_RR_140_160_hz)]
    mean_psd_RR_140_160_hz = np.mean(psd_RR_140_160_hz)
    psd_RR_normalized_140_160_hz = np.where( (140 <=freqs) & (freqs<= 160), psd_RR_normalized, np.nan)
    psd_RR_normalized_140_160_hz = psd_RR_normalized_140_160_hz[~np.isnan(psd_RR_normalized_140_160_hz)]
    mean_psd_RR_normalized_140_160_hz = np.mean(psd_RR_normalized_140_160_hz)
    # Mean power spectral density for 160-180 Hz frequency range
    psd_RR_160_180_hz = np.where( (160 <=freqs) & (freqs<= 180), psd_RR, np.nan)
    psd_RR_160_180_hz = psd_RR_160_180_hz[~np.isnan(psd_RR_160_180_hz)]
    mean_psd_RR_160_180_hz = np.mean(psd_RR_160_180_hz)
    psd_RR_normalized_160_180_hz = np.where( (160 <=freqs) & (freqs<= 180), psd_RR_normalized, np.nan)
    psd_RR_normalized_160_180_hz = psd_RR_normalized_160_180_hz[~np.isnan(psd_RR_normalized_160_180_hz)]
    mean_psd_RR_normalized_160_180_hz = np.mean(psd_RR_normalized_160_180_hz)
    # Mean power spectral density for 180-200 Hz frequency range
    psd_RR_180_200_hz = np.where( (180 <=freqs) & (freqs<= 200), psd_RR, np.nan)
    psd_RR_180_200_hz = psd_RR_180_200_hz[~np.isnan(psd_RR_180_200_hz)]
    mean_psd_RR_180_200_hz = np.mean(psd_RR_180_200_hz)
    psd_RR_normalized_180_200_hz = np.where( (180 <=freqs) & (freqs<= 200), psd_RR_normalized, np.nan)
    psd_RR_normalized_180_200_hz = psd_RR_normalized_180_200_hz[~np.isnan(psd_RR_normalized_180_200_hz)]
    mean_psd_RR_normalized_180_200_hz = np.mean(psd_RR_normalized_180_200_hz)
    # Mean power spectral density for 200-250 Hz frequency range
    psd_RR_200_250_hz = np.where( (200 <=freqs) & (freqs<= 250), psd_RR, np.nan)
    psd_RR_200_250_hz = psd_RR_200_250_hz[~np.isnan(psd_RR_200_250_hz)]
    mean_psd_RR_200_250_hz = np.mean(psd_RR_200_250_hz)
    psd_RR_normalized_200_250_hz = np.where( (200 <=freqs) & (freqs<= 250), psd_RR_normalized, np.nan)
    psd_RR_normalized_200_250_hz = psd_RR_normalized_200_250_hz[~np.isnan(psd_RR_normalized_200_250_hz)]
    mean_psd_RR_normalized_200_250_hz = np.mean(psd_RR_normalized_200_250_hz)
    # Mean power spectral density for 250-300 Hz frequency range
    psd_RR_250_300_hz = np.where( (250 <=freqs) & (freqs<= 300), psd_RR, np.nan)
    psd_RR_250_300_hz = psd_RR_250_300_hz[~np.isnan(psd_RR_250_300_hz)]
    mean_psd_RR_250_300_hz = np.mean(psd_RR_250_300_hz)
    psd_RR_normalized_250_300_hz = np.where( (250 <=freqs) & (freqs<= 300), psd_RR_normalized, np.nan)
    psd_RR_normalized_250_300_hz = psd_RR_normalized_250_300_hz[~np.isnan(psd_RR_normalized_250_300_hz)]
    mean_psd_RR_normalized_250_300_hz = np.mean(psd_RR_normalized_250_300_hz)
    # Mean power spectral density for 300-400 Hz frequency range
    psd_RR_300_400_hz = np.where( (300 <=freqs) & (freqs<= 400), psd_RR, np.nan)
    psd_RR_300_400_hz = psd_RR_300_400_hz[~np.isnan(psd_RR_300_400_hz)]
    mean_psd_RR_300_400_hz = np.mean(psd_RR_300_400_hz)
    psd_RR_normalized_300_400_hz = np.where( (300 <=freqs) & (freqs<= 400), psd_RR_normalized, np.nan)
    psd_RR_normalized_300_400_hz = psd_RR_normalized_300_400_hz[~np.isnan(psd_RR_normalized_300_400_hz)]
    mean_psd_RR_normalized_300_400_hz = np.mean(psd_RR_normalized_300_400_hz)
    
    # Power spectral density for different lower- and upper-bounds frequency ranges (Sys):
    freqs, psd_systole = signal.welch(systole, Fs)
    analytic_signal_systole = hilbert(psd_systole)
    amplitude_envelope_systole = np.abs(analytic_signal_systole)
    evelope_integral_psd_systole = np.trapz(amplitude_envelope_systole, dx=5)
    psd_systole_normalized = psd_systole/evelope_integral_psd_systole
    # Mean power spectral density for all frequencies
    mean_psd_systole = np.mean(psd_systole)
    mean_psd_systole_normalized = np.mean(psd_systole_normalized)
    # Mean power spectral density for 25-40 Hz frequency range
    psd_systole_25_40_hz = np.where( (25 <=freqs) & (freqs<= 40), psd_systole, np.nan)
    psd_systole_25_40_hz = psd_systole_25_40_hz[~np.isnan(psd_systole_25_40_hz)]
    mean_psd_systole_25_40_hz = np.mean(psd_systole_25_40_hz)
    psd_systole_normalized_25_40_hz = np.where( (25 <=freqs) & (freqs<= 40), psd_systole_normalized, np.nan)
    psd_systole_normalized_25_40_hz = psd_systole_normalized_25_40_hz[~np.isnan(psd_systole_normalized_25_40_hz)]
    mean_psd_systole_normalized_25_40_hz = np.mean(psd_systole_normalized_25_40_hz)
    # Mean power spectral density for 40-60 Hz frequency range
    psd_systole_40_60_hz = np.where( (40 <=freqs) & (freqs<= 60), psd_systole, np.nan)
    psd_systole_40_60_hz = psd_systole_40_60_hz[~np.isnan(psd_systole_40_60_hz)]
    mean_psd_systole_40_60_hz = np.mean(psd_systole_40_60_hz)
    psd_systole_normalized_40_60_hz = np.where( (40 <=freqs) & (freqs<= 60), psd_systole_normalized, np.nan)
    psd_systole_normalized_40_60_hz = psd_systole_normalized_40_60_hz[~np.isnan(psd_systole_normalized_40_60_hz)]
    mean_psd_systole_normalized_40_60_hz = np.mean(psd_systole_normalized_40_60_hz)
    # Mean power spectral density for 60-80 Hz frequency range
    psd_systole_60_80_hz = np.where( (60 <=freqs) & (freqs<= 80), psd_systole, np.nan)
    psd_systole_60_80_hz = psd_systole_60_80_hz[~np.isnan(psd_systole_60_80_hz)]
    mean_psd_systole_60_80_hz = np.mean(psd_systole_60_80_hz)
    psd_systole_normalized_60_80_hz = np.where( (60 <=freqs) & (freqs<= 80), psd_systole_normalized, np.nan)
    psd_systole_normalized_60_80_hz = psd_systole_normalized_60_80_hz[~np.isnan(psd_systole_normalized_60_80_hz)]
    mean_psd_systole_normalized_60_80_hz = np.mean(psd_systole_normalized_60_80_hz)
    # Mean power spectral density for 80-100 Hz frequency range
    psd_systole_80_100_hz = np.where( (80 <=freqs) & (freqs<= 100), psd_systole, np.nan)
    psd_systole_80_100_hz = psd_systole_80_100_hz[~np.isnan(psd_systole_80_100_hz)]
    mean_psd_systole_80_100_hz = np.mean(psd_systole_80_100_hz)
    psd_systole_normalized_80_100_hz = np.where( (80 <=freqs) & (freqs<= 100), psd_systole_normalized, np.nan)
    psd_systole_normalized_80_100_hz = psd_systole_normalized_80_100_hz[~np.isnan(psd_systole_normalized_80_100_hz)]
    mean_psd_systole_normalized_80_100_hz = np.mean(psd_systole_normalized_80_100_hz)
    # Mean power spectral density for 100-120 Hz frequency range
    psd_systole_100_120_hz = np.where( (100 <=freqs) & (freqs<= 120), psd_systole, np.nan)
    psd_systole_100_120_hz = psd_systole_100_120_hz[~np.isnan(psd_systole_100_120_hz)]
    mean_psd_systole_100_120_hz = np.mean(psd_systole_100_120_hz)
    psd_systole_normalized_100_120_hz = np.where( (100 <=freqs) & (freqs<= 120), psd_systole_normalized, np.nan)
    psd_systole_normalized_100_120_hz = psd_systole_normalized_100_120_hz[~np.isnan(psd_systole_normalized_100_120_hz)]
    mean_psd_systole_normalized_100_120_hz = np.mean(psd_systole_normalized_100_120_hz)
    # Mean power spectral density for 120-140 Hz frequency range
    psd_systole_120_140_hz = np.where( (120 <=freqs) & (freqs<= 140), psd_systole, np.nan)
    psd_systole_120_140_hz = psd_systole_120_140_hz[~np.isnan(psd_systole_120_140_hz)]
    mean_psd_systole_120_140_hz = np.mean(psd_systole_120_140_hz)
    psd_systole_normalized_120_140_hz = np.where( (120 <=freqs) & (freqs<= 140), psd_systole_normalized, np.nan)
    psd_systole_normalized_120_140_hz = psd_systole_normalized_120_140_hz[~np.isnan(psd_systole_normalized_120_140_hz)]
    mean_psd_systole_normalized_120_140_hz = np.mean(psd_systole_normalized_120_140_hz)
    # Mean power spectral density for 140-160 Hz frequency range
    psd_systole_140_160_hz = np.where( (140 <=freqs) & (freqs<= 160), psd_systole, np.nan)
    psd_systole_140_160_hz = psd_systole_140_160_hz[~np.isnan(psd_systole_140_160_hz)]
    mean_psd_systole_140_160_hz = np.mean(psd_systole_140_160_hz)
    psd_systole_normalized_140_160_hz = np.where( (140 <=freqs) & (freqs<= 160), psd_systole_normalized, np.nan)
    psd_systole_normalized_140_160_hz = psd_systole_normalized_140_160_hz[~np.isnan(psd_systole_normalized_140_160_hz)]
    mean_psd_systole_normalized_140_160_hz = np.mean(psd_systole_normalized_140_160_hz)
    # Mean power spectral density for 160-180 Hz frequency range
    psd_systole_160_180_hz = np.where( (160 <=freqs) & (freqs<= 180), psd_systole, np.nan)
    psd_systole_160_180_hz = psd_systole_160_180_hz[~np.isnan(psd_systole_160_180_hz)]
    mean_psd_systole_160_180_hz = np.mean(psd_systole_160_180_hz)
    psd_systole_normalized_160_180_hz = np.where( (160 <=freqs) & (freqs<= 180), psd_systole_normalized, np.nan)
    psd_systole_normalized_160_180_hz = psd_systole_normalized_160_180_hz[~np.isnan(psd_systole_normalized_160_180_hz)]
    mean_psd_systole_normalized_160_180_hz = np.mean(psd_systole_normalized_160_180_hz)
    # Mean power spectral density for 180-200 Hz frequency range
    psd_systole_180_200_hz = np.where( (180 <=freqs) & (freqs<= 200), psd_systole, np.nan)
    psd_systole_180_200_hz = psd_systole_180_200_hz[~np.isnan(psd_systole_180_200_hz)]
    mean_psd_systole_180_200_hz = np.mean(psd_systole_180_200_hz)
    psd_systole_normalized_180_200_hz = np.where( (180 <=freqs) & (freqs<= 200), psd_systole_normalized, np.nan)
    psd_systole_normalized_180_200_hz = psd_systole_normalized_180_200_hz[~np.isnan(psd_systole_normalized_180_200_hz)]
    mean_psd_systole_normalized_180_200_hz = np.mean(psd_systole_normalized_180_200_hz)
    # Mean power spectral density for 200-250 Hz frequency range
    psd_systole_200_250_hz = np.where( (200 <=freqs) & (freqs<= 250), psd_systole, np.nan)
    psd_systole_200_250_hz = psd_systole_200_250_hz[~np.isnan(psd_systole_200_250_hz)]
    mean_psd_systole_200_250_hz = np.mean(psd_systole_200_250_hz)
    psd_systole_normalized_200_250_hz = np.where( (200 <=freqs) & (freqs<= 250), psd_systole_normalized, np.nan)
    psd_systole_normalized_200_250_hz = psd_systole_normalized_200_250_hz[~np.isnan(psd_systole_normalized_200_250_hz)]
    mean_psd_systole_normalized_200_250_hz = np.mean(psd_systole_normalized_200_250_hz)
    # Mean power spectral density for 250-300 Hz frequency range
    psd_systole_250_300_hz = np.where( (250 <=freqs) & (freqs<= 300), psd_systole, np.nan)
    psd_systole_250_300_hz = psd_systole_250_300_hz[~np.isnan(psd_systole_250_300_hz)]
    mean_psd_systole_250_300_hz = np.mean(psd_systole_250_300_hz)
    psd_systole_normalized_250_300_hz = np.where( (250 <=freqs) & (freqs<= 300), psd_systole_normalized, np.nan)
    psd_systole_normalized_250_300_hz = psd_systole_normalized_250_300_hz[~np.isnan(psd_systole_normalized_250_300_hz)]
    mean_psd_systole_normalized_250_300_hz = np.mean(psd_systole_normalized_250_300_hz)
    # Mean power spectral density for 300-400 Hz frequency range
    psd_systole_300_400_hz = np.where( (300 <=freqs) & (freqs<= 400), psd_systole, np.nan)
    psd_systole_300_400_hz = psd_systole_300_400_hz[~np.isnan(psd_systole_300_400_hz)]
    mean_psd_systole_300_400_hz = np.mean(psd_systole_300_400_hz)
    psd_systole_normalized_300_400_hz = np.where( (300 <=freqs) & (freqs<= 400), psd_systole_normalized, np.nan)
    psd_systole_normalized_300_400_hz = psd_systole_normalized_300_400_hz[~np.isnan(psd_systole_normalized_300_400_hz)]
    mean_psd_systole_normalized_300_400_hz = np.mean(psd_systole_normalized_300_400_hz)
    
    # Power spectral density for different lower- and upper-bounds frequency ranges (diastole):
    freqs, psd_diastole = signal.welch(diastole, Fs)
    analytic_signal_diastole = hilbert(psd_diastole)
    amplitude_envelope_diastole = np.abs(analytic_signal_diastole)
    evelope_integral_psd_diastole = np.trapz(amplitude_envelope_diastole, dx=5)
    psd_diastole_normalized = psd_diastole/evelope_integral_psd_diastole
    # Mean power spectral density for all frequencies
    mean_psd_diastole = np.mean(psd_diastole)
    mean_psd_diastole_normalized = np.mean(psd_diastole_normalized)
    # Mean power spectral density for 25-40 Hz frequency range
    psd_diastole_25_40_hz = np.where( (25 <=freqs) & (freqs<= 40), psd_diastole, np.nan)
    psd_diastole_25_40_hz = psd_diastole_25_40_hz[~np.isnan(psd_diastole_25_40_hz)]
    mean_psd_diastole_25_40_hz = np.mean(psd_diastole_25_40_hz)
    psd_diastole_normalized_25_40_hz = np.where( (25 <=freqs) & (freqs<= 40), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_25_40_hz = psd_diastole_normalized_25_40_hz[~np.isnan(psd_diastole_normalized_25_40_hz)]
    mean_psd_diastole_normalized_25_40_hz = np.mean(psd_diastole_normalized_25_40_hz)
    # Mean power spectral density for 40-60 Hz frequency range
    psd_diastole_40_60_hz = np.where( (40 <=freqs) & (freqs<= 60), psd_diastole, np.nan)
    psd_diastole_40_60_hz = psd_diastole_40_60_hz[~np.isnan(psd_diastole_40_60_hz)]
    mean_psd_diastole_40_60_hz = np.mean(psd_diastole_40_60_hz)
    psd_diastole_normalized_40_60_hz = np.where( (40 <=freqs) & (freqs<= 60), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_40_60_hz = psd_diastole_normalized_40_60_hz[~np.isnan(psd_diastole_normalized_40_60_hz)]
    mean_psd_diastole_normalized_40_60_hz = np.mean(psd_diastole_normalized_40_60_hz)
    # Mean power spectral density for 60-80 Hz frequency range
    psd_diastole_60_80_hz = np.where( (60 <=freqs) & (freqs<= 80), psd_diastole, np.nan)
    psd_diastole_60_80_hz = psd_diastole_60_80_hz[~np.isnan(psd_diastole_60_80_hz)]
    mean_psd_diastole_60_80_hz = np.mean(psd_diastole_60_80_hz)
    psd_diastole_normalized_60_80_hz = np.where( (60 <=freqs) & (freqs<= 80), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_60_80_hz = psd_diastole_normalized_60_80_hz[~np.isnan(psd_diastole_normalized_60_80_hz)]
    mean_psd_diastole_normalized_60_80_hz = np.mean(psd_diastole_normalized_60_80_hz)
    # Mean power spectral density for 80-100 Hz frequency range
    psd_diastole_80_100_hz = np.where( (80 <=freqs) & (freqs<= 100), psd_diastole, np.nan)
    psd_diastole_80_100_hz = psd_diastole_80_100_hz[~np.isnan(psd_diastole_80_100_hz)]
    mean_psd_diastole_80_100_hz = np.mean(psd_diastole_80_100_hz)
    psd_diastole_normalized_80_100_hz = np.where( (80 <=freqs) & (freqs<= 100), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_80_100_hz = psd_diastole_normalized_80_100_hz[~np.isnan(psd_diastole_normalized_80_100_hz)]
    mean_psd_diastole_normalized_80_100_hz = np.mean(psd_diastole_normalized_80_100_hz)
    # Mean power spectral density for 100-120 Hz frequency range
    psd_diastole_100_120_hz = np.where( (100 <=freqs) & (freqs<= 120), psd_diastole, np.nan)
    psd_diastole_100_120_hz = psd_diastole_100_120_hz[~np.isnan(psd_diastole_100_120_hz)]
    mean_psd_diastole_100_120_hz = np.mean(psd_diastole_100_120_hz)
    psd_diastole_normalized_100_120_hz = np.where( (100 <=freqs) & (freqs<= 120), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_100_120_hz = psd_diastole_normalized_100_120_hz[~np.isnan(psd_diastole_normalized_100_120_hz)]
    mean_psd_diastole_normalized_100_120_hz = np.mean(psd_diastole_normalized_100_120_hz)
    # Mean power spectral density for 120-140 Hz frequency range
    psd_diastole_120_140_hz = np.where( (120 <=freqs) & (freqs<= 140), psd_diastole, np.nan)
    psd_diastole_120_140_hz = psd_diastole_120_140_hz[~np.isnan(psd_diastole_120_140_hz)]
    mean_psd_diastole_120_140_hz = np.mean(psd_diastole_120_140_hz)
    psd_diastole_normalized_120_140_hz = np.where( (120 <=freqs) & (freqs<= 140), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_120_140_hz = psd_diastole_normalized_120_140_hz[~np.isnan(psd_diastole_normalized_120_140_hz)]
    mean_psd_diastole_normalized_120_140_hz = np.mean(psd_diastole_normalized_120_140_hz)
    # Mean power spectral density for 140-160 Hz frequency range
    psd_diastole_140_160_hz = np.where( (140 <=freqs) & (freqs<= 160), psd_diastole, np.nan)
    psd_diastole_140_160_hz = psd_diastole_140_160_hz[~np.isnan(psd_diastole_140_160_hz)]
    mean_psd_diastole_140_160_hz = np.mean(psd_diastole_140_160_hz)
    psd_diastole_normalized_140_160_hz = np.where( (140 <=freqs) & (freqs<= 160), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_140_160_hz = psd_diastole_normalized_140_160_hz[~np.isnan(psd_diastole_normalized_140_160_hz)]
    mean_psd_diastole_normalized_140_160_hz = np.mean(psd_diastole_normalized_140_160_hz)
    # Mean power spectral density for 160-180 Hz frequency range
    psd_diastole_160_180_hz = np.where( (160 <=freqs) & (freqs<= 180), psd_diastole, np.nan)
    psd_diastole_160_180_hz = psd_diastole_160_180_hz[~np.isnan(psd_diastole_160_180_hz)]
    mean_psd_diastole_160_180_hz = np.mean(psd_diastole_160_180_hz)
    psd_diastole_normalized_160_180_hz = np.where( (160 <=freqs) & (freqs<= 180), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_160_180_hz = psd_diastole_normalized_160_180_hz[~np.isnan(psd_diastole_normalized_160_180_hz)]
    mean_psd_diastole_normalized_160_180_hz = np.mean(psd_diastole_normalized_160_180_hz)
    # Mean power spectral density for 180-200 Hz frequency range
    psd_diastole_180_200_hz = np.where( (180 <=freqs) & (freqs<= 200), psd_diastole, np.nan)
    psd_diastole_180_200_hz = psd_diastole_180_200_hz[~np.isnan(psd_diastole_180_200_hz)]
    mean_psd_diastole_180_200_hz = np.mean(psd_diastole_180_200_hz)
    psd_diastole_normalized_180_200_hz = np.where( (180 <=freqs) & (freqs<= 200), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_180_200_hz = psd_diastole_normalized_180_200_hz[~np.isnan(psd_diastole_normalized_180_200_hz)]
    mean_psd_diastole_normalized_180_200_hz = np.mean(psd_diastole_normalized_180_200_hz)
    # Mean power spectral density for 200-250 Hz frequency range
    psd_diastole_200_250_hz = np.where( (200 <=freqs) & (freqs<= 250), psd_diastole, np.nan)
    psd_diastole_200_250_hz = psd_diastole_200_250_hz[~np.isnan(psd_diastole_200_250_hz)]
    mean_psd_diastole_200_250_hz = np.mean(psd_diastole_200_250_hz)
    psd_diastole_normalized_200_250_hz = np.where( (200 <=freqs) & (freqs<= 250), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_200_250_hz = psd_diastole_normalized_200_250_hz[~np.isnan(psd_diastole_normalized_200_250_hz)]
    mean_psd_diastole_normalized_200_250_hz = np.mean(psd_diastole_normalized_200_250_hz)
    # Mean power spectral density for 250-300 Hz frequency range
    psd_diastole_250_300_hz = np.where( (250 <=freqs) & (freqs<= 300), psd_diastole, np.nan)
    psd_diastole_250_300_hz = psd_diastole_250_300_hz[~np.isnan(psd_diastole_250_300_hz)]
    mean_psd_diastole_250_300_hz = np.mean(psd_diastole_250_300_hz)
    psd_diastole_normalized_250_300_hz = np.where( (250 <=freqs) & (freqs<= 300), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_250_300_hz = psd_diastole_normalized_250_300_hz[~np.isnan(psd_diastole_normalized_250_300_hz)]
    mean_psd_diastole_normalized_250_300_hz = np.mean(psd_diastole_normalized_250_300_hz)
    # Mean power spectral density for 300-400 Hz frequency range
    psd_diastole_300_400_hz = np.where( (300 <=freqs) & (freqs<= 400), psd_diastole, np.nan)
    psd_diastole_300_400_hz = psd_diastole_300_400_hz[~np.isnan(psd_diastole_300_400_hz)]
    mean_psd_diastole_300_400_hz = np.mean(psd_diastole_300_400_hz)
    psd_diastole_normalized_300_400_hz = np.where( (300 <=freqs) & (freqs<= 400), psd_diastole_normalized, np.nan)
    psd_diastole_normalized_300_400_hz = psd_diastole_normalized_300_400_hz[~np.isnan(psd_diastole_normalized_300_400_hz)]
    mean_psd_diastole_normalized_300_400_hz = np.mean(psd_diastole_normalized_300_400_hz)
    
    # Ratio between the mean PSD of the systole and RR
    mean_psd_ratio_systole_RR = round(mean_psd_systole_normalized/mean_psd_RR_normalized, 4)
    # Ratio between the mean PSD of the diastole and RR
    mean_psd_ratio_diastole_RR = round(mean_psd_diastole_normalized/mean_psd_RR_normalized, 4)

    # Zero crossing rate of RR
    zero_crossing_RR = sum(librosa.zero_crossings(RR))
    # Zero crossing rate of S1
    zero_crossing_S1 = sum(librosa.zero_crossings(S1))
    # Zero crossing rate of RR
    zero_crossing_systole = sum(librosa.zero_crossings(systole))
    # Zero crossing rate of RR
    zero_crossing_S2 = sum(librosa.zero_crossings(S2))
    # Zero crossing rate of RR
    zero_crossing_diastole = sum(librosa.zero_crossings(diastole))

    n_fft = int(256)
    hop_length = int(256/4)
    # Chroma stft RR
    chroma_stft_RR = librosa.feature.chroma_stft(y=RR, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    chroma_stft1_RR = np.mean(chroma_stft_RR[0])
    chroma_stft2_RR = np.mean(chroma_stft_RR[1])
    chroma_stft3_RR = np.mean(chroma_stft_RR[2])
    chroma_stft4_RR = np.mean(chroma_stft_RR[3])
    chroma_stft5_RR = np.mean(chroma_stft_RR[4])
    chroma_stft6_RR = np.mean(chroma_stft_RR[5])
    chroma_stft7_RR = np.mean(chroma_stft_RR[6])
    chroma_stft8_RR = np.mean(chroma_stft_RR[7])
    chroma_stft9_RR = np.mean(chroma_stft_RR[8])
    chroma_stft10_RR = np.mean(chroma_stft_RR[9])
    chroma_stft11_RR = np.mean(chroma_stft_RR[10])
    chroma_stft12_RR = np.mean(chroma_stft_RR[11])
    # Chroma stft S1
    chroma_stft_S1 = librosa.feature.chroma_stft(y=S1, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    chroma_stft1_S1 = np.mean(chroma_stft_S1[0])
    chroma_stft2_S1 = np.mean(chroma_stft_S1[1])
    chroma_stft3_S1 = np.mean(chroma_stft_S1[2])
    chroma_stft4_S1 = np.mean(chroma_stft_S1[3])
    chroma_stft5_S1 = np.mean(chroma_stft_S1[4])
    chroma_stft6_S1 = np.mean(chroma_stft_S1[5])
    chroma_stft7_S1 = np.mean(chroma_stft_S1[6])
    chroma_stft8_S1 = np.mean(chroma_stft_S1[7])
    chroma_stft9_S1 = np.mean(chroma_stft_S1[8])
    chroma_stft10_S1 = np.mean(chroma_stft_S1[9])
    chroma_stft11_S1 = np.mean(chroma_stft_S1[10])
    chroma_stft12_S1 = np.mean(chroma_stft_S1[11])
    # Chroma stft systole
    chroma_stft_systole = librosa.feature.chroma_stft(y=systole, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    chroma_stft1_systole = np.mean(chroma_stft_systole[0])
    chroma_stft2_systole = np.mean(chroma_stft_systole[1])
    chroma_stft3_systole = np.mean(chroma_stft_systole[2])
    chroma_stft4_systole = np.mean(chroma_stft_systole[3])
    chroma_stft5_systole = np.mean(chroma_stft_systole[4])
    chroma_stft6_systole = np.mean(chroma_stft_systole[5])
    chroma_stft7_systole = np.mean(chroma_stft_systole[6])
    chroma_stft8_systole = np.mean(chroma_stft_systole[7])
    chroma_stft9_systole = np.mean(chroma_stft_systole[8])
    chroma_stft10_systole = np.mean(chroma_stft_systole[9])
    chroma_stft11_systole = np.mean(chroma_stft_systole[10])
    chroma_stft12_systole = np.mean(chroma_stft_systole[11])
    # Chroma stft S2
    chroma_stft_S2 = librosa.feature.chroma_stft(y=S2, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    chroma_stft1_S2 = np.mean(chroma_stft_S2[0])
    chroma_stft2_S2 = np.mean(chroma_stft_S2[1])
    chroma_stft3_S2 = np.mean(chroma_stft_S2[2])
    chroma_stft4_S2 = np.mean(chroma_stft_S2[3])
    chroma_stft5_S2 = np.mean(chroma_stft_S2[4])
    chroma_stft6_S2 = np.mean(chroma_stft_S2[5])
    chroma_stft7_S2 = np.mean(chroma_stft_S2[6])
    chroma_stft8_S2 = np.mean(chroma_stft_S2[7])
    chroma_stft9_S2 = np.mean(chroma_stft_S2[8])
    chroma_stft10_S2 = np.mean(chroma_stft_S2[9])
    chroma_stft11_S2 = np.mean(chroma_stft_S2[10])
    chroma_stft12_S2 = np.mean(chroma_stft_S2[11])
    # Chroma stft diastole
    chroma_stft_diastole = librosa.feature.chroma_stft(y=diastole, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    chroma_stft1_diastole = np.mean(chroma_stft_diastole[0])
    chroma_stft2_diastole = np.mean(chroma_stft_diastole[1])
    chroma_stft3_diastole = np.mean(chroma_stft_diastole[2])
    chroma_stft4_diastole = np.mean(chroma_stft_diastole[3])
    chroma_stft5_diastole = np.mean(chroma_stft_diastole[4])
    chroma_stft6_diastole = np.mean(chroma_stft_diastole[5])
    chroma_stft7_diastole = np.mean(chroma_stft_diastole[6])
    chroma_stft8_diastole = np.mean(chroma_stft_diastole[7])
    chroma_stft9_diastole = np.mean(chroma_stft_diastole[8])
    chroma_stft10_diastole = np.mean(chroma_stft_diastole[9])
    chroma_stft11_diastole = np.mean(chroma_stft_diastole[10])
    chroma_stft12_diastole = np.mean(chroma_stft_diastole[11])
    
    # Melspectrogram
    # Melspectrogram stft RR
    melspectrogram_RR = librosa.feature.melspectrogram(y=RR, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    melspectrogram1_RR = np.mean(melspectrogram_RR[0])
    melspectrogram2_RR = np.mean(melspectrogram_RR[1])
    melspectrogram3_RR = np.mean(melspectrogram_RR[2])
    melspectrogram4_RR = np.mean(melspectrogram_RR[3])
    melspectrogram5_RR = np.mean(melspectrogram_RR[4])
    melspectrogram6_RR = np.mean(melspectrogram_RR[5])
    melspectrogram7_RR = np.mean(melspectrogram_RR[6])
    melspectrogram8_RR = np.mean(melspectrogram_RR[7])
    melspectrogram9_RR = np.mean(melspectrogram_RR[8])
    melspectrogram10_RR = np.mean(melspectrogram_RR[9])
    melspectrogram11_RR = np.mean(melspectrogram_RR[10])
    melspectrogram12_RR = np.mean(melspectrogram_RR[11])
    # Melspectrogram stft S1
    melspectrogram_S1 = librosa.feature.melspectrogram(y=S1, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    melspectrogram1_S1 = np.mean(melspectrogram_S1[0])
    melspectrogram2_S1 = np.mean(melspectrogram_S1[1])
    melspectrogram3_S1 = np.mean(melspectrogram_S1[2])
    melspectrogram4_S1 = np.mean(melspectrogram_S1[3])
    melspectrogram5_S1 = np.mean(melspectrogram_S1[4])
    melspectrogram6_S1 = np.mean(melspectrogram_S1[5])
    melspectrogram7_S1 = np.mean(melspectrogram_S1[6])
    melspectrogram8_S1 = np.mean(melspectrogram_S1[7])
    melspectrogram9_S1 = np.mean(melspectrogram_S1[8])
    melspectrogram10_S1 = np.mean(melspectrogram_S1[9])
    melspectrogram11_S1 = np.mean(melspectrogram_S1[10])
    melspectrogram12_S1 = np.mean(melspectrogram_S1[11])
    # Melspectrogram stft systole
    melspectrogram_systole = librosa.feature.melspectrogram(y=systole, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    melspectrogram1_systole = np.mean(melspectrogram_systole[0])
    melspectrogram2_systole = np.mean(melspectrogram_systole[1])
    melspectrogram3_systole = np.mean(melspectrogram_systole[2])
    melspectrogram4_systole = np.mean(melspectrogram_systole[3])
    melspectrogram5_systole = np.mean(melspectrogram_systole[4])
    melspectrogram6_systole = np.mean(melspectrogram_systole[5])
    melspectrogram7_systole = np.mean(melspectrogram_systole[6])
    melspectrogram8_systole = np.mean(melspectrogram_systole[7])
    melspectrogram9_systole = np.mean(melspectrogram_systole[8])
    melspectrogram10_systole = np.mean(melspectrogram_systole[9])
    melspectrogram11_systole = np.mean(melspectrogram_systole[10])
    melspectrogram12_systole = np.mean(melspectrogram_systole[11])
    # Melspectrogram stft S2
    melspectrogram_S2 = librosa.feature.melspectrogram(y=S2, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    melspectrogram1_S2 = np.mean(melspectrogram_S2[0])
    melspectrogram2_S2 = np.mean(melspectrogram_S2[1])
    melspectrogram3_S2 = np.mean(melspectrogram_S2[2])
    melspectrogram4_S2 = np.mean(melspectrogram_S2[3])
    melspectrogram5_S2 = np.mean(melspectrogram_S2[4])
    melspectrogram6_S2 = np.mean(melspectrogram_S2[5])
    melspectrogram7_S2 = np.mean(melspectrogram_S2[6])
    melspectrogram8_S2 = np.mean(melspectrogram_S2[7])
    melspectrogram9_S2 = np.mean(melspectrogram_S2[8])
    melspectrogram10_S2 = np.mean(melspectrogram_S2[9])
    melspectrogram11_S2 = np.mean(melspectrogram_S2[10])
    melspectrogram12_S2 = np.mean(melspectrogram_S2[11])
    # Melspectrogram stft diastole
    melspectrogram_diastole = librosa.feature.melspectrogram(y=diastole, sr=Fs, n_fft = n_fft, hop_length = hop_length)
    melspectrogram1_diastole = np.mean(melspectrogram_diastole[0])
    melspectrogram2_diastole = np.mean(melspectrogram_diastole[1])
    melspectrogram3_diastole = np.mean(melspectrogram_diastole[2])
    melspectrogram4_diastole = np.mean(melspectrogram_diastole[3])
    melspectrogram5_diastole = np.mean(melspectrogram_diastole[4])
    melspectrogram6_diastole = np.mean(melspectrogram_diastole[5])
    melspectrogram7_diastole = np.mean(melspectrogram_diastole[6])
    melspectrogram8_diastole = np.mean(melspectrogram_diastole[7])
    melspectrogram9_diastole = np.mean(melspectrogram_diastole[8])
    melspectrogram10_diastole = np.mean(melspectrogram_diastole[9])
    melspectrogram11_diastole = np.mean(melspectrogram_diastole[10])
    melspectrogram12_diastole = np.mean(melspectrogram_diastole[11])
    
    # MFCC
    # MFCC RR
    mfcc_RR = librosa.feature.mfcc(y=RR, n_mfcc = 13, sr = Fs, n_fft= n_fft, hop_length= hop_length)
    mfcc1_RR = np.mean(mfcc_RR[0])
    mfcc2_RR = np.mean(mfcc_RR[1])
    mfcc3_RR = np.mean(mfcc_RR[2])
    mfcc4_RR = np.mean(mfcc_RR[3])
    mfcc5_RR = np.mean(mfcc_RR[4])
    mfcc6_RR = np.mean(mfcc_RR[5])
    mfcc7_RR = np.mean(mfcc_RR[6])
    mfcc8_RR = np.mean(mfcc_RR[7])
    mfcc9_RR = np.mean(mfcc_RR[8])
    mfcc10_RR = np.mean(mfcc_RR[9])
    mfcc11_RR = np.mean(mfcc_RR[10])
    mfcc12_RR = np.mean(mfcc_RR[11])
    mfcc13_RR = np.mean(mfcc_RR[12])
    # MFCC S1
    mfcc_S1 = librosa.feature.mfcc(y=S1, n_mfcc = 13, sr = Fs, n_fft = n_fft, hop_length= hop_length)
    mfcc1_S1 = np.mean(mfcc_S1[0])
    mfcc2_S1 = np.mean(mfcc_S1[1])
    mfcc3_S1 = np.mean(mfcc_S1[2])
    mfcc4_S1 = np.mean(mfcc_S1[3])
    mfcc5_S1 = np.mean(mfcc_S1[4])
    mfcc6_S1 = np.mean(mfcc_S1[5])
    mfcc7_S1 = np.mean(mfcc_S1[6])
    mfcc8_S1 = np.mean(mfcc_S1[7])
    mfcc9_S1 = np.mean(mfcc_S1[8])
    mfcc10_S1 = np.mean(mfcc_S1[9])
    mfcc11_S1 = np.mean(mfcc_S1[10])
    mfcc12_S1 = np.mean(mfcc_S1[11])
    mfcc13_S1 = np.mean(mfcc_S1[12])
    # MFCC  systole
    mfcc_systole = librosa.feature.mfcc(y=systole, n_mfcc = 13, sr = Fs, n_fft = n_fft, hop_length= hop_length)
    mfcc1_systole = np.mean(mfcc_systole[0])
    mfcc2_systole = np.mean(mfcc_systole[1])
    mfcc3_systole = np.mean(mfcc_systole[2])
    mfcc4_systole = np.mean(mfcc_systole[3])
    mfcc5_systole = np.mean(mfcc_systole[4])
    mfcc6_systole = np.mean(mfcc_systole[5])
    mfcc7_systole = np.mean(mfcc_systole[6])
    mfcc8_systole = np.mean(mfcc_systole[7])
    mfcc9_systole = np.mean(mfcc_systole[8])
    mfcc10_systole = np.mean(mfcc_systole[9])
    mfcc11_systole = np.mean(mfcc_systole[10])
    mfcc12_systole = np.mean(mfcc_systole[11])
    mfcc13_systole = np.mean(mfcc_systole[12])
    # MFCC  S2
    mfcc_S2 = librosa.feature.mfcc(y=S2, n_mfcc = 13, sr = Fs, n_fft = n_fft, hop_length= hop_length)
    mfcc1_S2 = np.mean(mfcc_S2[0])
    mfcc2_S2 = np.mean(mfcc_S2[1])
    mfcc3_S2 = np.mean(mfcc_S2[2])
    mfcc4_S2 = np.mean(mfcc_S2[3])
    mfcc5_S2 = np.mean(mfcc_S2[4])
    mfcc6_S2 = np.mean(mfcc_S2[5])
    mfcc7_S2 = np.mean(mfcc_S2[6])
    mfcc8_S2 = np.mean(mfcc_S2[7])
    mfcc9_S2 = np.mean(mfcc_S2[8])
    mfcc10_S2 = np.mean(mfcc_S2[9])
    mfcc11_S2 = np.mean(mfcc_S2[10])
    mfcc12_S2 = np.mean(mfcc_S2[11])
    mfcc13_S2 = np.mean(mfcc_S2[12])
    # MFCC diastole
    mfcc_diastole = librosa.feature.mfcc(y=diastole, n_mfcc = 13, sr = Fs, n_fft = n_fft, hop_length= hop_length)
    mfcc1_diastole = np.mean(mfcc_diastole[0])
    mfcc2_diastole = np.mean(mfcc_diastole[1])
    mfcc3_diastole = np.mean(mfcc_diastole[2])
    mfcc4_diastole = np.mean(mfcc_diastole[3])
    mfcc5_diastole = np.mean(mfcc_diastole[4])
    mfcc6_diastole = np.mean(mfcc_diastole[5])
    mfcc7_diastole = np.mean(mfcc_diastole[6])
    mfcc8_diastole = np.mean(mfcc_diastole[7])
    mfcc9_diastole = np.mean(mfcc_diastole[8])
    mfcc10_diastole = np.mean(mfcc_diastole[9])
    mfcc11_diastole = np.mean(mfcc_diastole[10])
    mfcc12_diastole = np.mean(mfcc_diastole[11])
    mfcc13_diastole = np.mean(mfcc_diastole[12])

    # RMS features
    RMS_RR = np.mean(librosa.feature.rms(y=RR,frame_length=n_fft, hop_length=hop_length)[0])
    RMS_S1 = np.mean(librosa.feature.rms(y=S1,frame_length=n_fft, hop_length=hop_length)[0])
    RMS_systole = np.mean(librosa.feature.rms(y=systole,frame_length=n_fft, hop_length=hop_length)[0])
    RMS_S2 = np.mean(librosa.feature.rms(y=S2,frame_length=n_fft, hop_length=hop_length)[0])
    RMS_diastole = np.mean(librosa.feature.rms(y=diastole,frame_length=n_fft, hop_length=hop_length)[0])
    RMS_ratio_S1_RR = np.round(RMS_S1/RMS_RR, 4)
    RMS_ratio_systole_RR = np.round(RMS_systole/RMS_RR, 4)
    RMS_ratio_S2_RR = np.round(RMS_S2/RMS_RR, 4)
    RMS_ratio_diastole_RR = np.round(RMS_diastole/RMS_RR, 4)
    RMS_ratio_systole_S1 = np.round(RMS_systole/RMS_S1, 4)
    RMS_ratio_diastole_S2 = np.round(RMS_diastole/RMS_S2, 4)
    RMS_ratio_systole_diastole = np.round(RMS_systole/RMS_diastole, 4)
    RMS_ratio_S1_S2 = np.round(RMS_S1/RMS_S2, 4)

    # Skewness features
    skew_RR = stats.skew(RR)
    skew_S1 = stats.skew(S1)
    skew_systole = stats.skew(systole)
    skew_S2 = stats.skew(S2)
    skew_diastole = stats.skew(diastole)

    # Kurtosis features
    kurtosis_RR = stats.kurtosis(RR)
    kurtosis_S1 = stats.kurtosis(S1)
    kurtosis_systole = stats.kurtosis(systole)
    kurtosis_S2 = stats.kurtosis(S2)
    kurtosis_diastole = stats.kurtosis(diastole)

    # Spectral centroid features
    spec_centroid_RR = np.mean(librosa.feature.spectral_centroid(y=RR, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_centroid_S1 = np.mean(librosa.feature.spectral_centroid(y=S1, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_centroid_systole = np.mean(librosa.feature.spectral_centroid(y=systole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_centroid_S2 = np.mean(librosa.feature.spectral_centroid(y=S2, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_centroid_diastole = np.mean(librosa.feature.spectral_centroid(y=diastole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    
    # Spectral bandwidth features
    spec_bandwidth_RR = np.mean(librosa.feature.spectral_bandwidth(y=RR, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_bandwidth_S1 = np.mean(librosa.feature.spectral_bandwidth(y=S1, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_bandwidth_systole = np.mean(librosa.feature.spectral_bandwidth(y=systole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_bandwidth_S2 = np.mean(librosa.feature.spectral_bandwidth(y=S2, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_bandwidth_diastole = np.mean(librosa.feature.spectral_bandwidth(y=diastole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    
    fmin = 25 # Frequency cutoff for the first bin [0, fmin] Subsequent bins will cover [fmin, 2*fmin]`, `[2*fmin, 4*fmin], etc.
    # Spectral contrast RR
    spec_contrast_RR = librosa.feature.spectral_contrast(y=RR, sr=Fs, n_fft=n_fft, fmin = fmin, hop_length=hop_length, n_bands=5)
    spec_contrast1_RR = np.mean(spec_contrast_RR[0])
    spec_contrast2_RR = np.mean(spec_contrast_RR[1])
    spec_contrast3_RR = np.mean(spec_contrast_RR[2])
    spec_contrast4_RR = np.mean(spec_contrast_RR[3])
    spec_contrast5_RR = np.mean(spec_contrast_RR[4])
    #spec_contrast6_RR = np.mean(spec_contrast_RR[5])
    # Spectral contrast S1
    spec_contrast_S1 = librosa.feature.spectral_contrast(y=S1, sr=Fs, n_fft=n_fft, fmin = fmin, hop_length=hop_length, n_bands=5)
    spec_contrast1_S1 = np.mean(spec_contrast_S1[0])
    spec_contrast2_S1 = np.mean(spec_contrast_S1[1])
    spec_contrast3_S1 = np.mean(spec_contrast_S1[2])
    spec_contrast4_S1 = np.mean(spec_contrast_S1[3])
    spec_contrast5_S1 = np.mean(spec_contrast_S1[4])
    #spec_contrast6_S1 = np.mean(spec_contrast_S1[5])
    # Spectral contrast  systole
    spec_contrast_systole = librosa.feature.spectral_contrast(y=systole, sr=Fs, n_fft=n_fft, fmin = fmin, hop_length=hop_length, n_bands=5)
    spec_contrast1_systole = np.mean(spec_contrast_systole[0])
    spec_contrast2_systole = np.mean(spec_contrast_systole[1])
    spec_contrast3_systole = np.mean(spec_contrast_systole[2])
    spec_contrast4_systole = np.mean(spec_contrast_systole[3])
    spec_contrast5_systole = np.mean(spec_contrast_systole[4])
    #spec_contrast6_systole = np.mean(spec_contrast_systole[5])
    # Spectral contrast  S2
    spec_contrast_S2 = librosa.feature.spectral_contrast(y=S2, sr=Fs, n_fft=n_fft, fmin = fmin, hop_length=hop_length, n_bands=5)
    spec_contrast1_S2 = np.mean(spec_contrast_S2[0])
    spec_contrast2_S2 = np.mean(spec_contrast_S2[1])
    spec_contrast3_S2 = np.mean(spec_contrast_S2[2])
    spec_contrast4_S2 = np.mean(spec_contrast_S2[3])
    spec_contrast5_S2 = np.mean(spec_contrast_S2[4])
    #spec_contrast6_S2 = np.mean(spec_contrast_S2[5])
    # Spectral contrast diastole
    spec_contrast_diastole = librosa.feature.spectral_contrast(y=diastole, sr=Fs, n_fft=n_fft, fmin = fmin, hop_length=hop_length, n_bands=5)
    spec_contrast1_diastole = np.mean(spec_contrast_diastole[0])
    spec_contrast2_diastole = np.mean(spec_contrast_diastole[1])
    spec_contrast3_diastole = np.mean(spec_contrast_diastole[2])
    spec_contrast4_diastole = np.mean(spec_contrast_diastole[3])
    spec_contrast5_diastole = np.mean(spec_contrast_diastole[4])
    #spec_contrast6_diastole = np.mean(spec_contrast_diastole[5])
    
    # Spectral flatness features
    spec_flatness_RR = np.mean(librosa.feature.spectral_flatness(y=RR, n_fft=n_fft, hop_length=hop_length)[0])
    spec_flatness_S1 = np.mean(librosa.feature.spectral_flatness(y=S1,  n_fft=n_fft, hop_length=hop_length)[0])
    spec_flatness_systole = np.mean(librosa.feature.spectral_flatness(y=systole, n_fft=n_fft, hop_length=hop_length)[0])
    spec_flatness_S2 = np.mean(librosa.feature.spectral_flatness(y=S2, n_fft=n_fft, hop_length=hop_length)[0])
    spec_flatness_diastole = np.mean(librosa.feature.spectral_flatness(y=diastole, n_fft=n_fft, hop_length=hop_length)[0])

    # Spectral roll-off features
    spec_rolloff_RR = np.mean(librosa.feature.spectral_rolloff(y=RR, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_rolloff_S1 = np.mean(librosa.feature.spectral_rolloff(y=S1, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_rolloff_systole = np.mean(librosa.feature.spectral_rolloff(y=systole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_rolloff_S2 = np.mean(librosa.feature.spectral_rolloff(y=S2, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    spec_rolloff_diastole = np.mean(librosa.feature.spectral_rolloff(y=diastole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    
    # Poly features
    poly_features_RR = np.mean(librosa.feature.poly_features(y=RR, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    poly_features_S1 = np.mean(librosa.feature.poly_features(y=S1, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    poly_features_systole = np.mean(librosa.feature.poly_features(y=systole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    poly_features_S2 = np.mean(librosa.feature.poly_features(y=S2, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])
    poly_features_diastole = np.mean(librosa.feature.poly_features(y=diastole, sr=Fs, n_fft=n_fft, hop_length=hop_length)[0])

    # Sample entropy features
    sample_entropy_RR = ent.sample_entropy(RR)
    sample_entropy_S1 = ent.sample_entropy(S1)
    sample_entropy_systole = ent.sample_entropy(systole)
    sample_entropy_S2 = ent.sample_entropy(S2)
    sample_entropy_diastole = ent.sample_entropy(diastole)
    
    # Wavelet coefficients
    dwt_RR = pywt.wavedec(RR, wavelet = 'db4', level = 5)
    cA_RR, cD_5_RR, cD_4_RR, cD_3_RR, cD_2_RR, cD_1_RR = dwt_RR
    dwt_S1 = pywt.wavedec(S1, wavelet = 'db4', level = 5)
    cA_S1, cD_5_S1, cD_4_S1, cD_3_S1, cD_2_S1, cD_1_S1 = dwt_S1
    dwt_systole = pywt.wavedec(systole, wavelet = 'db4', level = 5)
    cA_systole, cD_5_systole, cD_4_systole, cD_3_systole, cD_2_systole, cD_1_systole = dwt_systole
    dwt_S2 = pywt.wavedec(S2, wavelet = 'db4', level = 5)
    cA_S2, cD_5_S2, cD_4_S2, cD_3_S2, cD_2_S2, cD_1_S2 = dwt_S2
    dwt_diastole = pywt.wavedec(diastole, wavelet = 'db4', level = 5)
    cA_diastole, cD_5_diastole, cD_4_diastole, cD_3_diastole, cD_2_diastole, cD_1_diastole = dwt_diastole  
    
    # CREATE A FEATURE VECTOR
    vec = pd.Series(dtype=np.float64)
    vec['class'] = label
    vec['wav'] =  wav
    vec['segment'] =  segment
    vec['sig_qual'] =  sig_qual
    vec['split'] =  split

    vec['BPM'] = BPM
    vec['Dur_RR'] = duration_RR
    vec['Dur_S1'] = duration_S1
    vec['Dur_Sys'] = duration_systole
    vec['Dur_S2'] = duration_S2
    vec['Dur_Dia'] = duration_diastole
    vec['Dur_Ratio_S1RR'] = duration_ratio_S1_RR
    vec['Dur_Ratio_SysRR'] = duration_ratio_systole_RR
    vec['Dur_Ratio_S2RR'] = duration_ratio_S2_RR
    vec['Dur_Ratio_DiaRR'] = duration_ratio_diastole_RR
    vec['Dur_Ratio_S1S2'] = duration_ratio_S1_S2
    vec['Dur_Ratio_SysDia'] = duration_ratio_systole_diastole
    vec['Dur_Ratio_SysS1'] = duration_ratio_systole_S1
    vec['Dur_Ratio_DiaS2'] = duration_ratio_diastole_S2
    
    vec['MaxAmp_S1'] = max_amplitude_S1
    vec['MaxAmp_Sys'] = max_amplitude_systole
    vec['MaxAmp_S2'] = max_amplitude_S2
    vec['MaxAmp_Dia'] = max_amplitude_diastole
    vec['MaxAmp_Ratio_S1S2'] = max_amplitude_ratio_S1_S2
    vec['MaxAmp_Ratio_SysS1'] = max_amplitude_ratio_systole_S1
    vec['MaxAmp_Ratio_SysS2'] = max_amplitude_ratio_systole_S2
    vec['MaxAmp_Ratio_DiaS1'] = max_amplitude_ratio_diastole_S1
    vec['MaxAmp_Ratio_DiaS2'] = max_amplitude_ratio_diastole_S2
    vec['MaxAmp_Ratio_SysDia'] = max_amplitude_ratio_systole_diastole
    
    vec['EnvInt_Ratio_S1RR'] = envelope_integral_ratio_S1_RR
    vec['EnvInt_Ratio_SysRR'] = envelope_integral_ratio_systole_RR
    vec['EnvInt_Ratio_S2RR'] = envelope_integral_ratio_S2_RR
    vec['EnvInt_Ratio_DiaRR'] = envelope_integral_ratio_diastole_RR
    vec['EnvInt_Ratio_S1S2'] = envelope_integral_ratio_S1_S2
    vec['EnvInt_Ratio_SysDia'] = envelope_integral_ratio_systole_diastole
    vec['EnvInt_Ratio_SysS1'] = envelope_integral_ratio_systole_S1
    vec['EnvInt_Ratio_DiaS2'] = envelope_integral_ratio_diastole_S2
    
    vec['MeanEnv_RR'] = mean_envelope_RR # Used for throwing out the bad segments   
    vec['MeanEnv_Ratio_S1RR'] = mean_envelope_ratio_S1_RR
    vec['MeanEnv_Ratio_SysRR'] = mean_envelope_ratio_systole_RR
    vec['MeanEnv_Ratio_S2RR'] = mean_envelope_ratio_S2_RR
    vec['MeanEnv_Ratio_DiaRR'] = mean_envelope_ratio_diastole_RR
    vec['MeanEnv_Ratio_S1S2'] = mean_envelope_ratio_S1_S2
    vec['MeanEnv_Ratio_SysDia'] = mean_envelope_ratio_systole_diastole
    vec['MeanEnv_Ratio_SysS1'] = mean_envelope_ratio_systole_S1
    vec['MeanEnv_Ratio_DiaS2'] = mean_envelope_ratio_diastole_S2
        
    vec['PSD_RR_25_40Hz'] = mean_psd_RR_25_40_hz
    vec['PSD_RR_40_60Hz'] = mean_psd_RR_40_60_hz
    vec['PSD_RR_60_80Hz'] = mean_psd_RR_60_80_hz
    vec['PSD_RR_80_100Hz'] = mean_psd_RR_80_100_hz
    vec['PSD_RR_100_120Hz'] = mean_psd_RR_100_120_hz
    vec['PSD_RR_120_140Hz'] = mean_psd_RR_120_140_hz
    vec['PSD_RR_140_160Hz'] = mean_psd_RR_140_160_hz
    vec['PSD_RR_160_180Hz'] = mean_psd_RR_160_180_hz
    vec['PSD_RR_180_200Hz'] = mean_psd_RR_180_200_hz
    vec['PSD_RR_200_250Hz'] = mean_psd_RR_200_250_hz
    vec['PSD_RR_250_300Hz'] = mean_psd_RR_250_300_hz
    vec['PSD_RR_300_400Hz'] = mean_psd_RR_300_400_hz

    vec['PSD_Sys_25_40Hz'] = mean_psd_systole_25_40_hz
    vec['PSD_Sys_40_60Hz'] = mean_psd_systole_40_60_hz
    vec['PSD_Sys_60_80Hz'] = mean_psd_systole_60_80_hz
    vec['PSD_Sys_80_100Hz'] = mean_psd_systole_80_100_hz
    vec['PSD_Sys_100_120Hz'] = mean_psd_systole_100_120_hz
    vec['PSD_Sys_120_140Hz'] = mean_psd_systole_120_140_hz
    vec['PSD_Sys_140_160Hz'] = mean_psd_systole_140_160_hz
    vec['PSD_Sys_160_180Hz'] = mean_psd_systole_160_180_hz
    vec['PSD_Sys_180_200Hz'] = mean_psd_systole_180_200_hz
    vec['PSD_Sys_200_250Hz'] = mean_psd_systole_200_250_hz
    vec['PSD_Sys_250_300Hz'] = mean_psd_systole_250_300_hz
    vec['PSD_Sys_300_400Hz'] = mean_psd_systole_300_400_hz

    vec['PSD_Dia_25_40Hz'] = mean_psd_diastole_25_40_hz
    vec['PSD_Dia_40_60Hz'] = mean_psd_diastole_40_60_hz
    vec['PSD_Dia_60_80Hz'] = mean_psd_diastole_60_80_hz
    vec['PSD_Dia_80_100Hz'] = mean_psd_diastole_80_100_hz
    vec['PSD_Dia_100_120Hz'] = mean_psd_diastole_100_120_hz
    vec['PSD_Dia_120_140Hz'] = mean_psd_diastole_120_140_hz
    vec['PSD_Dia_140_160Hz'] = mean_psd_diastole_140_160_hz
    vec['PSD_Dia_160_180Hz'] = mean_psd_diastole_160_180_hz
    vec['PSD_Dia_180_200Hz'] = mean_psd_diastole_180_200_hz
    vec['PSD_Dia_200_250Hz'] = mean_psd_diastole_200_250_hz
    vec['PSD_Dia_250_300Hz'] = mean_psd_diastole_250_300_hz
    vec['PSD_Dia_300_400Hz'] = mean_psd_diastole_300_400_hz

    vec['ZC_RR'] = zero_crossing_RR
    vec['ZC_S1'] = zero_crossing_S1
    vec['ZC_Sys'] = zero_crossing_systole
    vec['ZC_S2'] = zero_crossing_S2
    vec['ZC_Dia'] = zero_crossing_diastole
    
    vec['chroma_stft1_RR'] = chroma_stft1_RR
    vec['chroma_stft1_S1'] = chroma_stft1_S1
    vec['chroma_stft1_Sys'] = chroma_stft1_systole
    vec['chroma_stft1_S2'] = chroma_stft1_S2
    vec['chroma_stft1_Dia'] = chroma_stft1_diastole
    
    vec['melspectrogram1_RR'] = melspectrogram1_RR
    vec['melspectrogram1_S1'] = melspectrogram1_S1
    vec['melspectrogram1_Sys'] = melspectrogram1_systole
    vec['melspectrogram1_S2'] = melspectrogram1_S2
    vec['melspectrogram1_Dia'] = melspectrogram1_diastole
    
    vec['mfcc1_RR'] = mfcc1_RR
    vec['mfcc2_RR'] = mfcc2_RR
    vec['mfcc3_RR'] = mfcc3_RR
    vec['mfcc4_RR'] = mfcc4_RR
    vec['mfcc5_RR'] = mfcc5_RR
    vec['mfcc6_RR'] = mfcc6_RR
    vec['mfcc7_RR'] = mfcc7_RR
    vec['mfcc8_RR'] = mfcc8_RR
    vec['mfcc9_RR'] = mfcc9_RR
    vec['mfcc10_RR'] = mfcc10_RR
    vec['mfcc11_RR'] = mfcc11_RR
    vec['mfcc12_RR'] = mfcc12_RR
    vec['mfcc13_RR'] = mfcc13_RR
    vec['mfcc1_S1'] = mfcc1_S1
    vec['mfcc2_S1'] = mfcc2_S1
    vec['mfcc3_S1'] = mfcc3_S1
    vec['mfcc4_S1'] = mfcc4_S1
    vec['mfcc5_S1'] = mfcc5_S1
    vec['mfcc6_S1'] = mfcc6_S1
    vec['mfcc7_S1'] = mfcc7_S1
    vec['mfcc8_S1'] = mfcc8_S1
    vec['mfcc9_S1'] = mfcc9_S1
    vec['mfcc10_S1'] = mfcc10_S1
    vec['mfcc11_S1'] = mfcc11_S1
    vec['mfcc12_S1'] = mfcc12_S1
    vec['mfcc13_S1'] = mfcc13_S1
    vec['mfcc1_Sys'] = mfcc1_systole
    vec['mfcc2_Sys'] = mfcc2_systole
    vec['mfcc3_Sys'] = mfcc3_systole
    vec['mfcc4_Sys'] = mfcc4_systole
    vec['mfcc5_Sys'] = mfcc5_systole
    vec['mfcc6_Sys'] = mfcc6_systole
    vec['mfcc7_Sys'] = mfcc7_systole
    vec['mfcc8_Sys'] = mfcc8_systole
    vec['mfcc9_Sys'] = mfcc9_systole
    vec['mfcc10_Sys'] = mfcc10_systole
    vec['mfcc11_Sys'] = mfcc11_systole
    vec['mfcc12_Sys'] = mfcc12_systole
    vec['mfcc13_Sys'] = mfcc13_systole
    vec['mfcc1_S2'] = mfcc1_S2
    vec['mfcc2_S2'] = mfcc2_S2
    vec['mfcc3_S2'] = mfcc3_S2
    vec['mfcc4_S2'] = mfcc4_S2
    vec['mfcc5_S2'] = mfcc5_S2
    vec['mfcc6_S2'] = mfcc6_S2
    vec['mfcc7_S2'] = mfcc7_S2
    vec['mfcc8_S2'] = mfcc8_S2
    vec['mfcc9_S2'] = mfcc9_S2
    vec['mfcc10_S2'] = mfcc10_S2
    vec['mfcc11_S2'] = mfcc11_S2
    vec['mfcc12_S2'] = mfcc12_S2
    vec['mfcc13_S2'] = mfcc13_S2
    vec['mfcc1_Dia'] = mfcc1_diastole
    vec['mfcc2_Dia'] = mfcc2_diastole
    vec['mfcc3_Dia'] = mfcc3_diastole
    vec['mfcc4_Dia'] = mfcc4_diastole
    vec['mfcc5_Dia'] = mfcc5_diastole
    vec['mfcc6_Dia'] = mfcc6_diastole
    vec['mfcc7_Dia'] = mfcc7_diastole
    vec['mfcc8_Dia'] = mfcc8_diastole
    vec['mfcc9_Dia'] = mfcc9_diastole
    vec['mfcc10_Dia'] = mfcc10_diastole
    vec['mfcc11_Dia'] = mfcc11_diastole
    vec['mfcc12_Dia'] = mfcc12_diastole
    vec['mfcc13_Dia'] = mfcc13_diastole
    
    vec['RMS_RR'] = RMS_RR     
    vec['RMS_S1'] = RMS_S1
    vec['RMS_Sys'] = RMS_systole
    vec['RMS_S2'] = RMS_S2
    vec['RMS_Dia'] = RMS_diastole
    vec['RMS_Ratio_S1RR'] = RMS_ratio_S1_RR
    vec['RMS_Ratio_SysRR'] = RMS_ratio_systole_RR
    vec['RMS_Ratio_S2RR'] = RMS_ratio_S2_RR
    vec['RMS_Ratio_DiaRR'] = RMS_ratio_diastole_RR
    vec['RMS_Ratio_SysS1'] = RMS_ratio_systole_S1
    vec['RMS_Ratio_DiaS2'] = RMS_ratio_diastole_S2
    vec['RMS_Ratio_SysDia'] = RMS_ratio_systole_diastole
    vec['RMS_Ratio_S1S2'] = RMS_ratio_S1_S2

    vec['Skewness_RR'] = skew_RR
    vec['Skewness_S1'] = skew_S1
    vec['Skewness_Sys'] = skew_systole
    vec['Skewness_S2'] = skew_S2
    vec['Skewness_Dia'] = skew_diastole

    vec['Kurtosis_RR'] = kurtosis_RR
    vec['Kurtosis_S1'] = kurtosis_S1
    vec['Kurtosis_Sys'] = kurtosis_systole
    vec['Kurtosis_S2'] = kurtosis_S2
    vec['Kurtosis_Dia'] = kurtosis_diastole
    
    vec['SpecCentroid_RR'] = spec_centroid_RR
    vec['SpecCentroid_S1'] = spec_centroid_S1
    vec['SpecCentroid_Sys'] = spec_centroid_systole
    vec['SpecCentroid_S2'] = spec_centroid_S2
    vec['SpecCentroid_Dia'] = spec_centroid_diastole
    
    vec['SpecBandwidth_RR'] = spec_bandwidth_RR
    vec['SpecBandwidth_S1'] = spec_bandwidth_S1
    vec['SpecBandwidth_Sys'] = spec_bandwidth_systole
    vec['SpecBandwidth_S2'] = spec_bandwidth_S2
    vec['SpecBandwidth_Dia'] = spec_bandwidth_diastole
    
    vec['SpecContrast2_RR'] = spec_contrast2_RR
    vec['SpecContrast3_RR'] = spec_contrast3_RR
    vec['SpecContrast4_RR'] = spec_contrast4_RR
    vec['SpecContrast5_RR'] = spec_contrast5_RR

    vec['SpecContrast2_S1'] = spec_contrast2_S1
    vec['SpecContrast3_S1'] = spec_contrast3_S1
    vec['SpecContrast4_S1'] = spec_contrast4_S1
    vec['SpecContrast5_S1'] = spec_contrast5_S1

    vec['SpecContrast2_Sys'] = spec_contrast2_systole
    vec['SpecContrast3_Sys'] = spec_contrast3_systole
    vec['SpecContrast4_Sys'] = spec_contrast4_systole
    vec['SpecContrast5_Sys'] = spec_contrast5_systole

    vec['SpecContrast2_S2'] = spec_contrast2_S2
    vec['SpecContrast3_S2'] = spec_contrast3_S2
    vec['SpecContrast4_S2'] = spec_contrast4_S2
    vec['SpecContrast5_S2'] = spec_contrast5_S2

    vec['SpecContrast2_Dia'] = spec_contrast2_diastole
    vec['SpecContrast3_Dia'] = spec_contrast3_diastole
    vec['SpecContrast4_Dia'] = spec_contrast4_diastole
    vec['SpecContrast5_Dia'] = spec_contrast5_diastole
        
    vec['SpecFlatness_RR'] = spec_flatness_RR
    vec['SpecFlatness_S1'] = spec_flatness_S1
    vec['SpecFlatness_Sys'] = spec_flatness_systole
    vec['SpecFlatness_S2'] = spec_flatness_S2
    vec['SpecFlatness_Dia'] = spec_flatness_diastole
    
    vec['SpecRolloff_RR'] = spec_rolloff_RR
    vec['SpecRolloff_S1'] = spec_rolloff_S1
    vec['SpecRolloff_Sys'] = spec_rolloff_systole
    vec['SpecRolloff_S2'] = spec_rolloff_S2
    vec['SpecRolloff_Dia'] = spec_rolloff_diastole
    
    vec['PolyFeatures_RR'] = poly_features_RR
    vec['PolyFeatures_S1'] = poly_features_S1
    vec['PolyFeatures_Sys'] = poly_features_systole
    vec['PolyFeatures_S2'] = poly_features_S2
    vec['PolyFeatures_Dia'] = poly_features_diastole

    vec['SE_RR'] = sample_entropy_RR
    vec['SE_S1'] = sample_entropy_S1
    vec['SE_Sys'] = sample_entropy_systole
    vec['SE_S2'] = sample_entropy_S2
    vec['SE_Dia'] = sample_entropy_diastole
    
    vec['dwt5_RR'] = np.mean(cD_5_RR)
    vec['dwt4_RR'] = np.mean(cD_4_RR)
    vec['dwt3_RR'] = np.mean(cD_3_RR)
    vec['dwt2_RR'] = np.mean(cD_2_RR)
    vec['dwt1_RR'] = np.mean(cD_1_RR)
    vec['dwt5_S1'] = np.mean(cD_5_S1)
    vec['dwt4_S1'] = np.mean(cD_4_S1)
    vec['dwt3_S1'] = np.mean(cD_3_S1)
    vec['dwt2_S1'] = np.mean(cD_2_S1)
    vec['dwt1_S1'] = np.mean(cD_1_S1)
    vec['dwt5_Sys'] = np.mean(cD_5_systole)
    vec['dwt4_Sys'] = np.mean(cD_4_systole)
    vec['dwt3_Sys'] = np.mean(cD_3_systole)
    vec['dwt2_Sys'] = np.mean(cD_2_systole)
    vec['dwt1_Sys'] = np.mean(cD_1_systole)
    vec['dwt5_S2'] = np.mean(cD_5_S2)
    vec['dwt4_S2'] = np.mean(cD_4_S2)
    vec['dwt3_S2'] = np.mean(cD_3_S2)
    vec['dwt2_S2'] = np.mean(cD_2_S2)
    vec['dwt1_S2'] = np.mean(cD_1_S2)
    vec['dwt5_Dia'] = np.mean(cD_5_diastole)
    vec['dwt4_Dia'] = np.mean(cD_4_diastole)
    vec['dwt3_Dia'] = np.mean(cD_3_diastole)
    vec['dwt2_Dia'] = np.mean(cD_2_diastole)
    vec['dwt1_Dia'] = np.mean(cD_1_diastole)
    
    return vec

# Used for Calculating the Mean Confidence Interval
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# Outputs Standard Deviation and 95% Confindence Interval
def mean_SD_95CI(data):
    mean = np.mean(data)
    std = np.std(data)
    m, m_low, m_top = mean_confidence_interval(data)
    decimals = ".2f"
    string = f'{format(mean, ".5f")} ({format(std, decimals)}; {format(m_low, decimals)}-{format(m_top, decimals)})'
    return string

# Used for genrating cv folds using recording identifier
def generate_nCV_folds(wavs, fold_number=5, seed=4):
    # Shuffle the list so the partition will be random
    random.Random(seed).shuffle(wavs)
    # Create 'fold_number' partitions
    partitions = [wavs[i::fold_number] for i in range(fold_number)]
    return partitions


def search_space_grid(clf_name, seed):
    grid_LogisticRegression = dict(solver = ['newton-cg', 'lbfgs', 'liblinear'],
                              penalty =  ['none', 'l1', 'l2', 'elasticnet'],
                              C = np.linspace(0.05, 2, 40),
                              max_iter = [50, 100, 150, 200],
                              random_state = [seed])
    grid_DecisionTreeClassifier = dict(criterion = ['gini', 'entropy'],
                                       splitter = ['best', 'random'],
                                       min_samples_split = [4, 10, 16, 24,  30, 
                                                            36, 42, 48,  54,  60,
                                                            66, 72,  78,  84,  90],
                                       max_features = ['auto', 'sqrt', 'log2'],
                                       random_state = [seed])
    grid_RandomForestClassifier = dict(n_estimators = [20, 80,  140,  200],
                                        criterion = ['gini', 'entropy'],
                                        min_samples_split = [4, 10, 16, 24,  30, 
                                                            36, 42, 48,  54,  60,
                                                            66, 72,  78,  84,  90],
                                        max_features = ['auto', 'sqrt', 'log2'],
                                        random_state = [seed])
    grid_KNeighborsClassifier = dict(n_neighbors = [3, 9,  15,  21,  27,
                                                    37,  43,  49,  55,  61, 67, 
                                                     73,  79,  85,  91,  97, 1],
                                    weights = ['uniform', 'distance'],
                                    metric = ['euclidean', 'manhattan', 'minkowski'])
    grid_GaussianNB = dict(var_smoothing =  np.logspace(0,-9, num=100))
    grid_SVC = dict(svc__C = np.linspace(0.05, 3, 60),
                    svc__kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    svc__gamma = ['auto'],
                    svc__probability = [True],
                    svc__random_state = [seed])
    grid_SGDClassifier = dict(loss = ['log'],
                                penalty = ['l2', 'l1', 'elasticnet'],
                              alpha = np.logspace(0,-9, num=100),
                              random_state = [seed])
    grid_GradientBoostingClassifier = dict(loss = ['deviance', 'exponential'],
                                          learning_rate = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                                          n_estimators = [20, 60, 100, 140, 180, 200],
                                          min_samples_split = np.linspace(0.1, 0.5, 12),
                                          max_features = ['auto', 'sqrt', 'log2'],
                                          random_state = [seed])
    grid_LGBMClassifier = dict(learning_rate = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                                n_estimators = [20, 60, 100, 140, 180, 200],
                               random_state = [seed])
    grid_XGBClassifier = dict(learning_rate = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                                n_estimators = [20, 60, 100, 140, 180, 200],
                               max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              verbosity = [0],
                             seed = [seed])
    if clf_name == 'LogisticRegression':
        return grid_LogisticRegression 
    elif clf_name == 'DecisionTreeClassifier':
        return grid_DecisionTreeClassifier 
    elif clf_name == 'RandomForestClassifier':
        return grid_RandomForestClassifier 
    elif clf_name == 'KNeighborsClassifier':
        return grid_KNeighborsClassifier 
    elif clf_name == 'GaussianNB':
        return grid_GaussianNB 
    elif clf_name == 'SVC':
        return grid_SVC 
    elif clf_name == 'SGDClassifier':
        return grid_SGDClassifier
    elif clf_name == 'GradientBoostingClassifier':
        return grid_GradientBoostingClassifier
    elif clf_name == 'LGBMClassifier':
        return grid_LGBMClassifier
    elif clf_name == 'XGBClassifier':
        return grid_XGBClassifier

# Performs train test split models evaluation
def run_experiment(FEATURES, 
                    dataset,
                    n_fraction,
                    keep_only_sd_m_fts=True, # Keep only the SD and mean aggregates of the features
                    majority_vote_prediction=True, # Merge the predictions for segments into prediction for recording
                    tbal=True,
                    seed_data =4,
                    finetune=False,
                    seed=4,
                    EXPERIMENTS='',
                    show_results=True):
    print(f'RUNNING CLASSICAL MACHINE LEARNING MODELS EXPERIMENT ON DATASET: {dataset}')
    random.seed(seed)  # python seed
    np.random.seed(seed) # numpy seed
    # Print out the cross validation settings
    print(f'\tFeatures file: {FEATURES}')
    print(f'\tn fraction: {n_fraction}')
    print(f'\tSeed(data): {seed_data}')
    print(f'\tKeep only mean and SD features: Enabled' if keep_only_sd_m_fts else f'\tKeep only mean and SD features: Disabled')
    print(f'\tMajority vote prediction: Enabled' if majority_vote_prediction else f'\tMajority vote prediction: Disabled')
    print(f'\tFinetuning: Enabled' if finetune else f'\tFinetuning: Disabled')
    print(f'\tSeed: {seed}')
    print(f'\tTrain balance: Enabled' if tbal else f'\tTrain balance: Disabled')
    print(f'\tShow results: Enabled' if show_results else f'\tShow results: Disabled')
    # Create results folder if not already existing
    EXPERIMENTS_DIR = utils.check_folder(os.path.join(EXPERIMENTS, f'{dataset}_classical_nfrac={n_fraction}_tbal={tbal}_finetune={finetune}_seed(data)={seed_data}_seed={seed}'))
    if not os.path.exists(EXPERIMENTS_DIR):
        os.makedirs(EXPERIMENTS_DIR)
    # Open the features
    print('Opening the feature file...')
    fts = pd.read_csv(FEATURES)
    print(f'Features shape: {fts.shape}')
    # Only keep a the same wavs as in NN experiment
    if dataset == 'PhysioNet':
        NFRAC_WAVS = os.path.join(EXPERIMENTS, f'{dataset}_seed(data)={seed_data}_wavs')
        TRAIN_WAV_SUB = os.path.join(NFRAC_WAVS, f'{dataset}_seed(data)={seed_data}_nfrac={n_fraction}_valid=False.txt')
        train_wav_sub = list(np.genfromtxt(TRAIN_WAV_SUB, dtype='str'))
        fts = fts[((fts['split']=='test') | ((fts['split']=='train') & (fts.wav.isin(train_wav_sub))))]
    print('Train length:')
    print('\tWavs:', len(list(set(fts[fts['split']=='train'].wav))))
    print('\tWavs:', train_wav_sub)
    print('\tSegments:', len(fts[fts['split']=='train']))
    print('Test length:')
    print('\tSegments:', len(fts[fts['split']=='test']))
    print('\tWavs:', len(list(set(fts[fts['split']=='test'].wav))))
    # Keep only aggregated features
    non_features = ['class', 'wav', 'segment', 'sig_qual', 'split']
    if keep_only_sd_m_fts:
        fts_start_search = ['m_', 'sd_']
        features_search = [x for x in fts.columns if x.startswith(tuple(fts_start_search))]
        features_search = [x for x in features_search if '_RR' not in x] # Throw away features extracted from RR
        features_search = [x for x in features_search if 'MaxAmp' not in x] # Throw away the MaxAmp features
        features_search = [x for x in features_search if 'EnvInt' not in x] # Throw away the EnvInt features
        features_search = [x for x in features_search if 'dwt5' not in x] # Throw away the level 5 DWT features     
        #features_search = [x for x in features_search if 'dwt4' not in x] # Throw away the level 4 DWT features   
        features_search = [x for x in features_search if 'chroma' not in x] # Throw away the chroma features   
        features_search = [x for x in features_search if 'melspectrogram1' not in x] # Throw away the melspectrogram features
        features_search = features_search + non_features
        fts = fts[features_search]
        print('Non-mean and -SD features were removed')
        print(f'Features shape: {fts.shape}')
    # Fill missing values
    if (fts.isna().any().any()):
        # This is here just to prevent interruptions
        num_missing = fts.isnull().sum().sum()
        print(f'WARNING: There are {num_missing} missing values that will be filled with 0')
        fts.fillna(0, inplace = True)
        warnings.warn('CUSTOM WARNING: Missing values filled with 0!')
    # Define the probabilistic classifiers and their names
    clfs = [
        LogisticRegression(random_state = seed), 
        DecisionTreeClassifier(random_state = seed),
        RandomForestClassifier(random_state = seed),
        KNeighborsClassifier(),
        GaussianNB(),
        make_pipeline(StandardScaler(), SVC(probability=True, gamma='auto', random_state = seed)),
        SGDClassifier(loss = 'log', random_state = seed),
        GradientBoostingClassifier(random_state = seed),
        LGBMClassifier(random_state = seed),
        XGBClassifier(verbosity = 0, seed = seed)
       ]
    clfs_names = [
        'LogisticRegression', 
        'DecisionTreeClassifier', 
        'RandomForestClassifier',
        'KNeighborsClassifier', 
        'GaussianNB',
        'SVC',
        'SGDClassifier',
        'GradientBoostingClassifier',
        'LGBMClassifier',
        'XGBClassifier'
        ]
    clfs_abbrvs = [
        'LR', 
        'DT', 
        'RF',
        'KN', 
        'GNB',
        'SVC',
        'SGD',
        'GB',
        'LGBM',
        'XGB'
        ] 

    # Select features
    seed_valids = [1, 2, 3]
    train_wav_valids = []
    for seed_valid in seed_valids:
        TRAIN_WAV_VALID = os.path.join(NFRAC_WAVS, f'{dataset}_seed(data)={seed_data}_nfrac={n_fraction}_valid=True_seed={seed_valid}.txt')
        train_wav_valid = list(np.genfromtxt(TRAIN_WAV_VALID, dtype='str'))
        train_wav_valids.append(train_wav_valid)
    print('Selecting features...')
    kb_num = 40
    train = fts[fts['split']=='train']
    X_train, y_train = train.drop(columns=non_features), train['class']
    FSfit = mutual_info_classif(X_train, y_train, random_state=seed)
    df_FSfit = pd.DataFrame({'features': X_train.columns, f'MI': FSfit})
    df_FSfit.sort_values(['MI'], ascending = False, inplace = True)
    features_sorted = list(df_FSfit['features'].head(kb_num).values)
    """ df_FSfit_arr = []
    df_FSfit_arr_MI_cols = []
    for i, fold in enumerate(train_wav_valids):
        train = fts[fts.wav.isin(fold)]
        print(f'\t{i+1}/{len(train_wav_valids)}')
        X_train, y_train = train.drop(columns=non_features), train['class']         
        # Select the features based on the train data
        FSfit = mutual_info_classif(X_train, y_train, random_state=seed)
        df_FSfit = pd.DataFrame({'features': X_train.columns, f'fold_{i+1}_MI': FSfit})
        df_FSfit_arr.append(df_FSfit)
        df_FSfit_arr_MI_cols.append(f'fold_{i+1}_MI')
    df_FSfit_all = reduce(lambda  left,right: pd.merge(left,right,on=['features'], how='outer'), df_FSfit_arr)
    df_FSfit_all['mean_fold_holders_MI'] = df_FSfit_all[df_FSfit_arr_MI_cols].values.mean(axis=1)
    df_FSfit_all['sd_fold_holders_MI'] = df_FSfit_all[df_FSfit_arr_MI_cols].values.std(axis=1)
    df_FSfit_all.sort_values(['mean_fold_holders_MI'], ascending = False, inplace = True)
    features_sorted = list(df_FSfit_all['features'].head(kb_num).values) """
    # Initialize dataframe and row
    resultsDataFrame = pd.DataFrame()
    resultsRow = pd.Series(dtype=np.float64)
    # Ignore the warnings - some classifiers output a lot of logs
    warnings.filterwarnings('ignore')
    # Run the cross validation
    for clf, clf_name, clf_abbrv in zip(clfs, clfs_names, clfs_abbrvs):
        if clf_abbrv in ['LGBM']:
            continue
        print(f'Currently working: {clf_name}')
        train, test = fts[fts['split']=='train'], fts[fts['split']=='test']
        X_train, y_train = train.drop(columns=non_features), train['class']
        X_test, y_test = test.drop(columns=non_features), test['class']
        """ print(f'{X_train.shape=}')
        print(f'{y_train.shape=}')
        print(f'{X_test.shape=}')
        print(f'{y_test.shape=}') """

        if finetune:
            print('Finetuning')
            finetune_grid = search_space_grid(clf_name, seed)
            print(finetune_grid)
        else: 
            # Feature selection
            X_train, X_test = X_train[features_sorted], X_test[features_sorted]
            # Turn into numpy arrays since column names include some special characters that classifiers might not like
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
            X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
            # Train and predict
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test) # Returns predicted class
            pred_proba = clf.predict_proba(X_test) # probabilities for each class
        
        if majority_vote_prediction:
            # Merge the predictions for segments into prediction for recording (majority, mean,...)
            merge_seg_preds_for_each_rec = pd.DataFrame()
            merge_seg_preds_for_each_rec['recordingName'] = test['wav']
            merge_seg_preds_for_each_rec['pred'] = pred
            merge_seg_preds_for_each_rec['pred_proba'] = list(pred_proba)
            merge_seg_preds_for_each_rec['y_test'] = test['class']

            pred_proba = []
            y_test = []
            for rec in merge_seg_preds_for_each_rec['recordingName'].unique():
                df_rec = merge_seg_preds_for_each_rec[merge_seg_preds_for_each_rec['recordingName'] == rec]
                mean_pred_proba = df_rec['pred_proba'].mean()
                yyy = df_rec['y_test'].mean()
                pred_proba.append(mean_pred_proba)
                y_test.append(yyy)  
            pred = [np.argmax(x) for x in pred_proba]
            pred_probas_1 = [x[1] for x in pred_proba]
                            
        # Specificity and sensitiviy
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        #Calculate Accuracy 
        acc = accuracy_score(y_test, pred)
        #Calculate Precision
        prec = precision_score(y_test, pred)
        #Calculate Recall
        rec = recall_score(y_test, pred)
        #Calculate F1 score
        f1 = f1_score(y_test, pred)
        #Calculate Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        roc_auc = roc_auc_score(y_test, pred_probas_1)

        # Save some static data
        resultsRow['Classifier'] = clf_abbrv
        resultsRow['Specificity'] = specificity
        resultsRow['Sensitivity'] = sensitivity
        resultsRow['Accuracy'] = acc
        resultsRow['Precision'] = prec
        resultsRow['Recall'] = rec
        resultsRow['F1'] = f1
        resultsRow['ROCAUC'] = roc_auc

        # Add the model results to the dataframe
        resultsDataFrame = pd.concat([resultsDataFrame, resultsRow], axis=1)

    # Transpose the result dataframe 
    resultsDataFrame = resultsDataFrame.T.reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(resultsDataFrame)
    # Save the result dataframe
    RESULTS = 'results.csv'
    RESULTS= os.path.join(EXPERIMENTS_DIR, RESULTS)
    resultsDataFrame.to_csv(RESULTS)

    return
