import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy
from sklearn.metrics import davies_bouldin_score
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sympy.solvers import solve
from sympy import Symbol
import random

import utils
import models

if False: # activate/deactivate
    # Load the pretrained model for hidden space generation
    ROOT = os.path.join('..', '..', '..', 'mnt', 'eol', 'Zacasno', 'davidsusic', 'CHF')
    EXPERIMENT_ARGS = utils.check_folder(os.path.join(ROOT, 'experiments'))
    MODEL_LATENT = os.path.join(EXPERIMENT_ARGS, 
                        'PhysioNet_ResCNN_base_epochs=10_bs=32_nfrac=1.0_op=adam_sched=True_lrmax=0.00089_tbal=True_chs=4_gc=0.1_seed(data)=3_valid=False_seed=1', 
                        'model.pth')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Latent space device set to: cuda')
    else: 
        device = torch.device('cpu')
        print('Latent space device set to: cpu')
    model_latent = models.ResCNN_TS(num_channels=4, num_classes=2)
    model_latent = nn.DataParallel(model_latent) # sets to all available cuda devices
    model_latent.to(device)
    model_latent.load_state_dict(torch.load(MODEL_LATENT))
    model_latent.eval()
    print('Latent space model parameters have been loaded')

def generate_latent_space(data):
    with torch.no_grad():
        fts_latent = model_latent(data, depth=5, pass_part='first')
        fts_latent = fts_latent.cpu().detach().numpy()
    return fts_latent

def save_latent_space(dct, split, step, RESULTS_ARGS):
    LATENT_SPACE = utils.check_folder(os.path.join(RESULTS_ARGS, 'latent_space'))
    FN = os.path.join(LATENT_SPACE, f'latent_space_{split}_{step}.pkl')
    utils.save_dict(dct, FN)

### OLD

def get_latent_space_features(data, model):
    ''' Works with input data only (depth=0) '''
    model.eval()
    with torch.no_grad():
        fts = model(data, 0, pass_part = 'latent_space')
        #confs = model(fts, 0, pass_part = 'hidden_rep_to_confs')
        fts = fts.cpu().detach().numpy()
        #confs = confs.cpu().detach().numpy()
    return fts#, confs

def get_hidden_features(model, data_loader, split, device):

    _loader, _, _ = data_loader.run(mode = split, transform_seed = -1)
    
    fts = np.empty(shape = (0, 32)) # Tweak the number of hidden space dimensions
    trgts = []
    confs = np.empty(shape = (0, 10))
    indcs = []
        
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, indices) in enumerate(_loader):
            data = data.to(device)
            out = model(data, 0, pass_part = 'hidden_rep')
            out_conf = model(out, 0, pass_part = 'hidden_rep_to_confs')
            out = out.cpu().detach().numpy()
            out_conf = out_conf.cpu().detach().numpy()
            
            fts = np.concatenate((fts, out), axis = 0)
            confs = np.concatenate((confs, out_conf), axis = 0)
            trgts = trgts + list(target.cpu().detach().numpy())
            indcs = indcs + list(indices.cpu().detach().numpy())
    model.train()
                
    return fts, trgts, confs, indcs
        
def dim_reduc_tsne(fts, fts_new, num_components=2):
    # First we use PCA to reduce to 50 dimensions
    fts_len = fts.shape[0]
    # Combine both features
    fts_all = np.concatenate((fts, fts_new), axis = 0)
    reduced_dim = 50
    """ # Transform both at once
    pca = PCA(n_components=reduced_dim)
    fts_all = pca.fit_transform(fts_all) """
    np.random.seed(4)
    fts_all = TSNE(n_components=num_components, learning_rate='auto', init='random', perplexity=15, random_state = 4).fit_transform(fts_all)
    # Separate the transformed features back
    fts = fts_all[:fts_len, :]
    fts_new = fts_all[fts_len:, :]    
    return fts, fts_new, -1
        
def dim_reduc_pca(fts, fts_new, num_components=2):
    pca = PCA(n_components=num_components)
    # fit on the original data
    pca_fit = pca.fit(fts)
    # transform both original and new data
    fts = pca_fit.transform(fts)
    fts_new = pca_fit.transform(fts_new)
    # get explained variance value
    expl_var = pca.explained_variance_ratio_
    expl_var_tot = np.sum(expl_var)
    return fts, fts_new, expl_var_tot
                
def normalize_points(fts):
    # scale x and y between 0 and 1
    fts_x = fts[:, 0]
    x_range = np.max(fts_x) - np.min(fts_x)
    fts_x = (fts_x - np.min(fts_x)) / x_range
    fts_y = fts[:, 1]
    y_range = np.max(fts_y) - np.min(fts_y)
    fts_y = (fts_y - np.min(fts_y)) / y_range
    fts = np.array(list(zip(fts_x, fts_y)))
    return fts
    
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
    
def plot_latent_space(latent_features, split, epoch, num_classes, method, RESULTS_ARGS, dim_reduc='tsne'):

    fts = latent_features['fts']
    trgts = latent_features['trgts']
    fts_new = latent_features['fts_new']
    trgts_new = latent_features['trgts_new']
    
    # reduce the hidden features to 2 dimensions
    if dim_reduc == 'tsne':
        fts, expl_var_tot = dim_reduc_tsne(fts, num_components=2)
    elif dim_reduc == 'pca':
        fts, fts_new, expl_var_tot = dim_reduc_pca(fts, fts_new, num_components=2)
                
    # normalize data
    maximum0 = np.max(fts[:, 0])
    minimum0 = np.min(fts[:, 0])
    fts[:, 0] = (fts[:, 0] - minimum0) / (maximum0 - minimum0)
    fts_new[:, 0] = (fts_new[:, 0] - minimum0) / (maximum0 - minimum0)
    maximum1 = np.max(fts[:, 1])
    minimum1 = np.min(fts[:, 1])
    fts[:, 1] = (fts[:, 1] - minimum1) / (maximum1 - minimum1)
    fts_new[:, 1] = (fts_new[:, 1] - minimum1) / (maximum1 - minimum1)

    # Plot the points
    fig = plt.figure(figsize = (6, 6))
    color = ['red', 'blue']#plt.cm.gist_rainbow(np.linspace(0,1,num_classes))
    for lbl in range(num_classes):
        idx_lbl = [idx for idx, trgt in enumerate(trgts) if trgt == lbl]
        fts_lbl = fts[idx_lbl]
        # calculate distance matrix and find medoid
        dist_matrix = scipy.spatial.distance_matrix(fts_lbl, fts_lbl)
        medoid_idx = np.argmin(dist_matrix.sum(axis=0))
        # plot the cluster with medoid and centroid
        fts_x_lbl = fts_lbl[:, 0]
        fts_y_lbl = fts_lbl[:, 1]
        plt.scatter(fts_x_lbl, fts_y_lbl, label = f'{lbl}', facecolors = 'none', edgecolors = color[lbl], s = 30, marker ='o', alpha=0.15)
        plt.scatter(np.mean(fts_x_lbl), np.mean(fts_y_lbl), color = color[lbl], marker ='x')
        #plt.scatter(fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx], color = color[lbl])
        plt.annotate(str(lbl), (fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx]))
    if method != 'base':
        color = ['darkred', 'darkblue']
        for lbl in range(num_classes):
            idx_lbl = [idx for idx, trgt in enumerate(trgts_new) if trgt == lbl]
            fts_lbl = fts_new[idx_lbl]
            # calculate distance matrix and find medoid
            dist_matrix = scipy.spatial.distance_matrix(fts_lbl, fts_lbl)
            medoid_idx = np.argmin(dist_matrix.sum(axis=0))
            # plot the cluster with medoid and centroid
            fts_x_lbl = fts_lbl[:, 0]
            fts_y_lbl = fts_lbl[:, 1]
            plt.scatter(fts_x_lbl, fts_y_lbl, label = f'{lbl}_new', facecolors = 'none', edgecolors = color[lbl], s = 30, marker ='P')
            plt.scatter(np.mean(fts_x_lbl), np.mean(fts_y_lbl), color = color[lbl], marker ='x')
            #plt.scatter(fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx], color = color[lbl])
            plt.annotate(str(lbl), (fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx]))
    plt.grid()
    plt.title(f'{dim_reduc}; Data: {split}; Total explained variace: {round(expl_var_tot, 3)}; Epoch: {epoch}')
    plt.legend()
    RESULTS_ARGS_LATENT = utils.check_folder(os.path.join(RESULTS_ARGS, 'latent_space'))
    PLOT = os.path.join(RESULTS_ARGS_LATENT, f'{dim_reduc}_{split}_{epoch}.png')
    plt.savefig(PLOT)
    plt.close()
    return

def plot_latent_space_test(latent_features, split, epoch, num_classes, method, RESULTS_ARGS, dim_reduc='tsne'):

    fts = latent_features['fts']
    trgts = latent_features['trgts']
    
    # reduce the hidden features to 2 dimensions
    if dim_reduc == 'tsne':
        fts, expl_var_tot = dim_reduc_tsne(fts, num_components=2)
    elif dim_reduc == 'pca':
        fts, _, expl_var_tot = dim_reduc_pca(fts, fts, num_components=2)
                
    # normalize data
    print(fts.shape)
    maximum0 = max(np.max(fts[:, 0]), np.max(fts[:, 0]))
    minimum0 = min(np.min(fts[:, 0]), np.min(fts[:, 0]))
    fts[:, 0] = (fts[:, 0] - minimum0) / (maximum0 - minimum0)
    maximum1 = max(np.max(fts[:, 1]), np.max(fts[:, 1]))
    minimum1 = min(np.min(fts[:, 1]), np.min(fts[:, 1]))
    fts[:, 1] = (fts[:, 1] - minimum1) / (maximum1 - minimum1)
    
    # Plot the points
    fig = plt.figure(figsize = (6, 6))
    color = ['red', 'blue']#plt.cm.gist_rainbow(np.linspace(0,1,num_classes))
    for lbl in range(num_classes):
        idx_lbl = [idx for idx, trgt in enumerate(trgts) if trgt == lbl]
        fts_lbl = fts[idx_lbl]
        # calculate distance matrix and find medoid
        dist_matrix = scipy.spatial.distance_matrix(fts_lbl, fts_lbl)
        medoid_idx = np.argmin(dist_matrix.sum(axis=0))
        # plot the cluster with medoid and centroid
        fts_x_lbl = fts_lbl[:, 0]
        fts_y_lbl = fts_lbl[:, 1]
        plt.scatter(fts_x_lbl, fts_y_lbl, label = f'{lbl}', facecolors = 'none', edgecolors = color[lbl], s = 30, marker ='o', alpha=0.15)
        plt.scatter(np.mean(fts_x_lbl), np.mean(fts_y_lbl), color = color[lbl], marker ='x')
        #plt.scatter(fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx], color = color[lbl])
        plt.annotate(str(lbl), (fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx]))
    plt.grid()
    plt.title(f'{dim_reduc}; Data: {split}; Total explained variace: {round(expl_var_tot, 3)}; Epoch: {epoch}')
    plt.legend()
    RESULTS_ARGS_LATENT = utils.check_folder(os.path.join(RESULTS_ARGS, 'latent_space'))
    PLOT = os.path.join(RESULTS_ARGS_LATENT, f'{dim_reduc}_{split}_{epoch}.png')
    plt.savefig(PLOT)
    plt.close()
    return

def plot_latent_space_test_train(latent_features_test, latent_features_train, split, epoch, num_classes, method, RESULTS_ARGS, dim_reduc='tsne'):

    fts_test = latent_features_test['fts']
    trgts_test = latent_features_test['trgts']
    fts_train = latent_features_train['fts_new']
    trgts_train = latent_features_train['trgts_new']
    
    # reduce the hidden features to 2 dimensions
    if dim_reduc == 'tsne':
        fts_test, fts_train, expl_var_tot = dim_reduc_tsne(fts_test, fts_train, num_components=2)
    elif dim_reduc == 'pca':
        fts_test, fts_train, expl_var_tot = dim_reduc_pca(fts_test, fts_train, num_components=2)
                
    # normalize data
    maximum0 = max(np.max(fts_test[:, 0]), np.max(fts_train[:, 0]))
    minimum0 = min(np.min(fts_test[:, 0]), np.min(fts_train[:, 0]))
    fts_test[:, 0] = (fts_test[:, 0] - minimum0) / (maximum0 - minimum0)
    fts_train[:, 0] = (fts_train[:, 0] - minimum0) / (maximum0 - minimum0)
    maximum1 = max(np.max(fts_test[:, 1]), np.max(fts_train[:, 1]))
    minimum1 = min(np.min(fts_test[:, 1]), np.min(fts_train[:, 1]))
    fts_test[:, 1] = (fts_test[:, 1] - minimum1) / (maximum1 - minimum1)
    fts_train[:, 1] = (fts_train[:, 1] - minimum1) / (maximum1 - minimum1)
    
   # Plot the points
    fig = plt.figure(figsize = (6, 6))
    color = ['red', 'blue']#plt.cm.gist_rainbow(np.linspace(0,1,num_classes))
    for lbl in range(num_classes):
        idx_lbl = [idx for idx, trgt in enumerate(trgts_test) if trgt == lbl]
        fts_lbl = fts_test[idx_lbl]
        # calculate distance matrix and find medoid
        dist_matrix = scipy.spatial.distance_matrix(fts_lbl, fts_lbl)
        medoid_idx = np.argmin(dist_matrix.sum(axis=0))
        # plot the cluster with medoid and centroid
        fts_x_lbl = fts_lbl[:, 0]
        fts_y_lbl = fts_lbl[:, 1]
        plt.scatter(fts_x_lbl, fts_y_lbl, label = f'{lbl} test', facecolors = 'none', edgecolors = color[lbl], s = 30, marker ='o', alpha=0.05)
        plt.scatter(np.mean(fts_x_lbl), np.mean(fts_y_lbl), color = color[lbl], marker ='x')
        #plt.scatter(fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx], color = color[lbl])
        plt.annotate(str(lbl), (fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx]))
    plt.grid()
    plt.title(f'{dim_reduc}; Data: {split}(test); Total explained variace: {round(expl_var_tot, 3)}; Epoch: {epoch}')
    plt.legend()
    RESULTS_ARGS_LATENT = utils.check_folder(os.path.join(RESULTS_ARGS, 'latent_space'))
    PLOT = os.path.join(RESULTS_ARGS_LATENT, f'{dim_reduc}_{split}(test)_{epoch}.png')
    plt.savefig(PLOT)
    plt.close()

    fig = plt.figure(figsize = (6, 6))
    color = ['darkred', 'darkblue']
    for lbl in range(num_classes):
        idx_lbl = [idx for idx, trgt in enumerate(trgts_train) if trgt == lbl]
        fts_lbl = fts_train[idx_lbl]
        # calculate distance matrix and find medoid
        dist_matrix = scipy.spatial.distance_matrix(fts_lbl, fts_lbl)
        medoid_idx = np.argmin(dist_matrix.sum(axis=0))
        # plot the cluster with medoid and centroid
        fts_x_lbl = fts_lbl[:, 0]
        fts_y_lbl = fts_lbl[:, 1]
        plt.scatter(fts_x_lbl, fts_y_lbl, label = f'{lbl} train', facecolors = 'none', edgecolors = color[lbl], s = 30, marker ='P', alpha=0.05)
        plt.scatter(np.mean(fts_x_lbl), np.mean(fts_y_lbl), color = color[lbl], marker ='x')
        #plt.scatter(fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx], color = color[lbl])
        plt.annotate(str(lbl), (fts_x_lbl[medoid_idx], fts_y_lbl[medoid_idx]))
    plt.grid()
    plt.title(f'{dim_reduc}; Data: {split}(train); Total explained variace: {round(expl_var_tot, 3)}; Epoch: {epoch}')
    plt.legend()
    RESULTS_ARGS_LATENT = utils.check_folder(os.path.join(RESULTS_ARGS, 'latent_space'))
    PLOT = os.path.join(RESULTS_ARGS_LATENT, f'{dim_reduc}_{split}(train)_{epoch}.png')
    plt.savefig(PLOT)
    plt.close()
    return