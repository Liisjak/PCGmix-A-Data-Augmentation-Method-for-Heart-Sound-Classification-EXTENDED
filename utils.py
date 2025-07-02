import numpy as np
import os
from six.moves import cPickle as pickle #for performance
import matplotlib.pyplot as plt
from matplotlib import gridspec

def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
    
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def experiment_already_done(args):
    EXPERIMENT_ARGS = experiment_dir(args)
    MODEL = os.path.join(EXPERIMENT_ARGS, 'model.pth')
    if os.path.exists(MODEL):
        return True
    else:
        return False

def experiment_dir(args):
    EXPERIMENT_ARGS = os.path.join(args.EXPERIMENTS, 
                                '{0}_{1}_{2}_epochs={3}_bs={4}_nfrac={5}_op={6}_sched={7}_lrmax={8}_tbal={9}_chs={10}_gc={11}_seed(data)={12}_valid={13}_seed={14}'.format(
                                args.dataset,
                                args.model,
                                args.method,
                                args.num_epochs,
                                args.batch_size,
                                args.n_fraction,
                                args.op,
                                args.use_sched,
                                args.lr_max,
                                args.train_balance,
                                args.num_channels,
                                args.grad_clip,
                                args.seed_data,
                                args.valid,
                                args.seed,
                                ))
    return EXPERIMENT_ARGS

def read_pkl_acc(EXPERIMENT_ARGS):
    DICT = os.path.join(EXPERIMENT_ARGS, 'accuracy.pkl')
    acc_dict = load_dict(DICT)
    acc_test = acc_dict['test']
    acc_train = acc_dict['train']
    return acc_test, acc_train

def read_pkl_perf(EXPERIMENT_ARGS):
    DICT = os.path.join(EXPERIMENT_ARGS, 'performance.pkl')
    dict = load_dict(DICT)
    return dict

def read_pkl_var(EXPERIMENT_ARGS):
    DICT = os.path.join(EXPERIMENT_ARGS, 'variability.pkl')
    var_dict = load_dict(DICT)
    steps = var_dict['steps']
    base = var_dict['base']
    pairs = var_dict['pairs']
    unique = var_dict['unique']
    return steps, base, pairs, unique
    
def read_pkl_hid_rep(args, split, epoch):
    EXPERIMENT_ARGS = experiment_dir(args)
    DIR = os.path.join(EXPERIMENT_ARGS, 'hid_rep')
    DICT = os.path.join(DIR, f'{split}_hid_rep_{epoch}.pkl')
    hid_dict = load_dict(DICT)
    fts = hid_dict['fts']
    trgts = hid_dict['trgts']
    confs = hid_dict['confs']
    return fts, trgts, confs

def show_spectrogram(spec, frames=[]):
    plt.figure(figsize=(8, 3))
    plt.imshow(spec.detach().cpu().numpy(), origin='lower', aspect=0.2)
    if frames != []:
        plt.axvline(x=frames[0], color='k')
        plt.axvline(x=frames[1], color='k')
        plt.axvline(x=frames[2], color='k')
        plt.axvline(x=frames[3], color='k')
    plt.xlim((0, len(spec.detach().cpu().numpy()[0])-1))
    plt.show()
    plt.close()

# Saliency visualization
def show_sal(saliency):
    # plot the heatmap
    plt.figure(figsize=(5, 2))
    plt.imshow(saliency.detach().cpu().numpy(), cmap='jet')#, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
    plt.close()

# Signal visualization
def show_sig(signal, frames=[], cuts=[], sal=None, ylim_bot=-8, ylim_top=8, sal_max=None):
    # Reduce to 4-th dimension only for easier comparison of aug and original instances
    signal = signal[0, :] # low frequency dimension is easiest to read
    print(f'{signal.shape=}')
    signal = signal[None, :]
    print(f'{signal.shape=}')
    num_channels = 1
    # Only plot max first 4 channels
    num_channels = min(signal.shape[0], 4)
    # Define colormap for saliency
    if sal is not None:
        from  matplotlib.colors import LinearSegmentedColormap
        custom_cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
        custom_cmap='jet' 
    # Signal to numpy
    signal = signal.detach().cpu().numpy()
    # Plot
    fig = plt.figure(figsize=(20, 1.125*num_channels))
    # Set height ratios for subplots
    gs = gridspec.GridSpec(num_channels, 1, height_ratios=[1]*num_channels) 
    axs = []
    for ch in range(num_channels):
        ax = plt.subplot(gs[ch])
        ax.plot(signal[ch], color='k')
        if sal is not None:
            sal_np = sal#.detach().cpu().numpy()
            if sal_max==None:
                sal_max = np.max(abs(sal_np))
            im = ax.imshow(sal_np, extent=(0, 2500, ylim_bot, ylim_top), cmap=custom_cmap, vmin=0, vmax=1, alpha=0.5)
            ax.axis('tight')
            ax.set_ylim(bottom=ylim_bot, top=ylim_top)
            if frames!=[]:
                ax.set_xlim(left=0, right=frames[-1])
                ax.set_xlim(left=0, right=1300)  # for easier comparison between augmented and original instances
        ax.set_ylim(bottom=ylim_bot, top=ylim_top)
        axs.append(ax)

    if frames != []:
        for ax in axs:
            for frame in frames:
                ax.axvline(x=frame, linestyle='--', color='k')
                ax.set_xlim(left=0, right=frames[-1])
                ax.set_xlim(left=0, right=1300)  # for easier comparison between augmented and original instances

    if cuts != []:
        for ax in axs:
            for cut in cuts:
                ax.axvline(x=cut, color='red')

    for i in range(num_channels-1):
        plt.setp(axs[i].get_xticklabels(), visible=False)
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    # Colorbar
    if sal is not None:
        fig.colorbar(im, ax=axs)
    plt.show()
    return

# Databuilder
import io
import pickle
import zlib

def dict2file(dataset, DATASET):
    bytes = io.BytesIO()
    pickle.dump(dataset, bytes)
    # compress bytes
    zbytes = zlib.compress(bytes.getbuffer())
    # create a file and store the bytes to file
    with open(DATASET, 'wb') as fd:
        fd.write(zbytes)
        
def file2dict(DATASET):
    with open(DATASET, 'rb') as fd:
        zbytes = fd.read()
    # decompress bytes
    bytes = zlib.decompress(zbytes)
    return pickle.loads(bytes)
