import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import mixture
import scipy

import utils

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_epoch_loss(epoch_loss_corr, epoch_loss_incorr, epoch, EXPERIMENT_ARGS, show = False):
    epoch_loss_all = np.append(epoch_loss_corr, epoch_loss_incorr)
    # normalize the losses
    epoch_loss_corr = epoch_loss_corr / np.max(epoch_loss_all)
    epoch_loss_incorr = epoch_loss_incorr / np.max(epoch_loss_all)
    bins = np.linspace(0, 1, 100)
    fig = plt.figure(figsize = (6, 6))
    plt.hist(epoch_loss_corr, bins, alpha=0.5, label='correct', color = 'royalblue')
    plt.hist(epoch_loss_incorr, bins, alpha=0.5, label='incorrect', color = 'crimson')
    plt.title(f'Epoch={epoch}')
    plt.xlabel('normalized loss')
    plt.ylabel('#samples')
    plt.legend()
    plt.grid()
    EXPERIMENT_ARGS_LOSS = utils.check_folder(os.path.join(EXPERIMENT_ARGS, 'losses'))
    FILENAME = os.path.join(EXPERIMENT_ARGS_LOSS, f'epoch_loss_{epoch}.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()

def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

def plot_epoch_loss_dst(epoch_loss_corr, epoch_loss_incorr, epoch, EXPERIMENT_ARGS, show = False):
    epoch_loss_all = np.append(epoch_loss_corr, epoch_loss_incorr)
    # normalize the losses
    epoch_loss_corr = epoch_loss_corr / np.max(epoch_loss_all)
    epoch_loss_incorr = epoch_loss_incorr / np.max(epoch_loss_all)
    epoch_loss_all = np.append(epoch_loss_corr, epoch_loss_incorr)
    # fit the mixed gaussian model and plot the fitted curves
    gm = mixture.GaussianMixture(n_components=2, random_state=4).fit(epoch_loss_all.reshape(-1, 1))
    gm_weights = gm.weights_
    gm_means = gm.means_.ravel()
    gm_covar = gm.covariances_.ravel()
    
    smpl_x = np.linspace(0, 1, 100)
    smpl_y_mixture = np.exp(gm.score_samples(smpl_x.reshape(-1, 1)))
    smpl_y_comp1 = gauss_function(x = smpl_x, amp = 1, x0 = gm_means[0], sigma = np.sqrt(gm_covar[0]))
    smpl_y_comp1 = smpl_y_comp1 /np.trapz(smpl_y_comp1, smpl_x) * gm_weights[0] # normalize
    smpl_y_comp2 = gauss_function(x = smpl_x, amp = 1, x0 = gm_means[1], sigma = np.sqrt(gm_covar[1]))
    smpl_y_comp2 = smpl_y_comp2 /np.trapz(smpl_y_comp2, smpl_x) * gm_weights[1] # normalize
    
    bins = np.linspace(0, 1, 100)
    fig = plt.figure(figsize = (6, 6))
    plt.hist(epoch_loss_all, bins, alpha=0.5, density = True, color = 'grey')
    plt.plot(smpl_x, smpl_y_mixture, color = 'k', label = 'gaussian mixture')
    plt.plot(smpl_x, smpl_y_comp1, linestyle = '--', color = 'royalblue', label = rf'component, $\mu_1$={np.round(gm_means[0], 2)}')
    plt.plot(smpl_x, smpl_y_comp2, linestyle = '--', color = 'crimson', label = rf'component, $\mu_2$={np.round(gm_means[1], 2)}')
    plt.axvline(x = gm_means[0], linestyle = '--', color = 'k', alpha = 0.8)
    plt.axvline(x = gm_means[1], linestyle = '--', color = 'k', alpha = 0.8)
    plt.title(rf'epoch {epoch};   |$\mu_1$ - $\mu_2$| = {np.abs(np.round(gm_means[1]-gm_means[0], 2))}')
    plt.xlabel('normalized loss')
    plt.ylabel('probability densitys')
    plt.legend()
    plt.grid()
    EXPERIMENT_ARGS_LOSS = utils.check_folder(os.path.join(EXPERIMENT_ARGS, 'losses'))
    FILENAME = os.path.join(EXPERIMENT_ARGS_LOSS, f'epoch_loss_dst_{epoch}.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()
    
    return np.abs(gm_means[1]-gm_means[0])

def plot_train_test_acc(acc_train, acc_test, valid, steps, EXPERIMENT_ARGS):
    acc_test_max = np.max(acc_test)
    step_max = steps[acc_test.index(acc_test_max)]
    acc_test_max = np.round(acc_test_max, 2)
    acc_test_fin = np.round(acc_test[-1], 2)
    fig = plt.figure(figsize = (6, 6))
    valid_str = 'valid' if valid else 'test'
    valid_color = 'royalblue' if valid else 'forestgreen'
    plt.plot(steps, acc_train, label='train', color = 'darkorange')
    plt.plot(steps, acc_test, label = valid_str, color = valid_color)
    plt.axhline(y = acc_test_max, color = valid_color, linestyle = '--', label = f'{valid_str} max {acc_test_max} @step {step_max}')
    plt.axhline(y = acc_test_fin, color = valid_color, linestyle = '-.', label = f'{valid_str} final {acc_test_fin}')
    plt.ylim(bottom = 0, top = 110)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.grid()
    PLOT = os.path.join(EXPERIMENT_ARGS, f'accuracy.jpg')
    plt.savefig(PLOT)
    plt.close()

def plot_train_test_loss(loss_train, loss_test, valid, steps, EXPERIMENT_ARGS):
    fig = plt.figure(figsize = (6, 6))
    valid_str = 'valid' if valid else 'test'
    valid_color = 'royalblue' if valid else 'forestgreen'
    train_final = np.round(loss_train[-1], 2)
    valid_final = np.round(loss_test[-1], 2)
    plt.plot(steps, loss_train, label='train', color = 'darkorange')
    plt.axhline(y = train_final, color = 'darkorange', linestyle = '-.', label = f'train final {train_final}')
    plt.plot(steps, loss_test, label = valid_str, color = valid_color)
    plt.axhline(y = valid_final, color = valid_color, linestyle = '-.', label = f'{valid_str} final {valid_final}')
    #plt.ylim(bottom = 0, top = 110)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    FILENAME = os.path.join(EXPERIMENT_ARGS, f'loss.jpg')
    plt.savefig(FILENAME)
    plt.close()

def plot_variability(variability_counter, EXPERIMENT_ARGS):
    steps = variability_counter.steps
    base = variability_counter.lens_base
    pairs = variability_counter.lens_pairs
    unique = variability_counter.lens_unique
    fig = plt.figure(figsize = (6, 6))
    plt.plot(steps, base, label='base', color = 'darkorange')
    plt.axhline(y=variability_counter.base_original, label='base_original', color = 'darkorange', linestyle='--')
    plt.plot(steps, pairs, label = 'pairs', color = 'forestgreen')
    plt.plot(steps, unique, label = 'unique', color = 'purple', linestyle='--')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative samples')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    var_dict = {'base':base, 'pairs':pairs, 'unique':unique, 'steps':steps}
    DICT = os.path.join(EXPERIMENT_ARGS, f'variability.pkl')
    utils.save_dict(var_dict, DICT)
    PLOT = os.path.join(EXPERIMENT_ARGS, f'variability.jpg')
    plt.savefig(PLOT)
    plt.close()
        
def plot_times(times, steps, EXPERIMENT_ARGS, show = False):
    times_sum = np.sum(times)
    hours, rem = divmod(times_sum, 3600)
    minutes, seconds = divmod(rem, 60)
        
    fig = plt.figure(figsize = (6, 6))
    plt.plot(steps, times, label = r'times', color = 'k')
    plt.ylim(bottom = 0)
    plt.xlabel('Steps')
    plt.ylabel(r'times [s]')
    plt.title("Total {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    plt.legend()
    plt.grid()
    FILENAME = os.path.join(EXPERIMENT_ARGS, f'times.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()
                
def plot_lr_per_step(lr_per_step, EXPERIMENT_ARGS, show = False):
    num_steps = len(lr_per_step)
        
    fig = plt.figure(figsize = (6, 6))
    plt.plot(np.arange(1, num_steps+1, 1), lr_per_step, label = f'learning_rate', color = 'k')
    plt.ylim(bottom = 0)
    plt.xlabel('Step')
    plt.ylabel(r'Learning rate')
    plt.legend()
    plt.grid()
    FILENAME = os.path.join(EXPERIMENT_ARGS, f'learning_rate.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()

def plot_m1(gmm_m1s, EXPERIMENT_ARGS, show = False):
    num_epochs = len(gmm_m1s)
    m1_max = np.max(gmm_m1s)
    epoch_max = np.arange(1, num_epochs+1, 1)[gmm_m1s.index(m1_max)]
    
    fig = plt.figure(figsize = (6, 6))
    plt.plot(np.arange(1, num_epochs+1, 1), gmm_m1s, label = r'$M_1$', color = 'rebeccapurple')
    plt.scatter(epoch_max, m1_max, color = 'k', label = f'max@epoch {epoch_max}')
    plt.ylim(bottom = 0)
    plt.xlabel('Epoch')
    plt.ylabel(r'$M_1$')
    plt.legend()
    plt.grid()
    FILENAME = os.path.join(EXPERIMENT_ARGS, f'm1.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()