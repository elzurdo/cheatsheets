from fastai.basics import torch, tensor, nn
import  matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

import ipywidgets as widgets
from ipywidgets import interact, fixed
from scipy.signal import savgol_filter

fontsize=18
plt.rcParams.update({'font.size': fontsize})
rc('xtick', labelsize=fontsize)
rc('ytick', labelsize=fontsize)

w1_true = 3.
w2_true = 2.

def mse(true, pred): return ((true-pred)**2).mean()

def generate_data(size=200, shift_data=0, plot=True, seed=1):
    if seed:
        torch.manual_seed(seed)
    
    x = torch.ones(size,2)  
    x[:,0].uniform_(-1. ,1)
    x[:,0].add_(shift_data)
    
    w1 = w1_true
    w2 = w2_true
    w = tensor(w1, w2)

    noise = (torch.rand(size)  - 0.5) * (shift_data**0.5 + 1) # 0.5 because torch.rand is centeralised around 0.5
    y = (x@w) + noise
    
    if plot:
        plt.scatter(x[:,0], y)
        #plt.plot([0, 1], [0, 1])
        
    return x, y, w


def update_batch(x, y, w_guess, lr=1.e-1, zero_grad=True):
    #print(w_guess.detach().numpy())
    y_pred = x@w_guess
    loss = mse(y, y_pred)
    
    loss.backward()
    
    with torch.no_grad():
        w_guess.sub_(lr * w_guess.grad)
        if zero_grad:
            w_guess.grad.zero_()
            
    return loss, w_guess


def plot_summary(x, y, y_guess, losses, weights_1, weights_2, epsilon=1.e-7, shift_data=0, no_weights=False, title=None):
    nplots =3

    if no_weights:
        nplots = 2

    min_x, max_x = float(x[:,0].min()), float(x[:,0].max())
    min_y, max_y = min_x * w1_true + w2_true, max_x * w1_true + w2_true

    plt.figure(figsize=(16/3. * nplots,5))
    plt.subplot(1, nplots, 1)
    plt.scatter(x[:,0],y, color='orange')
    plt.scatter(x[:,0],y_guess)
    plt.plot([min_x, max_x], [min_y, max_y], '--', color='green', alpha=0.7, linewidth=2)
    plt.xlabel('x'); plt.ylabel('y')
    if title:
        plt.title(title)
    
    plt.subplot(1, nplots, 2)
    plt.plot(np.log10( np.array(losses) + epsilon))
    plt.xlabel('iteration')
    plt.ylabel('log10(mse) loss')
    
    if not no_weights:
        plt.subplot(1, 3, 3)
        iterations_ = len(weights_1)
        slope = w1_true
        bias = w2_true 
        plt.plot(weights_1, color='green', label='a')
        plt.plot(weights_2, color='purple', label='b', linewidth=3, alpha=0.7)
        plt.plot([0, iterations_], [slope, slope], '--', alpha=0.7, color='green')
        plt.plot([0, iterations_], [bias, bias], '--', alpha=0.7, color='purple')
        plt.xlabel('iteration')
        plt.legend()
        plt.ylim(0., np.max([slope, bias]) * 1.05)

    plt.tight_layout()


def run_batch(shift_data=0, size=400, lr=1.e-1, iterations=200, zero_grad=True, verbose=False, plot=True, seed=1):

    x, y, w_true = generate_data(size=size, shift_data=shift_data, plot=False, seed=seed)

    w_guess = nn.Parameter(tensor(-1., 1))

    losses, weights_1, weights_2 = [], [], []
    for t in range(iterations): 
        loss, w_guess = update_batch(x, y, w_guess, lr=lr, zero_grad=zero_grad)
        losses.append(float(loss.detach().numpy())); weights_1.append(w_guess.detach().numpy()[0]); weights_2.append(w_guess.detach().numpy()[1])
        
        if (t % (iterations // 10) == 0) & verbose: 
            print(f'MSE {losses[-1]}')

    if plot:
        plot_summary(x, y, x@w_guess.detach().numpy(), losses, weights_1, weights_2, shift_data=shift_data, title='one batch')

    return np.array(losses)


def run_batches(shift_data=0, n_batches=20, batch_size=20, lr=1.e-1, 
               iterations=200, zero_grad=True, verbose=False, plot=True, seed=1):
    
    m = n_batches * batch_size
    x, y, w_true = generate_data(size=m, shift_data=shift_data, plot=False, seed=seed)

    w_guess = nn.Parameter(tensor(-1., 1))

    losses, weights_1, weights_2 = [], [], []

    batch_idx = -1
    for t in range(iterations): 
        # -- batch limits --
        batch_idx += 1
        batch_idx = batch_idx % n_batches
        start = batch_idx * batch_size
        end = start + batch_size
        # ------------------
        x_b = x[start:end].clone() #  clone makes sure we do not override
        y_b = y[start:end] # no need to clone this

        loss, w_guess = update_batch(x_b, y_b, w_guess, lr=lr, zero_grad=zero_grad)
        losses.append(float(loss.detach().numpy())); weights_1.append(w_guess.detach().numpy()[0]); weights_2.append(w_guess.detach().numpy()[1])
        
        if (t % (iterations // 10) == 0) & verbose: 
            print(f'MSE {losses[-1]}')

    if plot:
        plot_summary(x, y, x@w_guess.detach().numpy(), losses, weights_1, weights_2, shift_data=shift_data, title='mini batches')

    return np.array(losses)


def update_batch_norm(x_b, y_b, w_guess, mu, var, gamma, beta, momentum=0.9, lr=1.e-1, zero_grad=True, verbose=False, epsilon=1.e-5):    
    mu_b    = x_b.mean(axis=0)[0]
    var_b   = x_b.var(axis=0)[0]
    
    if momentum:
        mu  = mu  * momentum + mu_b * (1 - momentum)
        var = var * momentum + var_b * (1 - momentum)
    else:
        mu = mu_b
        var = var_b
    
    x_b[:,0].sub_(mu).div_((var + epsilon)**0.5)
    z_b = x_b * gamma + beta
    
    y_pred = z_b@w_guess   
    loss = mse(y_b, y_pred)

    loss.backward()
    
    with torch.no_grad():
        w_guess.sub_(lr * w_guess.grad)
        gamma.sub_(lr * gamma.grad)
        beta.sub_(lr * beta.grad)
        if zero_grad:
            w_guess.grad.zero_()
            gamma.grad.zero_()
            beta.grad.zero_()
            
    return loss, w_guess, mu, var

def _params_to_weights(gamma, beta, w, mu, var):
    sigma = var ** 0.5

    w1 = gamma[0] * w[0] / sigma
    w2 = w[0] * (beta[0] - gamma[0] * mu/sigma) + w[1] *(beta[1] + gamma[1])

    return w1, w2


def run_batch_norm(shift_data=0, momentum=0.9, n_batches=10, batch_size=20, lr=1.e-1,
                   iterations=100, zero_grad=True, verbose=False, epsilon=1.e-7, plot=True,
                   title='batch norm - momentum ', seed=1):
    
    m = n_batches * batch_size
    x, y, w_true = generate_data(size=m, shift_data=shift_data, plot=False, seed=seed)

    w_guess = nn.Parameter(tensor(-1., 1))
    gamma = nn.Parameter(tensor(-1., 1.))
    beta = nn.Parameter(tensor(-1., 1.))

    losses, weights_1, weights_2 = [], [], []

    batch_idx = -1
    for t in range(iterations): #(batch_size * n_batches // 2):
        # -- batch limits --
        batch_idx += 1
        batch_idx = batch_idx % n_batches
        start = batch_idx * batch_size
        end = start + batch_size
        # ------------------
        x_b = x[start:end].clone() #  clone makes sure we do not override
        y_b = y[start:end] # no need to clone this

        if not momentum:
            mu, var = None, None
            if (t == 0):
                title += '0'
        else:
            if (t == 0):
                mu  = x_b.mean(axis=0)[0]
                var = x_b.var(axis=0)[0]
                title += f'{momentum}'

        loss, w_guess, mu, var = update_batch_norm(x_b, y_b, w_guess, mu, var, gamma, beta, momentum=momentum, lr=lr, zero_grad=zero_grad, epsilon=epsilon)
        w1_guess, w2_guess = _params_to_weights(gamma.detach().numpy(), beta.detach().numpy(), w_guess.detach().numpy(), mu.detach().numpy(), var.detach().numpy())

        losses.append(float(loss.detach().numpy())); weights_1.append(w1_guess); weights_2.append(w2_guess)
        #print(gamma.detach().numpy(), beta.detach().numpy(), w_guess.detach().numpy(), mu.detach().numpy(), var.detach().numpy())
        if (t % 10 == 0) & verbose: 
            print(f'MSE {losses[-1]}')


    x_ = x.clone()
    x_[:,0].sub_(mu).div_((var + epsilon)**0.5)
    z_ = x_ * gamma + beta

    if plot:
        plot_summary(x, y, (z_@w_guess).detach().numpy(), losses, weights_1, weights_2, shift_data=shift_data, no_weights=False, title=title)

    return np.array(losses)


def _integers_to_widget(ints, continuous_update=False):
    return widgets.IntSlider(min=ints[0], max=ints[1], step=ints[2], continuous_update=continuous_update)

shift_data = [0, 1, 2, 3, 5, 10, 50]
lrs = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]  # learning rates
batch_sizes = [20, 40, 80]
momentums = [0., 0.1, 0.5, 0.9, 0.99, 0.999]
epsilons = [1.e-5, 1.e-6, 1.e-7]

sizes = (50, 400, 50)
n_batches = (20, 100, 10)
iterations = (50, 1000, 10)
seeds = (1, 20, 1)
continuous_update = False
sizes = _integers_to_widget(sizes, continuous_update=continuous_update)
n_batches = _integers_to_widget(n_batches, continuous_update=continuous_update)
iterations = _integers_to_widget(iterations, continuous_update=continuous_update)
seeds = _integers_to_widget(seeds, continuous_update=continuous_update)

def run_one_batch_interactive():
    return interact(run_batch, size=sizes, shift_data=shift_data, lr=lrs, iterations=iterations, seed=seeds, title=fixed('one batch'), plot=fixed(True))

def run_batches_interactive():
    return interact(run_batches, shift_data=shift_data, lr=lrs, iterations=iterations, batch_size=batch_sizes, n_batches=n_batches, seed=seeds, title=fixed('mini batch'), plot=fixed(True))

def run_batch_norm_interactive():
    return interact(run_batch_norm, shift_data=shift_data, momentum=momentums, n_batches=n_batches, batch_size=batch_sizes, lr=lrs, 
               iterations=iterations, epsilon=epsilons, seed=seeds, title=fixed('batch norm - momentum '), plot=fixed(True))

def run_comparison():
    interact(_run_comparison, shift_data=shift_data, n_batches=n_batches, batch_size=batch_sizes, lr=lrs, iterations=iterations, momentum=momentums, epsilon=epsilons, plot_all=False)

def _run_comparison(shift_data=0, n_batches=20, momentum=0.9, batch_size=20, lr=1.e-1, iterations=200, epsilon=1.e-7, plot_all=False):
    losses = {}

    size = n_batches * batch_size

    losses['one batch'] = run_batch(size=size, shift_data=shift_data, lr=lr, iterations=iterations, plot=plot_all)
    losses['mini batches'] = run_batches(shift_data=shift_data, lr=lr, iterations=iterations, batch_size=batch_size, n_batches=n_batches, plot=plot_all)
    losses['batch norm - momentum 0'] = run_batch_norm(shift_data=shift_data, momentum=0, n_batches=n_batches, batch_size=batch_size, lr=lr,
               iterations=iterations, epsilon=epsilon, plot=plot_all)
    losses[f'batch norm - momentum {momentum})'] = run_batch_norm(shift_data=shift_data, momentum=momentum, n_batches=n_batches, batch_size=batch_size, lr=lr,
               iterations=iterations, epsilon=epsilon, plot=plot_all)

    plt.figure(figsize=(16, 8))

    epsilon = 1.e-5
    for idx, item in enumerate(losses):
        width = idx + 1
        l_ = losses[item]
        l_log10 = np.log10(losses[item] + epsilon)
        plt.subplot(1, 2, 1)
        plt.plot(l_log10, linewidth=width)

        plt.subplot(1, 2, 2)
        l_ = savgol_filter(l_, 21, 2)
        l_log10 = np.log10(l_ + epsilon)
        plt.plot(l_log10, label=item, linewidth=width)

    plt.title('Smoothed')
    plt.ylabel('log10(MSE)')
    plt.xlabel('iteration')
    plt.legend(fontsize=20)

    plt.subplot(1, 2, 1)
    plt.title('Actual')
    plt.ylabel('log10(MSE)')
    plt.xlabel('iteration')

    plt.tight_layout()


