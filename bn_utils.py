from fastai.basics import torch, tensor, nn
import  matplotlib.pyplot as plt
import numpy as np

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
    
    y = (x@w) + torch.rand(size) - 0.5 # 0.5 because torch.rand is centeralised around 0.5
    
    if plot:
        plt.scatter(x[:,0], y)
        #plt.plot([0, 1], [0, 1])
        
    return x, y, w


def update_simple(x, y, w_guess, lr=1.e-1, zero_grad=True):
    #print(w_guess.detach().numpy())
    y_pred = x@w_guess
    loss = mse(y, y_pred)
    
    loss.backward()
    
    with torch.no_grad():
        w_guess.sub_(lr * w_guess.grad)
        if zero_grad:
            w_guess.grad.zero_()
            
    return loss, w_guess


def plot_summary(x, y, y_guess, losses, weights_1, weights_2, epsilon=1.e-7, shift_data=0, no_weights=False):
    nplots =3

    if no_weights:
        nplot = 2

    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = min_x * w1_true + w2_true, max_x * w1_true + w2_true

    plt.figure(figsize=(16,5))
    plt.subplot(1, nplots, 1)
    plt.scatter(x[:,0],y, color='orange')
    plt.scatter(x[:,0],y_guess)
    plt.plot([min_x, max_x], [min_y, max_y], '--', color='green', alpha=0.7, linewidth=2)
    
    plt.subplot(1, nplots, 2)
    plt.plot(np.log10( np.array(losses) + epsilon))
    plt.xlabel('iteration')
    plt.ylabel('log10(mse) loss')
    
    if not no_weights:
        plt.subplot(1, 3, 3)
        iterations = len(weights_1)
        slope = w1_true
        bias = w2_true #- shift_data
        plt.plot(weights_1, color='green', label='a')
        plt.plot(weights_2, color='purple', label='b')
        plt.plot([0, iterations], [slope, slope], '--', alpha=0.7, color='green')
        plt.plot([0, iterations], [bias, bias], '--', alpha=0.7, color='purple')
        plt.xlabel('iteration')
        plt.legend()
        plt.ylim(0., np.max([slope, bias]) * 1.05)

    plt.tight_layout()


def run_simple(shift_data=0, n_batches=20, batch_size=20, lr=1.e-1, 
               iterations=200, zero_grad=True, verbose=True):
    
    m = n_batches * batch_size
    x, y, w_true = generate_data(size=m, shift_data=shift_data, plot=False)

    w_guess = nn.Parameter(tensor(-1., 1))

    losses, weights_1, weights_2 = [], [], []
    for t in range(iterations): 
        loss, w_guess = update_simple(x, y, w_guess, lr=lr, zero_grad=zero_grad)
        losses.append(float(loss.detach().numpy())); weights_1.append(w_guess.detach().numpy()[0]); weights_2.append(w_guess.detach().numpy()[1])
        
        if (t % (iterations // 10) == 0) & verbose: 
            print(loss)

    plot_summary(x, y, x@w_guess.detach().numpy(), losses, weights_1, weights_2, shift_data=shift_data)

    return np.array(losses)


def run_batches(shift_data=0, n_batches=20, batch_size=20, lr=1.e-1, 
               iterations=200, zero_grad=True, verbose=True):
    
    m = n_batches * batch_size
    x, y, w_true = generate_data(size=m, 
                                 shift_data=shift_data, plot=False)

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

        loss, w_guess = update_simple(x_b, y_b, w_guess, lr=lr, zero_grad=zero_grad)
        losses.append(float(loss.detach().numpy())); weights_1.append(w_guess.detach().numpy()[0]); weights_2.append(w_guess.detach().numpy()[1])
        
        if (t % (iterations // 10) == 0) & verbose: 
            print(loss)

    plot_summary(x, y, x@w_guess.detach().numpy(), losses, weights_1, weights_2, shift_data=shift_data)

    return np.array(losses)


def update_batch_norm(x_b, y_b, w_guess, mu, var, gamma, beta, momentum=0.9, lr=1.e-1, zero_grad=True, verbose=True, epsilon=1.e-5):    
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

def run_batch_norm(shift_data=0, momentum = 0.9, n_batches=10, batch_size=20, lr=1.e-1, 
               iterations=100, zero_grad=True, verbose=True, epsilon=1.e-7):
    
    m = n_batches * batch_size
    x, y, w_true = generate_data(size=m, 
                                 shift_data=shift_data, plot=False)

    w_guess = nn.Parameter(tensor(-1., 1))
    gamma = nn.Parameter(tensor(-1., 1.))
    beta = nn.Parameter(tensor(-1., 1.))

    losses, weights_1, weights_2 = [], [], []

    batch_idx = -1
    for t in range(batch_size * n_batches // 2): 
        # -- batch limits --
        batch_idx += 1
        batch_idx = batch_idx % n_batches
        start = batch_idx * batch_size
        end = start + batch_size
        # ------------------
        x_b = x[start:end].clone() #  clone makes sure we do not override
        y_b = y[start:end] # no need to clone this

        if momentum is None:
            mu, var = None, None

        else:
            if (t == 0):
                mu  = x_b.mean(axis=0)[0]
                var = x_b.var(axis=0)[0]
            
        #loss, w_guess = update_simple(x_b, y_b, w_guess, lr=lr, zero_grad=zero_grad)
        loss, w_guess, mu, var = update_batch_norm(x_b, y_b, w_guess, mu, var, gamma, beta, momentum=momentum, lr=lr, zero_grad=zero_grad, epsilon=epsilon)
        losses.append(float(loss.detach().numpy())); weights_1.append(w_guess.detach().numpy()[0]); weights_2.append(w_guess.detach().numpy()[1])

        if (t % 10 == 0) & verbose: 
            print(loss)


    #plot_summary(x, y, w_guess, losses, weights_1, weights_2)
    x_ = x.clone()
    x_[:,0].sub_(mu).div_((var + epsilon)**0.5)
    z_ = x_ * gamma + beta

    plot_summary(x, y, (z_@w_guess).detach().numpy(), losses, weights_1, weights_2, shift_data=shift_data, no_weights=True)

    return np.array(losses)


