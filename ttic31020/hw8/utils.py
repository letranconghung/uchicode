from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def softmax(r: np.ndarray) -> np.ndarray:
    '''
    Applies the softmax operation on each row of the input, thus converting
    to normalized probabilities.
    
    Args:
        r: Inputs to normalize. shape (n, k)
    
    Returns:
        p: shape (n, k), where p[t, i] = exp(r[t, i]) / sum_j exp(r[t, j]).
    '''
    stable_num = np.exp(r - np.max(r, axis=1)[:, np.newaxis])
    return stable_num / np.sum(stable_num, axis=1)[:, np.newaxis]


@dataclass
class SGDLogger:
    '''
    Class for recording logs.
    
    Usage:
        ```
        # log attribute can either be given during instantiation
        # or will be populated during SGD using logging_func(w) outputs.
        SGDLogger(
            name='l2_norm_weights',
            logging_func=lambda w: np.linalg.norm(w, 'fro'),
            log=None,
            can_display=True,
            per_epoch=False
        )
        ```
        
    Args:
        name: Name for the logger.
        logging_func: Function from weights to some value that will be logged.
        log: (default None) Logged values.
        can_display: (default True) Flag for whether log can be printed neatly.
        per_epoch: (default False) Flag for whether logging_func(w) should
            be called per epoch in SGD method.
    '''
    name: str
    logging_func: Callable[[np.ndarray], Any] # f : w -> any
    log: Any = None # Initialize to None
    can_display: bool = True # Flag if log is displayable
    per_epoch: bool = False # Flag if logging_func should be called per epoch
    

def SGD(
    w0: np.ndarray,
    grad_calculator: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    m: int,
    batch_size: int = 32,
    eta: float = 0.01,
    n_epochs: int = 10,
    sampling: str = 'epoch_shuffle',
    loggers: List[SGDLogger] = [],
    verbose: bool = True,
    verbose_epoch_interval: int = 1
) -> np.ndarray:
    '''
    Optimizes the parameters initialized at w using MiniBatch SGD on the dataset of size m
    and returns the final parameters of the classifier.
    
    Args:
        w0: Initial parameters for SGD. Any shape.
        grad_calculator: Function with (w, idxs) as inputs where w are parameters
            the same shape as w0 and idxs is an optional array of samples' indices,
            returning an estimate of the gradient at w based on samples with
            those indices.
        m: Size of training set.
        batch_size: (default 32) Size for mini batch.
        eta: (default 0.1) Learning rate of the MiniBatch SGD algorithm.
        n_epochs: (default 10) Number of epochs to train for.
        sampling: (default 'epoch_shuffle') one of: ['cyclic', 'randperm', 'epoch_suffle', 'iid'].
            'cycling': cycle over data in input order.
            'randperm': cycle over a random permutation of data fixed across epochs.
            'epoch_shuffle': cycle over a random permutation of data shuffled randomly every epoch.
            'iid': iid sample from the m points every epoch.
        loggers: (default []) List of SGDLoggers, the logging functions of
            which will be called during training (frequency determined by per_epoch).
        verbose: (default True) Flag to display information while training.
        verbose_epoch_interval: (default 1) How regular verbose info should be displayed.
    
    Returns:
        w: shape (d, num_labels) model parameters
    '''
    assert sampling in ['cycling', 'randperm', 'epoch_shuffle', 'iid'], 'Unknown sampling method'
    
    w = w0

    for logger in loggers:
        if logger.per_epoch:
            logger.log = []

    if sampling == 'randperm':
        # One random permutation for all epochs
        shuffle_idxs = np.random.permutation(m)
    elif sampling == 'cyclic':
        # Cycle over data in input order
        shuffle_idxs = np.arange(m)
    
    for epoch in range(n_epochs):
        if sampling == 'epoch_shuffle':
            # Sample without replacements each epoch,
            # i.e. use an independently sampled permutation each round
            shuffle_idxs = np.random.permutation(m)
        elif sampling == 'iid':
            # iid sampling, as in SGD theory
            shuffle_idxs = np.random.randint(0, high=m, size=m)
        n_batches = m // batch_size
        batch_idxs = np.array_split(shuffle_idxs, n_batches)
        
        # Train on mini batch
        for b in range(n_batches):
            b_idxs = batch_idxs[b] # the samples to use in this minibatch
            
            g = grad_calculator(w, b_idxs) # the stochastic gradient estimate
            w = w - eta * g # gradient step
        
        # Log per epoch loggers
        for logger in loggers:
            if logger.per_epoch:
                logger.log.append(logger.logging_func(w))
        if verbose and (epoch % verbose_epoch_interval == 0):
            if epoch == 0:
                print()
            s = [f'--- Epoch: {epoch}']
            for logger in loggers:
                if logger.can_display and logger.per_epoch:
                    s.append(f'{logger.name}: {logger.log[-1]:5}')
            if len(s) > 1:
                print(', '.join(s))
    
    # Log final loggers
    for logger in loggers:
        if not logger.per_epoch:
            logger.log(logger.logging_func(w))
    if verbose:
        s = [f'Training complete']
        for logger in loggers:
            if logger.can_display and (not logger.per_epoch):
                s.append(f'{logger.name}: {logger.log:5}')
        if len(s) > 1:
            print(', '.join(s))
    
    return w


def relative_error(x, y, h):
    h = h or 1e-12
    if type(x) is np.ndarray and type(y) is np.ndarray:
        top = np.abs(x - y)
        bottom = np.maximum(np.abs(x) + np.abs(y), h)
        return np.amax(top/bottom)
    else:
        return abs(x - y) / max(abs(x) + abs(y), h)

    
def numeric_grad(f, x, df, eps):
    df = df or 1.0
    eps = eps or 1e-8
    n = x.size
    x_flat = x.reshape(n)
    dx_num = np.zeros(x.shape)
    dx_num_flat = dx_num.reshape(n)
    for i in range(n):
        orig = x_flat[i]
    
        x_flat[i] = orig + eps
        pos = f(x)
        if type(df) is np.ndarray:
            pos = pos.copy()
    
        x_flat[i] = orig - eps
        neg = f(x)
        if type(df) is np.ndarray:
            neg = neg.copy()

        d = (pos - neg) * df / (2 * eps)
        
        dx_num_flat[i] = d
        x_flat[i] = orig
    return dx_num

    
# Color maps for 2 labels
cmap_bg = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_fg = ['#FF0000', '#0000FF']


def add_label_noise(y, noise_level: float = 0.) -> np.ndarray:
    '''
    Adds noise to labels and returns a modified array of labels. Labels are {-1, +1} valued.
    Each labels is replaced with a random label with probability noise_level.
    noise_level=0 : no corruption, returns y itself
    noise_level=1 : returns uniformly random labels
    noise_level=0.5 : means approx. 1/2 the labels will be replaced with
    uniformly random labels, so only 1/4 would actually flip.
    
    Args:
        noise_level: probability of corruption
    '''
    
    assert 0 <= noise_level <= 1
    return y * (1 - 2 * (np.random.rand(len(y)) > 1-noise_level/2.0))


def generate_spiral_data(
    m: int,
    noise_level: float = 0.,
    theta_sigma: float = 0.,
    r_sigma: float = 0.
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generates m spiral data points from a distribution specified with theta_sigma
    and r_sigma. Labels are in {-1, +1}. With probability noise_level,
    each label is replaced with a random label.
    '''
    y = 1 - 2*(np.random.rand(m) > 0.5)
    true_r = np.random.rand(m)
    theta = true_r*10 + 5*y + theta_sigma*np.random.randn(m)
    r = (1 + r_sigma*np.random.randn(m))*true_r
    X = np.column_stack((r*np.cos(theta), r*np.sin(theta)))
    y = add_label_noise(y, noise_level)
    return X, y


def plot_decision_boundary(clf, X_train, y_train, X_test, y_test, labels=[-1, 1]):
    '''
    Plots the decision boundary of the given classifier on training and testing points.
    Colors the training points with true labels, and shows the incorrectly and correctly predicted test points.
    '''
    X, y = np.vstack([X_train, X_test]), np.hstack([y_train.flatten(), y_test.flatten()])
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    
    # Create a mesh of points
    eps = 0.
    x1s = np.linspace(x_min[0]-eps, x_max[0]+eps, 100)
    x2s = np.linspace(x_min[1]-eps, x_max[1]+eps, 100)
    xx1, xx2 = np.meshgrid(x1s, x2s)
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx1.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(xx1, xx2, Z, cmap=cmap_bg, shading='auto')

    # Plot training points
    for i, l in enumerate(labels):
        l_idxs = np.where(y_train == l)
        ax.scatter(X_train[l_idxs, 0], X_train[l_idxs, 1], label=f'train/{l}', c=cmap_fg[i], marker='.')
    
    # Plot test points
    y_test_predict = clf.predict(X_test)
    for i, l in enumerate(labels):
        # Mark the wrong ones
        wrong_idxs = np.where((y_test_predict == l) & (y_test_predict != y_test))
        ax.scatter(X_test[wrong_idxs, 0], X_test[wrong_idxs, 1], label=f'test/predicted {l} (wrong)', c=cmap_fg[1-i], marker='x', s=100)

        # Plot the correct ones
        corr_idxs = np.where((y_test_predict == l) & (y_test_predict == y_test))
        ax.scatter(X_test[corr_idxs, 0], X_test[corr_idxs, 1], label=f'test/predicted {l} (correct)', c=cmap_fg[i], marker='+', s=50)
        
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Decision boundary\nShaded regions show what the label clf would predict for a point there')
    plt.legend(title='label', bbox_to_anchor=(1.04, 1), loc='upper left')
    

def create_split(X: np.ndarray, y: np.ndarray, split_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Randomly splits (X, y) into sets (X1, y1, X2, y2) such that
    (X1, y1) contains split_ratio fraction of the data. Rest goes in (X2, y2).
    
    Args:
        X: data features of shape (m, d)
        y: data labels of shape (m)
        split_ratio: fraction of data to keep in (X1, y1) (must be between 0 and 1)
    
    Returns:
        (X1, y1, X2, y2): each is a numpy array
    '''
    assert 0. <= split_ratio <= 1.
    assert X.shape[0] == len(y)
    assert len(y.shape) == 1
    
    m = X.shape[0]
    idxs_shuffled = np.random.permutation(m)
    X_shuffled, y_shuffled = X[idxs_shuffled], y[idxs_shuffled]
    
    m1 = int(split_ratio * m)
    X1, y1 = X_shuffled[:m1], y_shuffled[:m1]
    X2, y2 = X_shuffled[m1:], y_shuffled[m1:]
    
    return (X1, y1, X2, y2)


def empirical_err(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    '''
    Returns the 0-1 empirical error for predictions y_pred against true y_true,
    with each data point weighted by sample_weight. If sample_weight is None,
    each data point is equally weighted. sample_weight must be a probability vector
    if given.
    
    Args:
        y_true: True data labels. shape (m)
        y_pred: Predicted data labels. shape (m)
        sample_weight: (default None) Probability vector to weigh data points. shape (m)
    
    Returns:
        scalar, denoting the empirical error.
    '''
    assert y_true.shape[0] == y_pred.shape[0]
    m = y_true.shape[0]

    if sample_weight is None:
        sample_weight = np.ones(m) / m
    else:
        assert y_true.shape[0] == sample_weight.shape[0]
        assert np.isclose(sample_weight.sum(), 1.)
    
    err = np.sum(sample_weight*np.array((y_true != y_pred)).astype(float))
    return err