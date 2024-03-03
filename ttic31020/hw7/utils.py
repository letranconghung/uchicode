import os
import sys
from typing import Optional, Tuple, Union

import gzip
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, transform

import numpy as np
import urllib
from tqdm import tqdm

try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import URLError
    from urllib import urlretrieve


def report_download_progress(chunk_number: int, chunk_size: int, file_size: int):
    '''
    Hook for urlretrieve for reporting progress.
    '''
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write(f'\r0% |{bar:<64}| {int(percent * 100):d}')


def download(destination_path: str, url: str, verbose: bool = True):
    '''
    Downloads resource at url to destination_path (must exist).
    '''
    try:
        hook = report_download_progress if verbose else None
        urlretrieve(url, destination_path, reporthook=hook)
    except URLError as e:
        raise RuntimeError(f'Error downloading from {url}') from e


def unzip(zipped_path: str):
    '''
    Unzips gzipped file at zipped_path.
    '''
    unzipped_path = os.path.splitext(zipped_path)[0]
    with gzip.open(zipped_path, 'rb') as zipped_file:
        with open(unzipped_path, 'wb') as unzipped_file:
            unzipped_file.write(zipped_file.read())
    

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
    
    err = np.sum(sample_weight*(y_true != y_pred).astype(float))
    return err


def load_raw_vj_data(pos_path="./data/faces_selected/", neg_path="./data/not_faces_selected/", cap_class=np.inf):
    paths = [pos_path, neg_path]
    pos_img_path_list = os.listdir(pos_path)
    neg_img_path_list = os.listdir(neg_path)
    np.random.shuffle(pos_img_path_list)
    np.random.shuffle(neg_img_path_list)
    positives = []
    negatives = []
    img_types = [positives, negatives]
    for i, img_list in enumerate([pos_img_path_list, neg_img_path_list]):
        for j, img_fname in tqdm(enumerate(img_list)):
            if img_fname == ".DS_Store":
                continue
            if j > cap_class:
                break
            img = mpimg.imread(os.path.join(paths[i], img_fname))
            img_types[i].append(img)
    y = np.hstack((np.ones(len(positives)), -np.ones(len(negatives))))
    X_raw = positives + negatives
    idxs = np.random.permutation(range(y.shape[0]))
    return [X_raw[i] for i in idxs], y[idxs] ## returns list of raw images and their labels

def plot_data_grid(class_1, class_2, h=3, w=3, title_string="images"):
    f, axarr = plt.subplots(h,2*w+1, figsize=(14,6))#, gridspec_kw={'hspace': 0.1})
    f.tight_layout()
    for i in tqdm(range(h)):
        for j in range(2*w+1):
            if j < w:
                axarr[i,j].imshow(class_1[i*w + j])# , aspect = "auto")
            elif j == w:
                axarr[i,j].imshow(np.ones((2,2,3)))# , aspect = "auto")
            elif j > w:
                axarr[i,j].imshow(transform.resize(class_2[i*w + j-w], class_1[0].shape))
            axarr[i,j].axis('off')
    f.subplots_adjust(hspace=0.0, wspace=0.0)#, right=0.7)
    
    f.suptitle(title_string)
    plt.show()