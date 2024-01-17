from typing import Tuple, List, Tuple

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Color maps for 2 labels
cmap_bg = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_fg = ['#FF0000', '#0000FF']


class TrainAndTestData:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def print_errors(self, clf):
        # Get training error
        train_err = empirical_err(clf, self.X_train, self.y_train)
        print(f'Train error: {train_err*100:0.2f}%')
        # Get test error
        test_err = empirical_err(clf, self.X_test, self.y_test)
        print(f'Test error: {test_err*100:0.2f}%')


class ConstantFalsePredictor:
    '''
    Use as:
        ```
        constant_clf = ConstantFalsePredictor()
        y_test_predict = constant_clf.predict(X_test)
        ```
    '''
    def __init__(self):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Return an array if -1's to any input

        Args:
            X: data features

        Returns:
            y: labels, same number as the data points
        '''

        return np.full(X.shape[0], fill_value = -1)


class WordConjPredictor:
    '''
    Use as:
        ```
        word_clf = WordConjPredictor()
        word_clf.program_word(word_list)  OR  word_clf.fit(X_train,y_train,num_words,count_threshold)
        y_test_predict = word_clf.predict(X_test)
        ```
    '''
    def __init__(self):
        self.spam_words = []

    def program_words(self, spam_words: List = []):
        '''Hard-code the words that indicate spamness'''
        self.spam_words = spam_words

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predicts the labels for data X based whether any of the words in self.spam_words can be found in the sentences

        Args:
            X: data features

        Returns:
            y: labels, same number as the data points
        '''
        y = []
        for x in X:
            xsplit = x.lower().split()
            is_spam = -1
            for word in self.spam_words:
                if word in xsplit:
                    is_spam = 1

            y.append(is_spam)

        return np.array(y)

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


def create_split(X: np.ndarray, y: np.ndarray, split_ratio: float, seed = 310202024) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Randomly splits (X, y) into sets (X1, y1, X2, y2) such that
    (X1, y1) contains split_ratio fraction of the data. Rest goes in (X2, y2).

    Args:
        X: data features of shape (m, d)
        y: data labels of shape (m)
        split_ratio: fraction of data to keep in (X1, y1) (must be between 0 and 1)
        seed (defaults to the arbirtrary number 31020): a seed to use for the random number generator.
        Using a hard coded seed ensures the same split every time the function is called.

    Returns:
        (X1, y1, X2, y2): each is a numpy array
    '''
    assert 0. <= split_ratio <= 1.
    assert X.shape[0] == len(y)
    assert len(y.shape) == 1

    m = X.shape[0]
    # The following line creates an independent source of psuedo-random numbers, which doesn't
    # affect subequent number generation that won't use the created rng.  We use this so that we
    # can get consistent splits in this routine, by using a fixed seed, without making subsequent
    # random numbers also be the same on each execution.
    rng = np.random.default_rng(seed)
    idxs_shuffled = rng.permutation(m)
    X_shuffled, y_shuffled = X[idxs_shuffled], y[idxs_shuffled]

    m1 = int(split_ratio * m)
    X1, y1 = X_shuffled[:m1], y_shuffled[:m1]
    X2, y2 = X_shuffled[m1:], y_shuffled[m1:]

    return (X1, y1, X2, y2)


def empirical_err(predictor, X, y):
    """
    Returns the empirical error of the predictor on the given sample.

    Args:
        predictor-- an object with predictor.predict(x) method
        X: array of input instances
        y: array of true (correct) labels

    Returns:
        err: empirical error value
    """
    assert len(X) == len(y)

    pred_y = predictor.predict(X)
    err = np.mean(y != pred_y)

    return err


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


def scatter_plot(X, y, labels, **plot_kwargs):
    plt.figure(figsize=(8, 6))
    for i, l in enumerate(labels):
        l_idxs = np.where(y == l)
        plt.scatter(X[l_idxs, 0], X[l_idxs, 1], label=l, c=cmap_fg[i], **plot_kwargs)
    plt.xlabel('$x_1$') # matplotlib allows basic latex in rendered text!
    plt.ylabel('$x_2$')
    plt.legend(title='label')
    
    
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess the text and converts the labels to ints for SMS spam data.
    Returns the dataframe.
    '''
    # Label to binary values
    data['label'] = 2*(data['label'] == 'spam') - 1
    
    # Remove punctuation from text
    data['text'] = data['text'].str.replace('[^\w\s]', '', regex=True)
    
    return data
    
def read_sms_spam_data(filepath: str, sep: str = '\t') -> Tuple[np.ndarray, np.ndarray]:
    '''
    Reads SMS Spam data from filepath stored as a CSV with separator sep. The
    first column is the label name (ham or spam) and the second column is the text.
    There are no header lines; data starts from the first line of the file.
    
    Args:
        filepath: path to CSV file
        sep: separator in the CSV file
    
    Returns:
        (text, label)
    '''
    # Read file
    data = pd.read_csv(filepath, sep=sep, header=None)
    data = data.drop_duplicates().reset_index(drop=True)
    data.columns = ['label', 'text']
    
    # Preprocess data
    data = preprocess_data(data)
    
    return data['text'].values, data['label'].values