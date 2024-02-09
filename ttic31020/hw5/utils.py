import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


def create_split(X: np.ndarray, y: np.ndarray, split_ratio: float, seed = 310202024):
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


def scatter_plot(X, y, cmap_fg = {-1:'#FF0000', 1:'#0000FF'}, **plot_kwargs):
    labels = np.unique(y)
    plt.figure(figsize=(8, 6))
    for l in labels:
        l_idxs = np.where(y == l)
        plt.scatter(X[l_idxs, 0], X[l_idxs, 1], label=l, c=cmap_fg[l], **plot_kwargs)


def plot_decision_boundary(clf, X_train, y_train=None, X_test=None, y_test=None, 
                           cmap_bg = ListedColormap(['#FFAAAA', '#AAAAFF']), cmap_fg = {-1:'#FF0000', 1:'#0000FF'}):
    '''
    Plots the decision boundary of the given classifier on training and testing points.
    Colors the training points with true labels, and shows the incorrectly and correctly predicted test points.
    '''
    if y_train is None:
        # X_train is a tupple of X,y.  Unpack it into X,y
        X_train, ytrain = X_train
    if X_test is None:
        # No test data to plot
        X_test = np.zeros((0,np.size(X_train,1)))
        y_test = np.zeros(0)
    elif y_test is None:
        # X_trst is a tupple of X,y.  Unpack it
        X_test, t_test = Xtest
    X, y = np.vstack([X_train, X_test]), np.hstack([y_train.flatten(), y_test.flatten()])
    labels = np.unique(y)
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
    for l in labels:
        l_idxs = np.where(y_train == l)
        ax.scatter(X_train[l_idxs, 0], X_train[l_idxs, 1], label=f'train/{l}', c=cmap_fg[l], marker='.')
    
    # Plot test points
    y_test_predict = clf.predict(X_test)
    for l in labels:
        # Mark the wrong ones
        wrong_idxs = np.where((y_test == l) & (y_test_predict != y_test))
        ax.scatter(X_test[wrong_idxs, 0], X_test[wrong_idxs, 1], label=f'test/should be {l} (wrong)', c=cmap_fg[l], marker='x', s=100)

        # Plot the correct ones
        corr_idxs = np.where((y_test_predict == l) & (y_test_predict == y_test))
        ax.scatter(X_test[corr_idxs, 0], X_test[corr_idxs, 1], label=f'test/predicted {l} (correct)', c=cmap_fg[l], marker='+', s=50)
        
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Decision boundary\nShaded regions show what the label clf would predict for a point there')
    plt.legend(title='label', bbox_to_anchor=(1.04, 1), loc='upper left')


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


def read_sms_spam_data(filepath: str, sep: str = '\t', drop_duplicates: bool = True):
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
    if drop_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)
    data.columns = ['label', 'text']
    
    # Preprocess data
    data = preprocess_data(data)
    
    return data['text'].values, data['label'].values


class SMS_Vectorizer:
    def __init__(self, sentences, d):
        # Count the total occurrences of each word in sentences
        word_freq = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        # Save the d words with the highest frequency
        self.high_freq_words = sorted([(word_freq[word], word) for word in word_freq], reverse=True)[: d]
        self.high_freq_words = [item[1] for item in self.high_freq_words]

    def vectorize(self, sentences):
        # vector[i][j]: how many occurrences of self.high_freq_words[j] we find in sentences[i]
        vectors = np.zeros((len(sentences), len(self.high_freq_words)))
        for i, sentence in enumerate(sentences):
            sentence = sentence.lower().split()
            for j, word in enumerate(self.high_freq_words):
                vectors[i][j] = sentence.count(word)
        return np.array(vectors)