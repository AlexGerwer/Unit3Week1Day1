"""
utility functions for working with DataFrames
"""
import pandas


def RowNaNcounts(dfObj):
    """ Count total NaN values in each row of a DataFrame"""
    for i in range(len(dfObj.index)):
        print("NaN in row ", i, " : ",  dfObj.iloc[i].isnull().sum())


def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertoolsimport numpy as np
    import matplotlib.pyplot as pltfrom sklearn
    import svm
    import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 plt.tight_layout())
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def train_val_test_split(data, train_portion, val_portion, test_portion):
    """ Train/validate/test split function for a dataframe """
    # Dependencies
    from sklearn.model_selection import train_test_split

    # Calculate the split
    total_portion = train_portion + val_portion + test_portion
    non_train_portion = val_portion + test_portion

    # round(number[, ndigits])
    train_input = round(train_portion/total_portion, 6)
    non_train_input = round(non_train_portion/total_portion, 6)
    val_input = round(val_portion/non_train_portion, 6)
    test_input = round(test_portion/non_train_portion, 6)

    # Split training and validation data
    train, non_train = train_test_split(data, train_size=train_input,
                                        test_size=non_train_input,
                                        random_state=42)

    # Split the validation and test data
    val, test = train_test_split(non_train, train_size=val_input,
                                 test_size=test_input, random_state=42)

    # Return train, val, test
    return (train, val, test)
