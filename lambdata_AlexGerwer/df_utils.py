"""
utility functions for working with DataFrames
"""
import pandas
import numpy as np
import collections as cl
import matplotlib.pyplot as plt

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
    import itertools
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
        plt.text(j, i, format(cm[i, j], fmt), plt.tight_layout(),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black"
                 )
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


class  Stats(object):
    """
    General representation of statistical utilities
    """

    def __init__(self,data):
        """Captures the data"""
        self.data = list(data)
        print ('The data = ', self.data)

    def stat_mean(self):
        """Calculates the mean"""
        self.mean_value=sum(self.data)/len(self.data)
        return self.mean_value

    def stat_median(self):
        """Calculates the median:"""
        self.data.sort()
        if len(self.data) % 2 != 0:
            idx = int((len(self.data) - 1) / 2)
            return self.data[idx]
        else:
            idx_1 = self.data[int((len(self.data) / 2))]
            idx_2 = self.data[int((len(self.data) / 2) - 1)]
            idx = (idx_1 + idx_2) / 2
            return self.data[idx]
        self.median_value=self.data[idx]
        return self.median_value

    def stat_mode(self):
        """Calculates the mode"""
        self_data_list = list(self.data)
        counter = cl.Counter(self_data_list)
        if len(counter) > 1:  # ensure at least 2 unique elements
            possible_mode, next_highest = counter.most_common(2)
            if possible_mode[1] > next_highest[1]:
                self.mode_value = possible_mode[0]
                return self.mode_value
        self.mode_value = np.nan
        return self.mode_value

    def stat_variance(self):
        """Calcalates the variance"""
        self_data_array = np.array(self.data)
        self.variance_value = ((self_data_array - self.stat_mean())**2).sum() / (len(self_data_array) - 1)
        return self.variance_value

    def stat_standard_deviation(self):
        """Calculates the standard deviation"""
        self_data_array = np.array(self.data)
        variance_value = ((self_data_array - self.stat_mean())**2).sum() / (len(self_data_array) - 1)
        self.standard_deviation_value = variance_value**(0.5)
        return self.standard_deviation_value

    def stat_coefficient_of_variation(self):
        """Calculates the coefficient of variation"""
        self.coefficient_of_variation_value = self.stat_standard_deviation() / self.stat_mean()
        return self.coefficient_of_variation_value

    def summary(self):
        return {"Mean": self.stat_mean(), "Median": self.stat_median(),
                "Mode": self.stat_mode(), "Variance": self.stat_variance(),
                "StandardDeviation": self.stat_standard_deviation(),
                "CoefficientOfVariation": self.stat_coefficient_of_variation()}
