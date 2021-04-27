'''
Script implements several preprocessing and evaluation steps that have to be done for the triplet loss network
'''

## Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter

def tackle_distribution_shift(data:pd.DataFrame, approach:str = "reuse") -> pd.DataFrame:
    '''
    function to balance the appearance of samples from different classes -> tackle distribution shift
    Parameters:
        - data: data with distribution shift [pandas.DataFrame]
        - approach: strategy to tackle the distribution shift. Possible values are [String]
            - reusing minor class samples --> 'reuse' = default
            - mixing both approaches by using the size of the median common class --> "mix"
            - constraining the number of major class samples --> 'constrain'
    Returns:
        - df: data without distribution shift [pandas.DataFrame]
    '''
    ## make sure not to overwrite given data
    df = data.copy()
    ## get all labels that exist
    labels = data.loc[:, "Label"]
    ## get appearances of each label
    counted_labels = Counter(labels).most_common()
    ## get max num of samples (valid for all classes)
    if approach == "reuse":
        ## take appearance value of most common label
        sample_size = counted_labels[0][1]
    elif approach == "mix":
        sample_size = counted_labels[int(counted_labels.__len__()*0.5)][1]
    elif approach == "constrain":
        ## take appearance value of least common label
        sample_size = counted_labels[-1][1]
    else:
        print("approach not implemented (yet)! Using 'resue' instead!")
        ## take appearance value of most common label
        sample_size = counted_labels[0][1]
    ## take a 'subset' or 'superset' of every class
    sampled_data = [df[df.Label == label].sample(n = sample_size, replace = True) for label in np.unique(labels)]
    ## return merged data
    return pd.concat(sampled_data).reset_index(drop = True)
    
def encode_objects(data:pd.DataFrame, ignore_columns:list = [], how:str = "binarizer") -> pd.DataFrame:
    '''
    goes through given dataset, encodes all object columns into numerical data
    Parameters:
        - data: DataFrame to anaylse [pandas.DataFrame]
        - ignore_columns: List of columns that shall be ignored [List, default = []]
        - how: strategy to encode. The following are possible [String]
            - Binarize: every unique value gets own column, filled with 0's and 1's, using pandas.get_dummies() --> 'binarizer' = Default
            - OrdinalEncoder: unique values get replaced by increasing number (same amount of features) using sklearn's OrdinalEncoder --> 'ordinal'
    Returns:
        - encoded_data: encoded DataFrame [pandas.DataFrame]
    '''
    ## make sure not to overwrite given data
    df = data.copy()
    ## check if data is pandas.DataFrame
    if not type(df) == pd.DataFrame:
        if type(df) == pd.Series:
            df = pd.DataFrame(data)
            df.columns = ["Series_Data"]
        else:
            print("data is no pandas.DataFrame, cannot be further processed")
            return df
    ## remaining columns that shall be checked (all - ignore_columns)
    columns = list(set(df.columns) - set(ignore_columns))
    ## define possible strategies
    if how == "binarizer":
        strategy = lambda x: pd.get_dummies(x)
    elif how == "ordinal":
        enc = OrdinalEncoder()
        strategy = lambda x: pd.DataFrame(enc.fit_transform(x), columns = x.columns)
    else:
        print("strategy not implemented (yet!). Using pandas.get_dummies() instead!")
        strategy = lambda x: pd.get_dummies(x)
    cols = []
    ## go through all remaining columns, check if 'object' features exist
    for column in columns:
        if pd.api.types.is_string_dtype(df[column]):
            cols.append(column)
    ## get all other columns from data
    other_columns = list(set(df.columns) - set(cols))
    ## get both subdatasets - the encoded one and the remaining original one
    encoded_data_raw = strategy(df[cols])
    data_raw = df[other_columns]
    ## merge both subdatasets
    encoded_data = pd.concat([encoded_data_raw, data_raw], axis = 1)
    return encoded_data

def check_nan(data:pd.Series) -> bool:
    '''
    checks whether given data contains NaN's
    Parameters:
        - data: data to check [pandas.Series], can also be pandas.DataFrame
    Returns:
        - nan's: True, if data contains NaN's, otherwise False [Boolean]
    '''
    ## make sure not to overwrite given data
    df = data.copy()
    if (not type(df) == pd.DataFrame) and (not type(df) == pd.Series):
        print("data is no pandas.DataFrame, no check for NaN's done")
        return False
    if type(df) == pd.DataFrame:
        return data.isna().sum().sum().astype(bool)
    return data.isna().sum().astype(bool)

def add_nan(data:pd.DataFrame, amount:float = 0.05) -> pd.DataFrame:
    '''
    taking the given DataFrame and randomly adds the given amount of NaN's into it
    Parameters:
        - data: given data to add NaN's to [pandas.DataFrame]
        - amount: desired amount of NaN's [Float, default = 0.05]
    Returns:
        - nan_data: data containing desired amount of NaN's [pandas.DataFrame]
    '''
    ## set a numpy array with <amount> number of `True`s in the shape of data
    nan_array = np.random.random(data.shape) < amount
    ## mask every element in 'data' with an NaN, when that element in 'nan_array' is set to True
    nan_data = data.mask(nan_array)
    return nan_data

def check_numeric(data:pd.DataFrame, ignore_columns:list = []) -> pd.DataFrame:
    '''
    function that converts all columns in DataFrame into numeric ones. Deletes all columns where `pandas.to_numeric()` fails (as they seem to be Strings)
    Parameters:
        - data: DataFrame with all different kinds of column types [pandas.DataFrame]
        - ignore_columns: List of columns that shall be ignored [List, default = []]
    Returns:
        - df: DataFrame with converted columns and without deleted columns [pandas.DataFrame]
    '''
    ## make sure not to overwrite given data
    df = data.copy()
    ## check if data is pandas.DataFrame
    if not type(df) == pd.DataFrame:
        if type(df) == pd.Series:
            num_df = pd.to_numeric(df, errors = "coerce")
            if num_df.isna().sum() > 0:
                print("data cannot be converted to numerical data, you have to encode it")
                return df
            return num_df
        print("data is no pandas.DataFrame, cannot be further processed")
        return df
    ## remaining columns that shall be checked (all - ignore_columns)
    columns = list(set(df.columns) - set(ignore_columns))
    ## iterate over all columns, convert them to numeric ones (float or int)
    for col in tqdm(columns, desc="make all columns numerical"):
        ## if error, then fill with NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ## drop all columns that contain NaN's
    df = df.dropna(axis=1)
    return df

def iter_columns(data:pd.DataFrame, columns:list, trait:str) -> str:
    '''
    iterator, going over all columns in DataFrame, checking their content for desired trait
    Parameters:
        - data: DataFrame to iterate over [pandas.DataFrame]
        - columns: columns to check [List]
        - trait: what shall the column be checked for. Possible values [String]
            - Unique: check for unique values per column, returns columns consisting of the same value over all samples --> 'unique'
    Returns:
        - col: column that contains only one different value [String]
    '''
    ## iterate over all given columns
    for col in tqdm(columns, desc=f"handle {trait}'s'"):
        ## check for Unique's
        if trait == "unique":
            ## check if column contains more than one different value
            if data[col].unique().__len__() == 1:
                ## if yes, return that column
                yield col
                
def handle_nans(data:pd.DataFrame, strategy:str = "null", ignore_columns:list = []) -> pd.DataFrame:
    '''
    function that drops all columns (=features) that only contain one different value
    Parameters:
        - data: DataFrame [pandas.DataFrame]
        - strategy: strategy to fill in the dataset. Possible values are [String]
            - 0: fill all with Zero --> 'null' = default
            - Mean: fill all with mean of respective feature --> 'mean'
            - Median: fill all with median of respective feature --> 'median'
            - Max: fill all with max of respective feature --> 'max'
            - Min: fill all with min of respective feature --> 'min'
        - ignore_columns: List of columns that shall be ignored [List, default = []]
    Returns:
        - df: DataFrame without deleted columns [pandas.DataFrame]
    '''
    ## make sure not to overwrite given data
    df = data.copy()
    ## check if data is data contains NaN's
    if not check_nan(df):
        print("no NaN's inside data")
        return df
    ## remaining columns that shall be checked (all - ignore_columns)
    columns = list(set(df.columns) - set(ignore_columns))
    ## init columns to drop
    cols = []
    ## check strategy, calculate filling value(s)
    if strategy == "null":
        value = [0 for _ in range(columns.__len__())]
    elif strategy == "mean":
        value = df[columns].mean()
    elif strategy == "median":
        value = df[columns].median()
    elif strategy == "min":
        value = df[columns].min()
    elif strategy == "max":
        value = df[columns].max()
    else:
        print("strategy not implemented (yet). Filling with 0")
        value = [0 for _ in range(columns.__len__())]
    df = df.fillna(dict(zip(columns, value)))
    ## drop columns that ONLY contain NaN's, no matter what 'ignore_columns' says
    df = df.dropna(how = "all", axis = 1)
    return df

def handle_uniques(data:pd.DataFrame, ignore_columns:list = []) -> pd.DataFrame:
    '''
    function that handles all columns (=features) that only contain one different value by dropping them --> they do not contain helpful (any) information
    Parameters:
        - data: DataFrame [pandas.DataFrame]
        - ignore_columns: List of columns that shall be ignored [List, default = []]
    Returns:
        - df: DataFrame without deleted columns [pandas.DataFrame]
    '''
    ## make sure not to overwrite given data
    df = data.copy()
    ## check if data is pandas.DataFrame
    if not type(df) == pd.DataFrame:
        print("data is no pandas.DataFrame, cannot be further processed")
        return df
    ## remaining columns that shall be checked (all - ignore_columns)
    columns = list(set(df.columns) - set(ignore_columns))
    ## init columns to drop
    cols = []
    ## make sure not to overwrite given data
    df = data.copy()
    for col in iter_columns(df, columns, "unique"):
        cols.append(col)
    df = df.drop(cols, axis=1)
    return df

def drop_features(data:pd.DataFrame, columns:list = []) -> pd.DataFrame:
    '''
    function that drops all columns are given by `columns`
    Parameters:
        - data: DataFrame with time columns [pandas.DataFrame]
        - columns: List of columns that shall be deleted [List, default = []]
    Returns:
        - df: DataFrame without deleted columns [pandas.DataFrame]
    '''
    ## make sure not to overwrite given data or the given columns
    cols = columns.copy()
    df = data.copy()
    ## check if data is pandas.DataFrame
    if not type(df) == pd.DataFrame:
        print("data is no pandas.DataFrame, cannot be further processed")
        return df
    df = df.drop(cols, axis=1)
    return df

def flatten(X:np.array) -> np.array:
    '''
    flattens a 3D array into 2D array    
    Parameters:
        - X: a 3D array with shape samples x width x height [numpy.array]
    Returns:
        - flattened_X: 2D array with shape sample x width*height [numpy.array]
    '''
    flattened_X = X.reshape(X.shape[0], -1)
    return flattened_X

def iter_scale(X:np.array, scaler:object) -> np.array:
    '''
    iterates over fiven X, scales given 3D array using given (trained) scaler
    Parameters:
        - X: 3D array with shape samples x length x features [numpy.array]
        - scaler: scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize [object]
    Returns:
        - scaled_X: scaled 3D array of same shape [numpy.array]
    '''
    ## copy X to make sure not to overwrite the original data
    scaled_X = X.copy()
    for X_to_scale in tqdm(scaled_X):
        yield scaler.transform(X_to_scale.reshape(1, -1)).reshape(X_to_scale.shape)   
        
def pca(X:np.array, y:np.array) -> (np.array, np.array):
    '''
    plots a scatter showing the transformed dataset (if it is 2D) with different coloring for the different classes
    Parameters:
        - X: Array containing the original x values [numpy.array]
        - y: Array containing the labels [numpy.array]
    Returns:
        - X_transformed: Array containing the transformed x values [numpy.array]
        - y: Array containing the labels [numpy.array]
    '''
    ## init pca with two components
    pca = PCA(n_components = 2)
    ## copy data to be sure not to accidentally overwrite something
    pca_x = X.copy()
    ## check whether data has more than two dimensions (example shape of [60, 28, 28])
    if pca_x.shape.__len__() > 2:
        print("Dimension too high, X gets reshaped")
        ## if yes, reshape (in this case [60, 784])
        pca_x = X.copy().reshape(X.shape[0], -1)
    ## fit PCA, transform data
    X_transformed = pca.fit_transform(pca_x, y)
    return X_transformed, y

def plot_reduced_data(X_new:np.array, y:np.array) -> None:
    '''
    plots a scatter showing the transformed dataset (if it is <= 2D) with different coloring for the different classes
    Parameters:
        - X_new: Array containing the transformed x values [numpy.array]
        - y: Array containing the labels [numpy.array]
    Returns:
        - None
    '''
    ## make DataFrame from transformed x values, add information about labels, rename the columns
    reduced_data = pd.DataFrame(X_new)
    if reduced_data.columns.__len__() == 1:
        reduced_data["y"] = 0
    reduced_data["Label"] = y
    reduced_data.columns = ["x","y","Label"]
    ## make a list of SubDataSets that each only contain data about respective label
    subdata = [reduced_data[reduced_data["Label"] == label] for label in np.unique(y)]
    ## set size, init figure
    size = 10
    fig=plt.figure(figsize=(2*size,size))
    ## add plots
    ax = fig.add_subplot()
    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.BASE_COLORS)
    for i in range(len(subdata)):
        ax.scatter(subdata[i]["x"], subdata[i]["y"], color = colors[i], label = i)
    ## update layout, without ticklabels and the grid to be on
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ## set title, new legend, save figure, show plot
    ax.set_title(f"Plot of the reduced data after PCA, colored in respective label", size=2*size)
    if subdata.__len__() > 12:
        ax.legend().set_visible(False)
    else:
        ax.legend(prop={'size': 1.5*size})
    plt.show()
    
def evaluate_model(clf:object, X:np.array, y:np.array) -> None:
    '''
    evaluates the given model with given data, prints different metrices [accuracy, precision, recall, f1 score]
    Parameters:
        - clf: model to evaluate [object]
        - X: x values [np.array]
        - y: labels [np.array]
    '''
    ## split data to train and test samples, fit classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)
    ## init plot
    size = 10
    fig=plt.figure(figsize=(2*size,size))
    ax=fig.add_subplot()
    ## plot confusion matrix
    plot_confusion_matrix(clf, X_test, y_test, normalize="true", ax = ax, cmap=plt.cm.Blues)
    ## predict test samples
    y_pred = clf.predict(X_test)
    ## calculate the metrics
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average="weighted")
    recall = recall_score(y_test, y_pred,average="weighted")
    f1 = f1_score(y_test, y_pred,average="weighted")
    ## print results
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")