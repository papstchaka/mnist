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
    ## check if data is pandas.DataFrame
    if not type(df) == pd.DataFrame:
        print("data is no pandas.DataFrame, cannot be further processed")
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

def iter_scale_sample(data:np.array, scaler:object = MinMaxScaler) -> np.array:
    '''
    iterates over the whole dataset, scales each sample by given scaler
    Parameters:
        - data: dataset to iterate over [numpy.array]
        - scaler: desired scaler [object, default = MinMaxScaler]
    Returns:
        - scaled_data: scaled dataset [np.array]
    '''
    ## init scaler
    sc = scaler()
    ## iterate over given data, yield scaled data
    for sample in tqdm(data):
        yield sc.fit_transform(sample)
        
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
    colors = list(mcolors.TABLEAU_COLORS)
    for i in range(len(subdata)):
        ax.scatter(subdata[i]["x"], subdata[i]["y"], color = colors[i], label = i)
    ## update layout, without ticklabels and the grid to be on
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ## set title, new legend, save figure, show plot
    ax.set_title(f"Plot of the reduced data after PCA, colored in respective label", size=2*size)
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