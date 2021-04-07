# MNIST

<h2 align="center">
    <img src="assets/mnist_dataset.png" alt="Example of the MNIST dataset" width="800px" />
</h2>

<br/><br/>

-------

## Content

- [`About the Project`](#about-the-project)
- [`What is Triplet Loss?`](#what-is-triplet-loss-?)
- [`Results`](#results)
- [`Next Steps`](#next-steps)

<br/><br/>

-------

## About the Project
This repositoriy is built to play around with the well-known [`MNIST-dataset`](http://yann.lecun.com/exdb/mnist/) and `Neural Networks` using [`Triplet Loss`](https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973). 

<br/><br/>

-------

## What is Triplet Loss?
`Triplet Loss` can be easily implemented into a 'normal' `Neural Network`, the only difference is made during the training procedure by using a different loss than well-known [`Categorical Cross Entropy Loss`](https://gombru.github.io/2018/05/23/cross_entropy_loss/), [`Hinge Loss`](https://medium.com/analytics-vidhya/understanding-loss-functions-hinge-loss-a0ff112b40a1) or others that are used for classification purposes. In contrast to those `loss functions`, `Triplet Loss` is not directly used for classification. Similar to [`(Fisher's) Linear Discriminant Analysis`](https://sebastianraschka.com/Articles/2014_python_lda.html), `Neural Networks` using `Triplet Loss` are transforming the given data - can be anything from images (Computer Vision) up to any other classification dataset - to find a representation maximizing both, the `in-class-similarity` and the `out-of-class` difference of all samples.

<br/><br/>

-------

## Results

After transforming the given raw dataset, the resulting representation can be used to fit a [`Principal Component Analysis (PCA)`](shorturl.at/lyNZ0).

-------

The following two graphics meanwhile show the immense power of `Neural Networks with Triplet Loss`. In this case, samples from the `MNIST` are transformed from their original shape (of `28x28` pixels) down to only `2` dimensions - that the result can be shown graphically:

<h2 align="center">
    <img src="assets/PCA.png" alt="Data after PCA" width="800px" />
</h2>

Using a `PCA` to transform the data down to `2` dimensions

<h2 align="center">
    <img src="assets/NN_Triplet_Loss_PCA.png" alt="Data after Neural Network with Triplet Loss and PCA" width="800px" />
</h2>

Using a `Neural Network with Triplet Loss` and a `PCA` to transform the data down to `2` dimensions

<br/><br/>

<h2 align="center">
    <img src="assets/PCA_confusion_matrix.png" alt="Confusion Matrix after PCA" width="800px" />
</h2>

Confusion Matrix after  a `PCA` transformed the data down to `2` dimensions

<h2 align="center">
    <img src="assets/NN_Triplet_Loss_PCA_confusion_matrix.png" alt="Confusion Matrix after Neural Network with Triplet Loss and PCA" width="800px" />
</h2>

Confusion Matrix after `Neural Network with Triplet Loss` and `PCA` to transformed the data down to `2` dimensions

-------

Given the result of the first transformation, it seems very hard for a (very simple) classifier - such as `Support Vector Machine (SVM)`, `K-Nearest Neighbors (KNN)`, etc. - to achieve high accuracies. In contrast to that, having the results of the second transformation, it is very likely for such a simple classifier to achieve very high and stable accuracies.

This can also be seen in the two different confusion matrixes, pointing out that the used `SVM` is much more successful to predict the transformed data. All evaluated metrics, namely `accuracy`, `precision`, `recall` and `f1 score` - described [`here`](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9) - are widely higher for the second transformation!

The exact results using the whole training dataset (predicting on the training dataset **as well!**):

|           |   | PCA    | NN with Triplet Loss + PCA |
| --------- | - | :----: | :------------------------: |
| Accuracy  |   | 45.98% | 99.89%                     |
| Precision |   | 0.46   | 1.00                       | 
| Recall    |   | 0.47   | 1.00                       |
| F1 Score  |   | 0.45   | 1.00                       |
|           |   |        |                            |

## Next Steps

- think about `encoding` columns that are still numeric (instead of dropping them during [`utils's`](scripts/utils.py) `check_numeric()`. Possible would be:
    - Binarizer, such as pandas's [`get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
    - sklearn's [`Ordinal Encoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)