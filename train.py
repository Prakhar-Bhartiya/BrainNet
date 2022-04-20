import numpy as np
import sklearn as sk
import tensorflow as tf
import h5py
import pandas as pd
from pymatreader import read_mat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class model:

  def calcPCA(data):
    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(data)
    eVal = pca.explained_variance_
    PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20'])
    return PCA_df.to_numpy()

  def logReg(X_train, X_test, y_train, y_test):
    logReg = LogisticRegression().fit(X_train,y_train)
    logTest = logReg.score(X_test, y_test)
    return logTest

  def kMeans(X_train, X_test, y_train, y_test):
    kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, max_iter=300)
    kmeans.fit(X_train)
    labels1 = kmeans.labels_
    kmeansTest = kmeans.predict(X_test)
    accuracy = accuracy_score(y_test, kmeansTest)
    return accuracy

  def SVM(X_train, X_test, y_train, y_test):
    svm = LinearSVC(max_iter=30000).fit(X_train,y_train)
    svmTest = svm.score(X_test, y_test)
    return svmTest

# read data
data = read_mat('Dataset1.mat')

arr = data['Raw_Data']
s0 = arr[:, 0, :]
s1 = arr[:, 1, :]
s2 = arr[:, 2, :]

realData = np.concatenate((s0, s1, s2))

data2 = read_mat('sampleAttack.mat')
arrTotal = data2['attackVectors']

arr0 = arrTotal[0, :, :, :]
arr1 = arrTotal[1, :, :, :]
arr2 = arrTotal[2, :, :, :]
arr3 = arrTotal[3, :, :, :]
arr4 = arrTotal[4, :, :, :]
arr5 = arrTotal[5, :, :, :]

a0s0 = arr0[:, 0, :]
a0s1 = arr0[:, 1, :]
a0s2 = arr0[:, 2, :]
a1s0 = arr1[:, 0, :]
a1s1 = arr1[:, 1, :]
a1s2 = arr1[:, 2, :]
a2s0 = arr2[:, 0, :]
a2s1 = arr2[:, 1, :]
a2s2 = arr2[:, 2, :]
a3s0 = arr3[:, 0, :]
a3s1 = arr3[:, 1, :]
a3s2 = arr3[:, 2, :]
a4s0 = arr4[:, 0, :]
a4s1 = arr4[:, 1, :]
a4s2 = arr4[:, 2, :]
a5s0 = arr5[:, 0, :]
a5s1 = arr5[:, 1, :]
a5s2 = arr5[:, 2, :]

fakeData = np.concatenate((a0s0, a0s1, a0s2, a1s0, a1s1, a1s2, a2s0, a2s1, a2s2, a3s0, a3s1, a3s2, a4s0, a4s1, a4s2, a5s0, a5s1, a5s2))

# Create total data matrix and labels
features = np.concatenate((realData[:, :4800], fakeData))
features = StandardScaler().fit_transform(features)
labels = np.concatenate((np.ones(realData.shape[0]), np.zeros(fakeData.shape[0])))
labels = np.reshape(labels, (labels.shape[0], 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

# PCA on training and testdata
PCA_train = model.calcPCA(X_train)
PCA_test = model.calcPCA(X_test)

# Log Reg
print("Log Reg: ", model.logReg(PCA_train, PCA_test, y_train, y_test)) #PCA

#K-Means
print("KMeans: ", model.kMeans(PCA_train, PCA_test, y_train, y_test)) #PCA

# SVM
print("SVM: ", model.SVM(PCA_train, PCA_test, y_train, y_test)) #PCA

    

