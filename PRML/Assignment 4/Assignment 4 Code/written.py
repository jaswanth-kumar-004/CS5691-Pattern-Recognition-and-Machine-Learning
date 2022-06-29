#Import libraries
import os
import math
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import resample
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

#Reading training data for Handwritten telugu characters
print("Reading Handwritten data")
path = 'Written/train'
flag = 0
y_train = []

#Looping over the train folder to access all files in it
for file in os.listdir(path):
    if file.endswith(".txt"):
        
        #Opening the file & reading data
        f = open(f"{path}/{file}", "r")
        A = f.readline().split()
        
        #Appending classes of training data set to y_train
        if(file[0:1] == 'c'):
            y_train.append(file[0:3])
        else:
            y_train.append(file[0:2])
        
        #Converting each training sample into matrix of floating point numbers
        x = []; y = []
        i = 1
        while i < len(A):
            x.append(float(A[i]))
            i += 1
            y.append(float(A[i]))
            i += 1
        t = np.column_stack((x,y))
        
        #Extracting slope
        np.seterr(divide='ignore')
        d1 = np.diff(t[:,0])/np.diff(t[:,1])
        
        #Convert infinite slope values to 10^7
        for i in range(d1.shape[0]):
            if math.isinf(d1[i]):
                d1[i] = 10000000
        
        #Combining slope with coordinates data
        t = np.column_stack((t[0:-1],d1))
        
        #Concatenating & equating all data sets length to 300 by resampling
        t = resample(t.flatten(),300)
        
        #Combining all tranining samples in global variable train
        if flag == 0:
            globals()['x_train'] = t
        else:
            globals()['x_train'] = np.row_stack((x_train,t))
        flag = 1
        
x_train = x_train/x_train.max()  #Normalizing training data set

#Reading test data for Handwritten telugu characters
path = 'Written/dev'

#Numbering each test data set for different classes
nai = nbA = nchA = nlA = ntA = 1

y_test = []
truth = []
flag = 0

#Looping over the dev folder to access all files in it
for file in os.listdir(path):
    if file.endswith(".txt"):

        #Opening the file & reading data
        f = open(f"{path}/{file}", "r")
        A = f.readline().split()
        
        #Creating list of class labels for roc & det curves
        if file[0:1] == 'a':
            truth.append(0)
        elif file[0:1] == 'b':
            truth.append(1)
        elif file[0:1] == 'c':
            truth.append(2)
        elif file[0:1] == 'l':
            truth.append(3)
        elif file[0:1] == 't':
            truth.append(4)
            
        #Appending classes of test data set to y_test
        if file[0:1] == 'c':
            y_test.append(file[0:3])
        else:
            y_test.append(file[0:2])
            
        #Converting each test sample into matrix of floating point numbers
        x = []; y = []
        i = 1
        while i < len(A):
            x.append(float(A[i]))
            i += 1
            y.append(float(A[i]))
            i += 1
        t = np.column_stack((x,y))
        
        #Extracting slope
        np.seterr(divide='ignore')
        d1 = np.diff(t[:,0])/np.diff(t[:,1])
        
        #Convert infinite slope values to 10^7
        for i in range(d1.shape[0]):
            if math.isinf(d1[i]):
                d1[i] = 10000000
        
        #Combining slope with coordinates data
        t = np.column_stack((t[0:-1],d1))
        #Concatenating & equating all data sets length by resampling
        t = resample(t.flatten(),300)
        
        #Combining all test samples in global variable x_test
        if flag == 0:
            globals()['x_test'] = t
        else:
            globals()['x_test'] = np.row_stack((x_test,t))
        flag = 1
        
x_test = x_test/x_test.max()     #Normalizing test data set

#Function for Principal Component Analysis
def PCA_components(x , n):
     
    #Mean Centering the data  
    x_meaned = x - np.mean(x , axis = 0)
     
    #Calculating the covariance matrix of the mean-centered data
    cov_mat = np.cov(x_meaned , rowvar = False)

    #Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eig_values , eig_vectors = np.linalg.eig(cov_mat)
    eig_vectors = eig_vectors.T
    
    #Sort the eigenvectors in descending order of corresponding eigen values
    idx = np.argsort(abs(eig_values))[::-1]
    sorted_vectors = eig_vectors[idx]
     
    #Return the first n eigenvectors
    return sorted_vectors[0:n].T

#Dimension reduction using PCA
print("Dimensionality reduction using PCA")
x_train_pca = np.dot(x_train - np.mean(x_train , axis = 0), PCA_components(x_train, 30))
x_test_pca = np.dot(x_test - np.mean(x_test , axis = 0), PCA_components(x_train, 30))

#Function for Linear Discriminant Analysis
def LDA_components(x, y, n):

    features = x.shape[1]
    classes = np.unique(y)
    
    #Caclulating overall mean
    m = np.mean(x, axis=0)
    
    #Initializing between class & within class scatter matrix
    Sw = np.zeros((features, features))
    Sb = np.zeros((features, features))
    
    #Calculating between class & within class scatter matrix
    for c in classes:
        #Accumulaing feature vectors which belong to class c
        x_c = np.empty((1,features))
        for idx, a in enumerate(y):
            if c == a:
                x_c = np.row_stack((x_c, x[idx]))
        x_c = np.delete(x_c, (0), axis=0)

        mi = np.mean(x_c, axis=0)
        Sw += np.dot((x_c - mi).T, (x_c - mi))

        n_c = x_c.shape[0]
        Sb += n_c * np.dot((mi - m), (mi - m).T)

    #Caculating Sw^-1 * Sb
    A = np.dot(np.linalg.inv(Sw), Sb)

    #Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eig_values , eig_vectors = np.linalg.eig(A)
    eig_vectors = eig_vectors.T
    
    #Sort the eigenvectors in descending order of corresponding eigen values
    idx = np.argsort(abs(eig_values))[::-1]
    sorted_vectors = eig_vectors[idx]
     
    #Return the first n eigenvectors
    return sorted_vectors[0:n].T

#Dimension reduction using LDA
print("Dimensionality reduction using LDA")
x_train_lda = np.dot(x_train, LDA_components(x_train, y_train, 30).real)
x_test_lda = np.dot(x_test, LDA_components(x_train, y_train, 30).real)

#Function for confusion Matrix
def confusion(y_true, y_pred, model):
    val= ['ai', 'bA', 'chA', 'lA','tA']  #Class labels
    cm = confusion_matrix(y_true, y_pred)
    
    #Formatting the confusion matrix
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(len(val), len(val))
    
    #Colouring the matrix
    fig, ax = plt.subplots(figsize=(8,7))
    ax = sns.heatmap(cm/np.sum(cm), annot=labels, fmt='',annot_kws={"size": 13})
    
    ax.xaxis.set_ticklabels(val) 
    ax.yaxis.set_ticklabels(val)
    
    #Labelling the confusion matrix
    ax.set_title('Confusion matrix by %s'% model+" for Handwritten data")    
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values ');

    plt.show()
    return cm

#Accuracy of prediction using comfusion matrix
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

def find_score_and_plot_confusion(x_train, y_train, x_test, model, dim_red = ""):
    #Calling classifiers accordingly
    if model[0:3] == "KNN":
        y_pred, score = apply_knn(x_train, y_train, x_test)
    elif model[0:19] == "Logistic Regression":
        y_pred, score = apply_logreg(x_train, y_train, x_test)
    elif model[0:3] == "SVM":
        y_pred, score = apply_svm(x_train, y_train, x_test)
    elif model[0:3] == "ANN":
        y_pred, score = apply_ann(x_train, y_train, x_test)
    
    #Plotting confusion matrix
    cm = confusion(y_test, y_pred,model+dim_red)
    
    #Computing accuracy & misclassification rate
    print("Accuracy = %.2f%%" % (accuracy(cm)*100))
    print("Misclassification rate = %.2f%%\n" % (100 - accuracy(cm)*100))
    
    #Returing computed score
    return score

#KNN Classifier
class KNN_function:
    def __init__(self, k):
        self.k = k
        
    #Funciton to fit training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    #Computing Eucledian distance
    def Eucledian_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    #Class prediction for test data set
    def predict(self, X, classes = 5):
        self.s = np.zeros((X.shape[0], classes))  #Initializing score matrix
        self.i = 0  #Row count in score matrix
        
        #Calling predictor function for each sample of test data set
        y_pred = [self.predict_function(x) for x in X]
        return y_pred, self.s

    def predict_function(self, x):
        #Compute distances between x and all the training sets
        distances = [self.Eucledian_distance(x, x_train) for x_train in self.X_train]
        
        #Sort by distance and return indices of the first k neighbors
        idx = np.argsort(distances)[: self.k]
        
        #Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in idx]
        
        #Calling likelihood function
        self.prob(k_neighbor_labels)
        
        #Return the most common class label from top k
        return Counter(k_neighbor_labels).most_common(1)[0][0]
    
    #Computing likelihood for every ith sample
    def prob(self, neighbour):
        t = Counter(neighbour)
        self.s[self.i, 0] = t['ai']
        self.s[self.i, 1] = t['bA']
        self.s[self.i, 2] = t['chA']
        self.s[self.i, 3] = t['lA']
        self.s[self.i, 4] = t['tA']
        self.i += 1  #Increment column number
        
#Function to run KNN classifier
k = 5
def apply_knn(x_train, y_train, x_test):
    knn = KNN_function(k)
    knn.fit(x_train, y_train)  #Fitting training data
    y_pred, score = knn.predict(x_test) #Prediction for test set
    return y_pred, score

#Computing scores & plotting confusion matrix for KNN classifier
print("Applying KNN for Handwritten data")
score_knn = find_score_and_plot_confusion(x_train, y_train, x_test, "KNN")

#Computing scores & plotting confusion matrix after dimensionality reduction for KNN classifier
print("Applying KNN using PCA for Handwritten data")
score_knn_pca = find_score_and_plot_confusion(x_train_pca, y_train, x_test_pca, "KNN", " using PCA")
print("Applying KNN using LDA for Handwritten data")
score_knn_lda = find_score_and_plot_confusion(x_train_lda, y_train, x_test_lda, "KNN", " using LDA")

#Logistic Regression classifier
class LogReg:
    
    def __init__(self, iterations = 1000, threshold=0.0001, learning_rate = 0.001):
        self.iter = iterations
        self.thres = threshold
        self.lr = learning_rate
    
    def fit(self, X, y, batch_size=100): 
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        
        #Adding bias
        X = np.insert(X, 0, 1, axis=1)
        y = np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
        self.loss = []
        
        #Initializing weights
        self.weights = np.zeros((len(self.classes),X.shape[1]))
        
        #Looping for given number of iterations
        i = 0
        while (not self.iter or i < self.iter):
            #Computing likelihood
            probs = self.proability(X)
            
            #Appending cross entropy
            self.loss.append(-1 * np.mean(y * np.log(probs)))
            
            #Randomly choosing samples for given batch_size
            index = np.random.choice(X.shape[0], batch_size)
            X_batch = X[index]
            y_batch = y[index]
            
            #Computing error
            error = y_batch - self.proability(X_batch)
            
            #Updating parameters
            update = (self.lr * np.dot(error.T, X_batch))
            self.weights += update
            
            #Breaking loop when update reaches given threshold value
            if np.abs(update).max() < self.thres: 
                break
            
            #Incrementing iteration
            i +=1

    #Likelihood function
    def proability(self, x):
        vals = np.dot(x, self.weights.T).reshape(-1,len(self.classes))
        return self.Softmax(vals)

    #Using softmax funciton for multiclass classification
    def Softmax(self, x):
        s = np.sum(np.exp(x), axis=1)  #Computing denominator
        return np.exp(x)/s.reshape(-1,1)
    
    #Predicting class of test data set x
    def predict(self, x):
        probs = self.proability(np.insert(x, 0, 1, axis=1))
        y_pred = np.vectorize(lambda c: self.classes[c])(np.argmax(probs, axis=1))
        return y_pred.tolist(), probs
    
#Function to run Logisitic Regression classifier
def apply_logreg(x_train, y_train, x_test):
    lrm = LogReg()
    lrm.fit(x_train, y_train)  #Fitting training data
    y_pred, score = lrm.predict(x_test) #Prediction for test set
    
    #Multiplying computed proabability with prior
    score[:,0:1] *= (y_test.count("ai")/len(y_test))
    score[:,1:2] *= (y_test.count("bA")/len(y_test))
    score[:,2:3] *= (y_test.count("chA")/len(y_test))
    score[:,3:4] *= (y_test.count("lA")/len(y_test))
    score[:,4:5] *= (y_test.count("tA")/len(y_test))
    
    return y_pred, score

#Computing scores & plotting confusion matrix for Logistic Regression classifier
print("Applying Logistic Regression for Handwritten data")
score_logreg = find_score_and_plot_confusion(x_train, y_train, x_test, "Logistic Regression")

#Computing scores & plotting confusion matrix after dimensionality reduction for Logistic Regression classifier
print("Applying Logistic Regression using PCA for Handwritten data")
score_logreg_pca = find_score_and_plot_confusion(x_train_pca, y_train, x_test_pca, "Logistic Regression", " using PCA")
print("Applying Logistic Regression using LDA for Handwritten data")
score_logreg_lda = find_score_and_plot_confusion(x_train_lda, y_train, x_test_lda, "Logistic Regression", " using LDA")

#SVM Classifier
def apply_svm(x_train, y_train, x_test):
    svm = SVC(probability=True, kernel = "linear", random_state = 0)
    svm.fit(x_train, y_train)
    return svm.predict(x_test).tolist(), svm.predict_proba(x_test)

#Computing scores & plotting confusion matrix for SVM classifier
print("Applying SVM for Handwritten data")
score_svm = find_score_and_plot_confusion(x_train, y_train, x_test, "SVM")

#Computing scores & plotting confusion matrix after dimensionality reduction for SVM classifier
print("Applying SVM using PCA for Handwritten data")
score_svm_pca = find_score_and_plot_confusion(x_train_pca, y_train, x_test_pca, "SVM", " using PCA")
print("Applying SVM using LDA for Handwritten data")
score_svm_lda = find_score_and_plot_confusion(x_train_lda, y_train, x_test_lda, "SVM", " using LDA")

#ANN Classifier
def apply_ann(x_train, y_train, x_test):
    ann = MLPClassifier(activation="logistic", max_iter=500, random_state = 0)
    ann.fit(x_train, y_train)
    return ann.predict(x_test).tolist(), ann.predict_proba(x_test)

#Computing scores & plotting confusion matrix for ANN classifier
print("Applying ANN for Handwritten data")
score_ann = find_score_and_plot_confusion(x_train, y_train, x_test, "ANN")

#Computing scores & plotting confusion matrix after dimensionality reduction for ANN classifier
print("Applying ANN using PCA for Handwritten data")
score_ann_pca = find_score_and_plot_confusion(x_train_pca, y_train, x_test_pca, "ANN", " using PCA")
print("Applying ANN using LDA for Handwritten data")
score_ann_lda = find_score_and_plot_confusion(x_train_lda, y_train, x_test_lda, "ANN", " using LDA")

def roc_curve(s, knn_k = None):
    #Declaring True positive rate & False positive rate
    TPR = []
    FPR = []
    if knn_k != None:
        s = s/knn_k
    #Looping over thresholds
    for threshold in np.arange(s.min(), s.max(), 0.001):
        TP = TN = FP = FN = 0

        #Looping over test data sets
        for i in range(100):
            groundthruth = truth[i]
            
            #Looping over classes
            for j in range(5):
                if s[i,j] >= threshold:                   
                    if groundthruth == j:
                        TP = TP + 1
                    else:
                        FP = FP + 1;
                else:
                    if groundthruth == j:
                        FN = FN + 1
                    else:
                        TN = TN + 1;
                        
        #Adding elements to tpr & fpr list for different threshold values
        TPR = np.append(TPR, TP/(TP + FN))
        FPR = np.append(FPR, FP/(FP + TN))
    
    return (TPR, FPR)

def plot_roc(T1, F1, T2, F2, T3, F3, T4, F4, dim_red = ""):
    #Creating subplot
    plt.subplots(figsize=(6,6))
    
    plt.plot(F1,T1)
    plt.plot(F2,T2)
    plt.plot(F3,T3)
    plt.plot(F4,T4)
    
    #Labelling ROC curve
    plt.legend(["KNN", "Logisitc Regression", "SVM","ANN"])
    plt.grid()
    plt.title("ROC Curve %s"% dim_red)
    plt.xlabel(xlabel='False Positive Rate (FPR)')
    plt.ylabel(ylabel='True Positive Rate (TPR)')

    plt.show()
    
def plot_det(T1, F1, T2, F2, T3, F3, T4, F4, dim_red = ""):    
    #Creating subplot
    plt.subplots(figsize=(6,6))
    
    #Computing Missed Detection Rate & False Alarm Rate using norm.ppf function
    plt.plot(norm.ppf(F1), norm.ppf(1-T1))
    plt.plot(norm.ppf(F2), norm.ppf(1-T2))
    plt.plot(norm.ppf(F3), norm.ppf(1-T3))
    plt.plot(norm.ppf(F4), norm.ppf(1-T4))
    
    #Labelling DET curve
    plt.legend(["KNN", "Logisitc Regression", "SVM","ANN"])
    plt.grid()
    plt.title("DET Curve %s"% dim_red)
    plt.xlabel(xlabel='False Alarm Rate')
    plt.ylabel(ylabel='Missed Detection Rate')

    plt.show()
    
def plot_all_classifiers_together(score_knn, score_logreg, score_svm, score_ann, dim_red = ""):
    #Computing True positive rate & False positive rate for different classifiers
    T1,F1 = roc_curve(score_knn, k)
    T2,F2 = roc_curve(score_logreg)
    T3,F3 = roc_curve(score_svm)
    T4,F4 = roc_curve(score_ann)
    
    #Plotting ROC & DET curves
    plot_roc(T1, F1, T2, F2, T3, F3, T4, F4, dim_red)
    plot_det(T1, F1, T2, F2, T3, F3, T4, F4, dim_red)
    
#ROC & DET curves
print("Plotting ROC & DET curves for Handwritten data")
plot_all_classifiers_together(score_knn, score_logreg, score_svm, score_ann)
#ROC & DET curves using PCA
print("Plotting ROC & DET curves using PCA for Handwritten data")
plot_all_classifiers_together(score_knn_pca, score_logreg_pca, score_svm_pca, score_ann_pca, "using PCA")
#ROC & DET curves using LDA
print("Plotting ROC & DET curves using LDA for Handwritten data")
plot_all_classifiers_together(score_knn_lda, score_logreg_lda, score_svm_lda, score_ann_lda, "using LDA")