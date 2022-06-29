from scipy.spatial.distance import euclidean as eu
from numpy import linalg as la
from math import dist
import math
from numpy.linalg import norm 
import numpy as np
import os
from statistics import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import norm as det_norm
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
import random
import subprocess

##Dynamic Time Warping of Isolated Spoken digits data
#DTW cost calculation function
def dtw(x, y):
    nx = len(x)
    ny = len(y)
    table = np.zeros((nx+1, ny+1))
    
    table[1:, 0] = np.inf
    table[0, 1:] = np.inf
    
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            d = norm(x[i-1, :] - y[j-1, :])
            table[i, j] = d + min(table[i-1, j], min(table[i, j-1], table[i-1, j-1]))
    return table[nx, ny]
#Reading training data for Isolated spoken digits
path = 'Digits/train'

n2 = n6 = n7 = n9 = no = 1
flag = 0
train_nf = [0]
train_digit = []
for file in os.listdir(path):
    if file.endswith(".mfcc"):
        f = open(f"{path}/{file}", "r")
        nc, nf = f.readline().split()
        
        nc = int(nc)
        nf = int(nf)
        train_nf.append(train_nf[-1] + nf)
        train_digit.append(file[3])

        name = np.empty((1,nc))
        for i in range(nf):
            y = f.readline().split()
            for i in range(nc):
                y[i] = float(y[i])
            name = np.row_stack((name,y))
            
        name = np.delete(name, (0), axis=0)
        if flag == 0:
            globals()['train'] = name
        else:
            globals()['train'] = np.row_stack((train,name))
        flag = 1
        if file[3] == '2':
            globals()[f"train{file[3]}{n2}"] = name
            n2 = n2 + 1
        elif file[3] == '6':
            globals()[f"train{file[3]}{n6}"] = name
            n6 = n6 + 1
        elif file[3] == '7':
            globals()[f"train{file[3]}{n7}"] = name
            n7 = n7 + 1
        elif file[3] == '9':
            globals()[f"train{file[3]}{n9}"] = name
            n9 = n9 + 1
        elif file[3] == 'o':
            globals()[f"train{file[3]}{no}"] = name
            no = no + 1
#Reading development data for Isolated spoken digits
path = 'Digits/dev'
n2 = n6 = n7 = n9 = no = 1
y2 = []
y6 = []
y7 = []
y9 = []
yo = []
for file in os.listdir(path):
    if file.endswith(".mfcc"):
        f = open(f"{path}/{file}", "r")
        nc, nf = f.readline().split()
        
        nc = int(nc)
        nf = int(nf)

        name = np.empty((1,nc))
        for i in range(nf):
            y = f.readline().split()
            for i in range(nc):
                y[i] = float(y[i])
            name = np.row_stack((name,y))
            
        name = np.delete(name, (0), axis=0)

        if file[3] == '2':
            globals()[f"dev{file[3]}{n2}"] = name
            y2.append(file[3])
            n2 = n2 + 1
        elif file[3] == '6':
            globals()[f"dev{file[3]}{n6}"] = name
            y6.append(file[3])
            n6 = n6 + 1
        elif file[3] == '7':
            globals()[f"dev{file[3]}{n7}"] = name
            y7.append(file[3])
            n7 = n7 + 1
        elif file[3] == '9':
            globals()[f"dev{file[3]}{n9}"] = name
            y9.append(file[3])
            n9 = n9 + 1
        elif file[3] == 'o':
            globals()[f"dev{file[3]}{no}"] = name
            yo.append(file[3])
            no = no + 1
            
y_true = y2 + y6 + y7 + y9 + yo
#Predicting spoken digit by top K minimum dtw costs
def predict(dev):
    mod_list = []
    for i in range(1,13):
        print("DTW cost calculation for digit", dev[3], "test case", i)
        d = []
        for j in range(len(train_nf) - 1):
            d.append(dtw(globals()[f"{dev}{i}"], train[train_nf[j]:train_nf[j + 1]]))

        f = {}
        for k in range(len(d)):
            f[d[k]] = train_digit[k]
        f = dict(sorted(f.items()))
        globals()[f"dtw{dev[3]}{i}"] = f
        
        lst = dict(list(f.items())[:39])
        m = mode(list(lst.values()))
        mod_list.append(m)

    return mod_list
#Calculating list of predicted digists
y_pred_dtw = predict("dev2") + predict("dev6") + predict("dev7") + predict("dev9") + predict("devo")
#Confusion Matrix for DTW
fig, ax = plt.subplots(figsize=(8, 6))
val= ['2', '6', '7', '9', 'o']
cm = confusion_matrix(y_true, y_pred_dtw)

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(val), len(val))

ax = sns.heatmap(cm/np.sum(cm), annot=labels, fmt='',annot_kws={"size": 12})

ax.xaxis.set_ticklabels(val) 
ax.yaxis.set_ticklabels(val)

ax.set_title('Confusion matrix for speech data by DTW');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values ');
plt.savefig("Confusion_digit_dtw.jpg")
print("Confusion matrix for DTW of speech data is saved as Confusion_digit_dtw.jpg")
plt.show()

##HMM for Isolated spoken digit data
#K-means clustering
def find_clusters(X, n_clusters, rseed=23):
    centers = np.array(random.choices(X, k=n_clusters))
    i = 0
    while i < 10:
        #Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        #Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        #Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
        i = i + 1
    return centers, labels
#Codebook generation by k-means clustering
centers, labels = find_clusters(train, 5)
#Create symbol sequence for training data of digit 2
f = open("train2.hmm.seq", "w")

for i in range(1,40):
    x = globals()[f"train2{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of digit 6
f = open("train6.hmm.seq", "w")

for i in range(1,40):
    x = globals()[f"train6{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of digit 7
f = open("train7.hmm.seq", "w")

for i in range(1,40):
    x = globals()[f"train7{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of digit 9
f = open("train9.hmm.seq", "w")

for i in range(1,40):
    x = globals()[f"train9{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of digit o
f = open("traino.hmm.seq", "w")

for i in range(1,40):
    x = globals()[f"traino{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()
#Create symbol sequence for development data of digit 2
f = open("dev2.hmm.seq", "w")

for i in range(1,13):
    x = globals()[f"dev2{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of digit 6
f = open("dev6.hmm.seq", "w")

for i in range(1,13):
    x = globals()[f"dev6{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of digit 7
f = open("dev7.hmm.seq", "w")

for i in range(1,13):
    x = globals()[f"dev7{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of digit 9
f = open("dev9.hmm.seq", "w")

for i in range(1,13):
    x = globals()[f"dev9{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of digit o
f = open("devo.hmm.seq", "w")

for i in range(1,13):
    x = globals()[f"devo{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()
#Training HMM for Isolated spoken digit data using given HMM-code
for file in os.listdir():
    if file.startswith("train2") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","5","5",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("train6") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","6","5",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("train7") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","7","5",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("train9") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","9","5",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("traino") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","8","5",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
#Testing each class development data against trained HMM model for each class
for file in os.listdir():
    if file.startswith("dev2"):
        output = subprocess.run(["./HMM-Code/test_hmm","dev2.hmm.seq","train2.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev2.hmm.seq","train6.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev2.hmm.seq","train7.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev2.hmm.seq","train9.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev2.hmm.seq","traino.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("dev6"):
        output = subprocess.run(["./HMM-Code/test_hmm","dev6.hmm.seq","train2.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev6.hmm.seq","train6.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev6.hmm.seq","train7.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev6.hmm.seq","train9.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev6.hmm.seq","traino.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("dev7"):
        output = subprocess.run(["./HMM-Code/test_hmm","dev7.hmm.seq","train2.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev7.hmm.seq","train6.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev7.hmm.seq","train7.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev7.hmm.seq","train9.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev7.hmm.seq","traino.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("dev9"):
        output = subprocess.run(["./HMM-Code/test_hmm","dev9.hmm.seq","train2.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev9.hmm.seq","train6.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev9.hmm.seq","train7.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev9.hmm.seq","train9.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","dev9.hmm.seq","traino.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("devo"):
        output = subprocess.run(["./HMM-Code/test_hmm","devo.hmm.seq","train2.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devo.hmm.seq","train6.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devo.hmm.seq","train7.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devo.hmm.seq","train9.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devo.hmm.seq","traino.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
#Scores for digit 2
s_2 = np.zeros((12,5))
for file in os.listdir():
    if file.startswith("score2"):
        f = open(f"{file}", "r")
        if file[6] == '2':
            for i in range(12):
                s_2[i,0] = np.array(f.readline().split())
        elif file[6] == '6':
            for i in range(12):
                s_2[i,1] = np.array(f.readline().split())
        elif file[6] == '7':
            for i in range(12):
                s_2[i,2] = np.array(f.readline().split())
        elif file[6] == '9':
            for i in range(12):
                s_2[i,3] = np.array(f.readline().split())
        elif file[6] == 'o':
            for i in range(12):
                s_2[i,4] = np.array(f.readline().split())

#Scores for digit 6
s_6 = np.zeros((12,5))
for file in os.listdir():
    if file.startswith("score6"):
        f = open(f"{file}", "r")
        if file[6] == '2':
            for i in range(12):
                s_6[i,0] = np.array(f.readline().split())
        elif file[6] == '6':
            for i in range(12):
                s_6[i,1] = np.array(f.readline().split())
        elif file[6] == '7':
            for i in range(12):
                s_6[i,2] = np.array(f.readline().split())
        elif file[6] == '9':
            for i in range(12):
                s_6[i,3] = np.array(f.readline().split())
        elif file[6] == 'o':
            for i in range(12):
                s_6[i,4] = np.array(f.readline().split())

#Scores for digit 7
s_7 = np.zeros((12,5))
for file in os.listdir():
    if file.startswith("score7"):
        f = open(f"{file}", "r")
        if file[6] == '2':
            for i in range(12):
                s_7[i,0] = np.array(f.readline().split())
        elif file[6] == '6':
            for i in range(12):
                s_7[i,1] = np.array(f.readline().split())
        elif file[6] == '7':
            for i in range(12):
                s_7[i,2] = np.array(f.readline().split())
        elif file[6] == '9':
            for i in range(12):
                s_7[i,3] = np.array(f.readline().split())
        elif file[6] == 'o':
            for i in range(12):
                s_7[i,4] = np.array(f.readline().split())

#Scores for digit 9
s_9 = np.zeros((12,5))
for file in os.listdir():
    if file.startswith("score9"):
        f = open(f"{file}", "r")
        if file[6] == '2':
            for i in range(12):
                s_9[i,0] = np.array(f.readline().split())
        elif file[6] == '6':
            for i in range(12):
                s_9[i,1] = np.array(f.readline().split())
        elif file[6] == '7':
            for i in range(12):
                s_9[i,2] = np.array(f.readline().split())
        elif file[6] == '9':
            for i in range(12):
                s_9[i,3] = np.array(f.readline().split())
        elif file[6] == 'o':
            for i in range(12):
                s_9[i,4] = np.array(f.readline().split())

#Scores for digit o
s_o = np.zeros((12,5))
for file in os.listdir():
    if file.startswith("scoreo"):
        f = open(f"{file}", "r")
        if file[6] == '2':
            for i in range(12):
                s_o[i,0] = np.array(f.readline().split())
        elif file[6] == '6':
            for i in range(12):
                s_o[i,1] = np.array(f.readline().split())
        elif file[6] == '7':
            for i in range(12):
                s_o[i,2] = np.array(f.readline().split())
        elif file[6] == '9':
            for i in range(12):
                s_o[i,3] = np.array(f.readline().split())
        elif file[6] == 'o':
            for i in range(12):
                s_o[i,4] = np.array(f.readline().split())
#Prediction of digits from HMM model
y_pred_hmm = list(s_2.argmax(axis=1)) + list(s_6.argmax(axis=1)) + list(s_7.argmax(axis=1)) + list(s_9.argmax(axis=1)) + list(s_o.argmax(axis=1))

for i in range(60):
    if y_pred_hmm[i] == 0:
        y_pred_hmm[i] = '2'
    elif y_pred_hmm[i] == 1:
        y_pred_hmm[i] = '6'
    elif y_pred_hmm[i] == 2:
        y_pred_hmm[i] = '7'
    elif y_pred_hmm[i] == 3:
        y_pred_hmm[i] = '9'
    elif y_pred_hmm[i] == 4:
        y_pred_hmm[i] = 'o'

#Confusion Matrix for HMM
fig, ax = plt.subplots(figsize=(8, 6))
val= ['2', '6', '7', '9', 'o']
cm = confusion_matrix(y_true, y_pred_hmm)

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(val), len(val))

ax = sns.heatmap(cm/np.sum(cm), annot=labels, fmt='',annot_kws={"size": 12})

ax.xaxis.set_ticklabels(val) 
ax.yaxis.set_ticklabels(val)

ax.set_title('Confusion matrix for speech data by HMM');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values ');
plt.savefig("Confusion_digit_hmm.jpg")
print("Confusion matrix for HMM of speech data is saved as Confusion_digit_hmm.jpg")
plt.show()
#ROC & DET curve for speech data by DTW
#Function to count number of correct and incorrect predictions in DTW
def dtw_count(i, dig, tr):
    f = globals()[f"dtw{dig}{i}"]
    lst = dict(list(f.items())[:39])
    
    from collections import Counter
    res = Counter(lst.values())
    return res[tr]

dtw_roc = np.zeros((39,2))
dtw_det = np.zeros((39,2))

for th in range(1, 39):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    # Test for digit 2
    for i in range(1, 13):
        #Test against digit 2
        if dtw_count(i, '2', '2') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit 6
        if dtw_count(i, '2', '6') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if dtw_count(i, '2', '7') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if dtw_count(i, '2', '9') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if dtw_count(i, '2', 'o') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit 6
    for i in range(1, 13):
        #Test against digit 2
        if dtw_count(i, '6', '2') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if dtw_count(i, '6', '6') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit 7
        if dtw_count(i, '6', '7') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if dtw_count(i, '6', '9') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if dtw_count(i, '6', 'o') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit 7
    for i in range(1, 13):
        #Test against digit 2
        if dtw_count(i, '7', '2') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if dtw_count(i, '7', '6') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if dtw_count(i, '7', '7') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit 9
        if dtw_count(i, '7', '9') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if dtw_count(i, '7', 'o') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit 9
    for i in range(1, 13):
        #Test against digit 2
        if dtw_count(i, '9', '2') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if dtw_count(i, '9', '6') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if dtw_count(i, '9', '7') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if dtw_count(i, '9', '9') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit o
        if dtw_count(i, '9', 'o') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit o
    for i in range(1, 13):
        #Test against digit 2
        if dtw_count(i, 'o', '2') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if dtw_count(i, 'o', '6') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if dtw_count(i, 'o', '7') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if dtw_count(i, 'o', '9') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if dtw_count(i, 'o', 'o') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
            
    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    FNR = 1 - TPR
    
    dtw_roc[th - 1, :] = [FPR, TPR]
    dtw_det[th - 1, :] = [det_norm.ppf(FPR), det_norm.ppf(FNR)]

#ROC & DET curve for speech data by HMM
#Range calculation for threshold in ROC & DET curves for HMM
s_max = max(np.amax(s_2),np.amax(s_6),np.amax(s_7),np.amax(s_9),np.amax(s_o))
s_min = min(np.amin(s_2),np.amin(s_6),np.amin(s_7),np.amin(s_9),np.amin(s_o))
n_hmm = int(s_max - s_min)
hmm_roc = np.zeros((n_hmm,2))
hmm_det = np.zeros((n_hmm,2))

count = 0
for th in range(int(s_min), int(s_max) - 1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    # Test for digit 2
    for i in range(12):
        #Test against digit 2
        if s_2[i,0] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit 6
        if s_2[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if s_2[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if s_2[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if s_2[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit 6
    for i in range(12):
        #Test against digit 2
        if s_6[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if s_6[i,1] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit 7
        if s_6[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if s_6[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if s_6[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit 7
    for i in range(12):
        #Test against digit 2
        if s_7[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if s_7[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if s_7[i,2] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit 9
        if s_7[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if s_7[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit 9
    for i in range(12):
        #Test against digit 2
        if s_9[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if s_9[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if s_9[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if s_9[i,3] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against digit o
        if s_9[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for digit o
    for i in range(12):
        #Test against digit 2
        if s_o[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 6
        if s_o[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 7
        if s_o[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit 9
        if s_o[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against digit o
        if s_o[i,4] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
            
    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    FNR = 1 - TPR
    
    hmm_roc[count, :] = [FPR, TPR]
    hmm_det[count, :] = [det_norm.ppf(FPR), det_norm.ppf(FNR)]
    count = count + 1
    
plt.subplots(figsize=(6,6))
plt.plot(dtw_roc[:, 0], dtw_roc[:, 1])
plt.plot(hmm_roc[:, 0], hmm_roc[:, 1])
plt.grid()
plt.title("ROC Curves for Isolated spoken digits")
plt.legend(['DTW', 'HMM'], loc = 'lower right')
plt.xlabel(xlabel='False Positive Rate (FPR)')
plt.ylabel(ylabel='True Positive Rate (TPR)')
plt.savefig("ROC_digit.jpg")
print("ROC curve for Isolated spoken digits is saved as ROC_digit.jpg")
plt.show()
plt.plot(dtw_det[:, 0], dtw_det[:, 1])
plt.plot(hmm_det[:, 0], hmm_det[:, 1])
plt.grid()
plt.title("DET Curves for Isolated spoken digits")
plt.legend(['DTW', 'HMM'], loc = 'lower right')
plt.xlabel(xlabel='False Alarm Rate')
plt.ylabel(ylabel='Missed Detection Rate')
plt.savefig("DET_digit.jpg")
print("DET curve for Isolated spoken digits is saved as DET_digit.jpg")
plt.show()