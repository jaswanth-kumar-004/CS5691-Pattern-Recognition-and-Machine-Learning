from statistics import mode
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
import math 
from numpy import diff

#Dynamic Time Warping of Handwritten Telugu letters
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
#Reading training data for Handwritten Telugu Characters
path = 'Written/train'
flag = 0
train_nc = [0]
train_char = []
for file in os.listdir(path):
    if file.endswith(".txt"):
        f = open(f"{path}/{file}", "r")
        A = f.readline().split()
        
        nc = int(A[0])
        train_nc.append(train_nc[-1] + nc)
        if(file[0:1] == 'c'):
            train_char.append(file[0:3])
        else:
            train_char.append(file[0:2])

        x = []
        y = []
        i = 1
        while i < len(A):
            x.append(float(A[i]))
            i += 1
            y.append(float(A[i]))
            i += 1
        t = np.column_stack((x,y))

        if flag == 0:
            globals()['train'] = t
        else:
            globals()['train'] = np.row_stack((train,t))
        flag = 1
#Reading development data for Handwritten Telugu Characters
path = 'Written/dev'
nai = nbA = nchA = nlA = ntA = 1
yai = []
ybA = []
ychA = []
ylA = []
ytA = []

for file in os.listdir(path):
    if file.endswith(".txt"):
        f = open(f"{path}/{file}", "r")
        A = f.readline().split()
        
        nc = int(A[0])
        train_nc.append(train_nc[-1] + nc)
        train_char.append(file[0:2])

        name = np.empty((1,2))
        x = []
        y = []
        i = 1
        while i < len(A):
            x.append(float(A[i]))
            i += 1
            y.append(float(A[i]))
            i += 1
        t = np.column_stack((x,y))

        if file[0:2] == 'ai':
            globals()[f"dev{file[0:2]}{nai}"] = t
            yai.append(file[0:2])
            nai = nai + 1
        elif file[0:2] == 'bA':
            globals()[f"dev{file[0:2]}{nbA}"] = t
            ybA.append(file[0:2])
            nbA = nbA + 1
        elif file[0:3] == 'chA':
            globals()[f"dev{file[0:3]}{nchA}"] = t
            ychA.append(file[0:3])
            nchA = nchA + 1
        elif file[0:2] == 'lA':
            globals()[f"dev{file[0:2]}{nlA}"] = t
            ylA.append(file[0:2])
            nlA = nlA + 1
        elif file[0:2] == 'tA':
            globals()[f"dev{file[0:2]}{ntA}"] = t
            ytA.append(file[0:2])
            ntA = ntA + 1
            
y_true_dtw = yai + ybA + ychA + ylA + ytA
#Predicting spoken digit by top K minimum dtw costs
def predict_char(dev):
    if(dev[3:4] == 'c'):
        ch = dev[3:6]
    else:
        ch = dev[3:5]
        
    mod_list = []
    for i in range(1, 21):
        print("DTW cost calculation of character", ch, "for test case", i)
        d = []
        for j in range(len(train_nc) - 1):
            d.append(dtw(globals()[f"dev{ch}{i}"], train[train_nc[j]:train_nc[j + 1]]))
            
        f = {}
        for k in range(len(d)):
            f[d[k]] = train_char[k]
        f = dict(sorted(f.items()))
        globals()[f"dtw{ch}{i}"] = f
        
        lst = dict(list(f.items())[:66])
        m = mode(list(lst.values()))
        mod_list.append(m)

    return mod_list
#Calculating list of predicted characters
y_pred_dtw = predict_char("devai") + predict_char("devbA") +  predict_char("devchA") +  predict_char("devlA") + predict_char("devtA")
#Confusion Matrix for DTW
fig, ax = plt.subplots(figsize=(8, 6))
val= ['ai', 'bA', 'chA', 'lA', 'tA']
cm = confusion_matrix(y_true_dtw, y_pred_dtw)

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(val), len(val))

ax = sns.heatmap(cm/np.sum(cm), annot=labels,fmt='',annot_kws={"size": 12})

ax.xaxis.set_ticklabels(val) 
ax.yaxis.set_ticklabels(val)

ax.set_title('Confusion matrix for written data by DTW');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values ');
plt.savefig("Confusion_written_dtw.jpg")
print("Confusion matrix for DTW of written data is saved as Confusion_written_dtw.jpg")
plt.show()

##HMM for Online Hanwritten Telugu characters
#Reading training data for Hanwritten Telugu characters
nai = nbA = nchA = nlA = ntA = 1
path = 'Written/train'
flag = 0
for file in os.listdir(path):
    if file.endswith(".txt"):
        f = open(f"{path}/{file}", "r")
        A = f.readline().split()

        x = []
        y = []
        i = 1
        while i < len(A):
            x.append(float(A[i]))
            i += 1
            y.append(float(A[i]))
            i += 1
        t = np.column_stack((x,y))
        
        np.seterr(divide='ignore')
        slope = diff(t[:,0])/diff(t[:,1])
        for i in range(slope.shape[0]):
            if math.isinf(slope[i]):
                slope[i] = 10000000

        t = np.column_stack((t[0:t.shape[0]-1],slope))
        
        if flag == 0:
            globals()['train'] = t
        else:
            globals()['train'] = np.row_stack((train,t))
        flag = 1
        
        if file[0:2] == 'ai':
            globals()[f"train{file[0:2]}{nai}"] = t
            nai = nai + 1
        elif file[0:2] == 'bA':
            globals()[f"train{file[0:2]}{nbA}"] = t
            nbA = nbA + 1
        elif file[0:3] == 'chA':
            globals()[f"train{file[0:3]}{nchA}"] = t
            nchA = nchA + 1
        elif file[0:2] == 'lA':
            globals()[f"train{file[0:2]}{nlA}"] = t
            nlA = nlA + 1
        elif file[0:2] == 'tA':
            globals()[f"train{file[0:2]}{ntA}"] = t
            ntA = ntA + 1
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
centers, labels = find_clusters(train, 16)
#Create symbol sequence for training data of charcter ai
f = open("trainai.hmm.seq", "w")

for i in range(1,nai):
    x = globals()[f"trainai{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of character bA
f = open("trainbA.hmm.seq", "w")

for i in range(1,nbA):
    x = globals()[f"trainbA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of character chA
f = open("trainchA.hmm.seq", "w")

for i in range(1,nchA):
    x = globals()[f"trainchA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of character lA
f = open("trainlA.hmm.seq", "w")

for i in range(1,nlA):
    x = globals()[f"trainlA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for training data of character tA
f = open("traintA.hmm.seq", "w")

for i in range(1,ntA):
    x = globals()[f"traintA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Reading development data for Hanwritten Telugu characters
path = 'Written/dev'
nai = nbA = nchA = nlA = ntA = 1
yai = []
ybA = []
ychA = []
ylA = []
ytA = []

for file in os.listdir(path):
    if file.endswith(".txt"):
        f = open(f"{path}/{file}", "r")
        A = f.readline().split()

        name = np.empty((1,2))
        x = []
        y = []
        i = 1
        while i < len(A):
            x.append(float(A[i]))
            i += 1
            y.append(float(A[i]))
            i += 1
        t = np.column_stack((x,y))
        
        np.seterr(divide='ignore')
        slope = diff(t[:,0])/diff(t[:,1])
        for i in range(slope.shape[0]):
            if math.isinf(slope[i]):
                slope[i] = 10000000

        t = np.column_stack((t[0:t.shape[0]-1],slope))
        
        for i in range(t.shape[0]):
            if math.isinf(t[i,2]):
                t[i,2] = 10000000
                
        if file[0:2] == 'ai':
            globals()[f"dev{file[0:2]}{nai}"] = t
            yai.append(file[0:2])
            nai = nai + 1
        elif file[0:2] == 'bA':
            globals()[f"dev{file[0:2]}{nbA}"] = t
            ybA.append(file[0:2])
            nbA = nbA + 1
        elif file[0:3] == 'chA':
            globals()[f"dev{file[0:3]}{nchA}"] = t
            ychA.append(file[0:3])
            nchA = nchA + 1
        elif file[0:2] == 'lA':
            globals()[f"dev{file[0:2]}{nlA}"] = t
            ylA.append(file[0:2])
            nlA = nlA + 1
        elif file[0:2] == 'tA':
            globals()[f"dev{file[0:2]}{ntA}"] = t
            ytA.append(file[0:2])
            ntA = ntA + 1
            
y_true_hmm = yai + ybA + ychA + ylA + ytA

#Create symbol sequence for development data of charcter ai
f = open("devai.hmm.seq", "w")

for i in range(1,nai):
    x = globals()[f"devai{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of character bA
f = open("devbA.hmm.seq", "w")

for i in range(1,nbA):
    x = globals()[f"devbA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of character chA
f = open("devchA.hmm.seq", "w")

for i in range(1,nchA):
    x = globals()[f"devchA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of character lA
f = open("devlA.hmm.seq", "w")

for i in range(1,nlA):
    x = globals()[f"devlA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Create symbol sequence for development data of character tA
f = open("devtA.hmm.seq", "w")

for i in range(1,ntA):
    x = globals()[f"devtA{i}"]
    
    vq = []
    for j in range(len(x)):
        d = []
        for l in range(len(centers)):
            d.append(norm(x[j,:] - centers[l, :]))
        vq.append(d.index(min(d)))
    line = ' '.join(map(str, vq)) + '\n'
    f.write(line)
    
f.close()

#Training HMM for Handwritten characters data using given HMM-code
for file in os.listdir():
    if file.startswith("trainai") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","14","16",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("trainbA") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","15","16",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("trainchA") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","15","16",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("trainlA") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","15","16",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("traintA") & file.endswith(".seq"):
        output = subprocess.run(["./HMM-Code/train_hmm",file,"1234","15","16",".01"],stdout=subprocess.PIPE,universal_newlines=True).stdout
#Testing each class development data against trained HMM model for each class
for file in os.listdir():
    if file.startswith("devai"):
        output = subprocess.run(["./HMM-Code/test_hmm","devai.hmm.seq","trainai.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devai.hmm.seq","trainbA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devai.hmm.seq","trainchA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devai.hmm.seq","trainlA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devai.hmm.seq","traintA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("devbA"):
        output = subprocess.run(["./HMM-Code/test_hmm","devbA.hmm.seq","trainai.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devbA.hmm.seq","trainbA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devbA.hmm.seq","trainchA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devbA.hmm.seq","trainlA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devbA.hmm.seq","traintA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("devchA"):
        output = subprocess.run(["./HMM-Code/test_hmm","devchA.hmm.seq","trainai.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devchA.hmm.seq","trainbA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devchA.hmm.seq","trainchA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devchA.hmm.seq","trainlA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devchA.hmm.seq","traintA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("devlA"):
        output = subprocess.run(["./HMM-Code/test_hmm","devlA.hmm.seq","trainai.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devlA.hmm.seq","trainbA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devlA.hmm.seq","trainchA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devlA.hmm.seq","trainlA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devlA.hmm.seq","traintA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
    elif file.startswith("devtA"):
        output = subprocess.run(["./HMM-Code/test_hmm","devtA.hmm.seq","trainai.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devtA.hmm.seq","trainbA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devtA.hmm.seq","trainchA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devtA.hmm.seq","trainlA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout
        output = subprocess.run(["./HMM-Code/test_hmm","devtA.hmm.seq","traintA.hmm.seq.hmm"],stdout=subprocess.PIPE,universal_newlines=True).stdout

#Scores for character ai
s_ai = np.zeros((20,5))
for file in os.listdir():
    if file.startswith("scoreai"):
        f = open(f"{file}", "r")
        if file[7] == 'a':
            for i in range(20):
                s_ai[i,0] = np.array(f.readline().split())
        elif file[7] == 'b':
            for i in range(20):
                s_ai[i,1] = np.array(f.readline().split())
        elif file[7] == 'c':
            for i in range(20):
                s_ai[i,2] = np.array(f.readline().split())
        elif file[7] == 'l':
            for i in range(20):
                s_ai[i,3] = np.array(f.readline().split())
        elif file[7] == 't':
            for i in range(20):
                s_ai[i,4] = np.array(f.readline().split())

#Scores for character bA
s_bA = np.zeros((20,5))
for file in os.listdir():
    if file.startswith("scorebA"):
        f = open(f"{file}", "r")
        if file[7] == 'a':
            for i in range(20):
                s_bA[i,0] = np.array(f.readline().split())
        elif file[7] == 'b':
            for i in range(20):
                s_bA[i,1] = np.array(f.readline().split())
        elif file[7] == 'c':
            for i in range(20):
                s_bA[i,2] = np.array(f.readline().split())
        elif file[7] == 'l':
            for i in range(20):
                s_bA[i,3] = np.array(f.readline().split())
        elif file[7] == 't':
            for i in range(20):
                s_bA[i,4] = np.array(f.readline().split())


#Scores for character chA
s_chA = np.zeros((20,5))
for file in os.listdir():
    if file.startswith("scorechA"):
        f = open(f"{file}", "r")
        if file[8] == 'a':
            for i in range(20):
                s_chA[i,0] = np.array(f.readline().split())
        elif file[8] == 'b':
            for i in range(20):
                s_chA[i,1] = np.array(f.readline().split())
        elif file[8] == 'c':
            for i in range(20):
                s_chA[i,2] = np.array(f.readline().split())
        elif file[8] == 'l':
            for i in range(20):
                s_chA[i,3] = np.array(f.readline().split())
        elif file[8] == 't':
            for i in range(20):
                s_chA[i,4] = np.array(f.readline().split())


#Scores for character lA
s_lA = np.zeros((20,5))
for file in os.listdir():
    if file.startswith("scorelA"):
        f = open(f"{file}", "r")
        if file[7] == 'a':
            for i in range(20):
                s_lA[i,0] = np.array(f.readline().split())
        elif file[7] == 'b':
            for i in range(20):
                s_lA[i,1] = np.array(f.readline().split())
        elif file[7] == 'c':
            for i in range(20):
                s_lA[i,2] = np.array(f.readline().split())
        elif file[7] == 'l':
            for i in range(20):
                s_lA[i,3] = np.array(f.readline().split())
        elif file[7] == 't':
            for i in range(20):
                s_lA[i,4] = np.array(f.readline().split())


#Scores for character tA
s_tA = np.zeros((20,5))
for file in os.listdir():
    if file.startswith("scoretA"):
        f = open(f"{file}", "r")
        if file[7] == 'a':
            for i in range(20):
                s_tA[i,0] = np.array(f.readline().split())
        elif file[7] == 'b':
            for i in range(20):
                s_tA[i,1] = np.array(f.readline().split())
        elif file[7] == 'c':
            for i in range(20):
                s_tA[i,2] = np.array(f.readline().split())
        elif file[7] == 'l':
            for i in range(20):
                s_tA[i,3] = np.array(f.readline().split())
        elif file[7] == 't':
            for i in range(20):
                s_tA[i,4] = np.array(f.readline().split())
#Prediction of characters from HMM model
y_pred_hmm = list(s_ai.argmax(axis=1)) + list(s_bA.argmax(axis=1)) + list(s_chA.argmax(axis=1)) + list(s_lA.argmax(axis=1)) + list(s_tA.argmax(axis=1))
for i in range(100):
    if y_pred_hmm[i] == 0:
        y_pred_hmm[i] = 'ai'
    elif y_pred_hmm[i] == 1:
        y_pred_hmm[i] = 'bA'
    elif y_pred_hmm[i] == 2:
        y_pred_hmm[i] = 'chA'
    elif y_pred_hmm[i] == 3:
        y_pred_hmm[i] = 'lA'
    elif y_pred_hmm[i] == 4:
        y_pred_hmm[i] = 'tA'
#Confusion Matrix for HMM
fig, ax = plt.subplots(figsize=(8, 6))

val= ['ai', 'bA', 'chA', 'lA', 'tA']
cm = confusion_matrix(y_true_hmm, y_pred_hmm)

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(val), len(val))

ax = sns.heatmap(cm/np.sum(cm), annot=labels, fmt='',annot_kws={"size": 12})

ax.xaxis.set_ticklabels(val) 
ax.yaxis.set_ticklabels(val)

ax.set_title('Confusion matrix for written data by HMM');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values ');
plt.savefig("Confusion_written_hmm.jpg")
print("Confusion matrix for HMM of written data is saved as Confusion_written_hmm.jpg")
plt.show()

#ROC & DET curve for written data by DTW
#Function to count number of correct and incorrect predictions in DTW
def dtw_count(i, char, tr):
    f = globals()[f"dtw{char}{i}"]
    lst = dict(list(f.items())[:66])
    
    from collections import Counter
    res = Counter(lst.values())
    return res[tr]

dtw_roc = np.zeros((66,2))
dtw_det = np.zeros((66,2))

for th in range(1, 66):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    #Test for character ai
    for i in range(1, 21):
        #Test against character ai
        if dtw_count(i, 'ai', 'ai') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character bA
        if dtw_count(i, 'ai', 'bA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if dtw_count(i, 'ai', 'chA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if dtw_count(i, 'ai', 'lA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if dtw_count(i, 'ai', 'tA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    #Test for character bA
    for i in range(1, 21):
        #Test against character ai
        if dtw_count(i, 'bA', 'ai') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if dtw_count(i, 'bA', 'bA') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character chA
        if dtw_count(i, 'bA', 'chA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if dtw_count(i, 'bA', 'lA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if dtw_count(i, 'bA', 'tA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    #Test for character chA
    for i in range(1, 21):
        #Test against character ai
        if dtw_count(i, 'chA', 'ai') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if dtw_count(i, 'chA', 'bA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if dtw_count(i, 'chA', 'chA') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character lA
        if dtw_count(i, 'chA', 'lA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if dtw_count(i, 'chA', 'tA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    #Test for character lA
    for i in range(1, 21):
        #Test against character ai
        if dtw_count(i, 'lA', 'ai') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if dtw_count(i, 'lA', 'bA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if dtw_count(i, 'lA', 'chA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if dtw_count(i, 'lA', 'lA') > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character tA
        if dtw_count(i, 'lA', 'tA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    #Test for character tA
    for i in range(1, 21):
        #Test against character ai
        if dtw_count(i, 'tA', 'ai') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if dtw_count(i, 'tA', 'bA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if dtw_count(i, 'tA', 'chA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if dtw_count(i, 'tA', 'lA') > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if dtw_count(i, 'tA', 'tA') > th - 1:
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
s_max = max(np.amax(s_ai),np.amax(s_bA),np.amax(s_chA),np.amax(s_lA),np.amax(s_tA))
s_min = min(np.amax(s_ai),np.amax(s_bA),np.amax(s_chA),np.amax(s_lA),np.amax(s_tA))
n_hmm = int(s_max - s_min)

hmm_roc = np.zeros((n_hmm,2))
hmm_det = np.zeros((n_hmm,2))

count = 0
for th in range(int(s_min), int(s_max) - 1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    # Test for character ai
    for i in range(20):
        #Test against character ai
        if s_ai[i,0] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character bA
        if s_ai[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if s_ai[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if s_ai[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if s_ai[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for character bA
    for i in range(20):
        #Test against character ai
        if s_bA[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if s_bA[i,1] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character chA
        if s_bA[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if s_bA[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if s_bA[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for character chA
    for i in range(20):
        #Test against character ai
        if s_chA[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if s_chA[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if s_chA[i,2] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character lA
        if s_chA[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if s_chA[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for character lA
    for i in range(20):
        #Test against character ai
        if s_lA[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if s_lA[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if s_lA[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if s_lA[i,3] > th - 1:
            TP = TP + 1
        else:
            FN = FN + 1
        #Test against character tA
        if s_lA[i,4] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    # Test for character tA
    for i in range(20):
        #Test against character ai
        if s_tA[i,0] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character bA
        if s_tA[i,1] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character chA
        if s_tA[i,2] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character tA
        if s_tA[i,3] > th - 1:
            FP = FP + 1
        else:
            TN = TN + 1
        #Test against character lA
        if s_tA[i,4] > th - 1:
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
plt.title("ROC Curves")
plt.legend(['DTW','HMM'], loc = 'lower right')
plt.xlabel(xlabel='False Positive Rate (FPR)')
plt.ylabel(ylabel='True Positive Rate (TPR)')
plt.savefig("ROC_written.jpg")
print("ROC curve for Handwritten characters is saved as ROC_written.jpg")
plt.show()

plt.plot(dtw_det[:, 0], dtw_det[:, 1])
plt.plot(hmm_det[:, 0], hmm_det[:, 1])
plt.grid()
plt.title("DET Curves")
plt.legend(['DTW','HMM'], loc = 'lower right')
plt.xlabel(xlabel='False Alarm Rate')
plt.ylabel(ylabel='Missed Detection Rate')
plt.savefig("DET_written.jpg")
print("DET curve for Handwritten is saved as DET_written.jpg")
plt.show()