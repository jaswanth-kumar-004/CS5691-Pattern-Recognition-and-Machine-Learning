#Team 34 Code for Bayesian Classifier
#Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

#Read data by taking input from user
data = input("Enter the type of dataset:\n(Write ls for Linearly Separable data)\n(Write nls for Non Linearly Separable data)\n(Write rd for Real data)\n")
A = np.loadtxt('trian_'+data+'.txt', delimiter=",")
B = np.loadtxt('dev_'+data+'.txt', delimiter=",")

n = A.shape[0]
f = int(n/3)
#Segregate each class data
c1_data = A[0:f,0:2]
c2_data = A[f:(2*f),0:2]
c3_data = A[(2*f):n,0:2]
#Caclculate mean of each class data
mu1 = np.mean(c1_data.T,axis=1)
mu2 = np.mean(c2_data.T,axis=1)
mu3 = np.mean(c3_data.T,axis=1)

#Covariance matrices for different cases

#Full covariance matrix, same for all classes 
def cov_matrices_case_1(c1, c2, c3):
    x1_data = np.vstack((c1_data[:,0:1], c2_data[:,0:1],c3_data[:,0:1]))
    x2_data = np.vstack((c1[:,1:2],c2[:,1:2],c3[:,1:2]))
    mu1 = np.mean(x1_data.T,axis=1)
    mu2 = np.mean(x2_data.T,axis=1)
    
    N = x1_data.shape[0]
    
    cov = np.zeros((2,2))
    cov[0,0] = (np.sum((x1_data - mu1)**2))/(N-1)
    cov[1,1] = (np.sum((x2_data - mu2)**2))/(N-1)
    cov[0,1] = (np.sum((x1_data - mu1)*(x2_data - mu2)))/(N-1)
    cov[1,0] = (np.sum((x1_data - mu1)*(x2_data - mu2)))/(N-1)
    return cov, cov, cov

#Full covariance matrix different for each class 
def cov_matrices_case_2(c1, c2, c3):
    sigma1 = np.cov(c1_data.T, bias = False)
    sigma2 = np.cov(c2_data.T, bias = False)
    sigma3 = np.cov(c3_data.T, bias = False)
    return sigma1, sigma2, sigma3
################################
#Covariance matrix equal to sigma^2 * I for each class
def cov_matrices_case_3(c1, c2, c3):
    x1_data = np.vstack((c1_data[:,0:1], c2_data[:,0:1],c3_data[:,0:1]))
    x2_data = np.vstack((c1[:,1:2],c2[:,1:2],c3[:,1:2]))
    mu1 = np.mean(x1_data.T,axis=1)
    mu2 = np.mean(x2_data.T,axis=1)
    
    N = x1_data.shape[0]
    s1 = np.sum((x1_data - mu1)**2)
    s2 = np.sum((x2_data - mu2)**2)
    
    sigma = ((s1 + s2)*0.5)/(N-1)
    cov = sigma*np.identity(2)
    return cov, cov, cov

#Diagonal covariance matrix, same for all classes 
def cov_matrices_case_4(c1, c2, c3):
    s1, s2, s3 = cov_matrices_case_1(c1, c2, c3)
    sigma1 = np.diag(np.diag(s1))
    sigma2 = np.diag(np.diag(s2))
    sigma3 = np.diag(np.diag(s3))
    return sigma1, sigma2, sigma3

#Diagonal covariance matrix different for each class 
def cov_matrices_case_5(c1, c2, c3):
    s1, s2, s3 = cov_matrices_case_2(c1, c2, c3)
    sigma1 = np.diag(np.diag(s1))
    sigma2 = np.diag(np.diag(s2))
    sigma3 = np.diag(np.diag(s3))
    return sigma1, sigma2, sigma3
##########################
#Implementation of 2D gaussian PDF
def multivariate_gaussian(pos, mu, Sigma):
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**2 * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N
def pdf(x, y, mean, cov):
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    return multivariate_gaussian(pos, mean, cov)

#Here each x & y is the meshgrid coordinate
def g1(x, y):
    return pdf(x, y, mean=mu1, cov=sigma1)

def g2(x, y):
    return pdf(x, y, mean=mu2, cov=sigma2)

def g3(x, y):
    return pdf(x, y, mean=mu3, cov=sigma3)
#########################
#Take input for the which case plot are to be made
case = input("Enter the case number:\n(Write number n for case n) ")
sigma1, sigma2, sigma3 = eval('cov_matrices_case_'+case)(c1_data,c2_data,c3_data)
#PDF plots for Gaussians
p = A[:, 0:1]
q = A[:, 1:2]

#Make mesh grid
N = 1000
p = np.linspace(np.amin(p, axis=0), np.amax(p, axis=0), N)
q = np.linspace(np.amin(q, axis=0), np.amax(q, axis=0), N)
X, Y = np.meshgrid(p,q)

#Caculate PDF for each class and add them because they all classes are disjoint
Z = np.empty([1000,1000])
Z = g1(X,Y) + g2(X,Y) + g3(X,Y)
#Plot the pdf
fig = plt.figure(figsize=(9, 9))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z,linewidth=1, antialiased=True, cmap=cm.viridis)
ax.view_init(20, -50)
print("Probability Distribution is saved as PDF.png")
plt.savefig('PDF.png')
plt.show()
######################
#Constant density curves and Eigen vectors
fig = plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z)

def plot_eigen_vector(mu,sigma):
    eigen_values, eigen_vectors = np.linalg.eig(sigma)
    eig_vec1 = eigen_vectors[:,0]
    eig_vec2 = eigen_vectors[:,1]
    plt.quiver(*mu, *eig_vec1, color=['r'], scale=3)
    plt.quiver(*mu, *eig_vec2, color=['b'], scale=3)

plot_eigen_vector(mu1,sigma1)
plot_eigen_vector(mu2,sigma2)
plot_eigen_vector(mu3,sigma3)
print("Contour plots and Eigen vectors are saved as Contour plots & Eigen vector.png")
plt.savefig('Contour plots & Eigen vector.png')
plt.show()
######################
#Decision boundary & Decision surface
p = A[:, 0:1]
q = A[:, 1:2]

#Make meshgrid
N = 1000
p = np.linspace(np.amin(p, axis=0), np.amax(p, axis=0), N)
q = np.linspace(np.amin(q, axis=0), np.amax(q, axis=0), N)
x, y = np.meshgrid(p, q)

#Calculate likelihood of each class for given input vector
z = np.array((g1(x, y), g2(x, y), g3(x, y)))
#Return the index of the greatest likelihood
z = np.argmax(z, axis=0)
cp = plt.contourf(x, y, z,colors=('red','green','blue'), levels=[-0.5, 0.5, 1.5, 2.5])

plt.scatter(A[0:f, 0:1], A[0:f, 1:2],label='Class 1')
plt.scatter(A[f:(2*f), 0:1], A[f:(2*f), 1:2], label='Class 2')
plt.scatter(A[(2*f):n, 0:1], A[(2*f):n, 1:2], label='Class 3')

plt.legend()
plt.xlabel(xlabel=r'input feature x1',size=15)
plt.ylabel(ylabel=r'input feature x2',size=15)
print("Decision Boundary and Surface are saved as Decision Boundary & Surface.png")
plt.savefig('Decision Boundary & Surface.png')
plt.show()
####################
#Function to reduce the name multivariate_gaussian to p
def p(x, mu, sigma):
    return multivariate_gaussian(x, mu, sigma)

#Make Confusion matrix
#Groundtruth
y_true = A[0:n, 2:3]
#Predicted value
y_pred = []
for i in range(n):
    z = np.array((p(A[i, 0:2], mu1, sigma1), p(A[i, 0:2], mu2, sigma2), p(A[i, 0:2], mu3, sigma3)))
    z = np.argmax(z, axis=0) + 1
    y_pred = np.append(y_pred, z)
    
cm = confusion_matrix(y_true, y_pred)

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(3,3)

cm = confusion_matrix(y_true, y_pred)
ax = sns.heatmap(cm/np.sum(cm), annot=labels, fmt='',annot_kws={"size": 12})

ax.set_title('Confusion matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values ');

print("Confusion matrix is saved as Confusion Matrix.png")
plt.savefig('Confusion Matrix.png')
plt.show()
#######################
#Plotting ROC curve
prior = (1/3)

#Score Calculation
def find_s(cov1, cov2, cov3):
    s = np.empty((n,3))
    for i in range(n):
        for j in range(3):
            if j == 0:
                s[i,j] = (p(A[i, 0:2], mu1, cov1) * prior)
            elif j == 1:
                s[i,j] = (p(A[i, 0:2], mu2, cov2) * prior)
            else:
                s[i,j] = (p(A[i, 0:2], mu3, cov3) * prior)
    return s

#Finding score for all five cases
a,b,c = cov_matrices_case_1(c1_data, c2_data, c3_data)
s1 = find_s(a,b,c)

a,b,c = cov_matrices_case_2(c1_data, c2_data, c3_data)
s2 = find_s(a,b,c)

a,b,c = cov_matrices_case_3(c1_data, c2_data, c3_data)
s3 = find_s(a,b,c)

a,b,c = cov_matrices_case_4(c1_data, c2_data, c3_data)
s4 = find_s(a,b,c)

a,b,c = cov_matrices_case_5(c1_data, c2_data, c3_data)
s5 = find_s(a,b,c)

#Function to find True positive Rate & False Positive Rate
def ROC_curve(s):
    TPR = []
    FPR = []
    thres = np.sort(s.flatten())
    for threshold in thres:
        TP = TN = FP = FN = 0
        for i in range(n):
            groundthruth = (A[i,2] - 1)
            for j in range(3):
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
        TPR = np.append(TPR, TP/(TP + FN))
        FPR = np.append(FPR, FP/(FP + TN))
    return (TPR, FPR)

plt.subplots(figsize=(6,6))
T1,F1 = ROC_curve(s1)
plt.plot(F1,T1)
T2,F2 = ROC_curve(s2)
plt.plot(F2,T2)
T3,F3 = ROC_curve(s3)
plt.plot(F3,T3)
T4,F4 = ROC_curve(s4)
plt.plot(F4,T4)
T5,F5 = ROC_curve(s5)
plt.plot(F5,T5)


plt.grid()
plt.title("ROC Curve")
plt.legend(['Case 1','Case 2','Case 3','Case 4','Case 5'])
plt.xlabel(xlabel='False Positive Rate (FPR)')
plt.ylabel(ylabel='True Positive Rate (TPR)')
print("ROC curve is saved as ROC curve.png")
plt.savefig('ROC curve.png')
plt.show()