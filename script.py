import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    #Initialize empty means vector
    means = np.empty((2,0))

    for i in range(len(np.unique(y))):
        # loop through each unique val of y
        yi=(np.unique(y)[i])
        
        #filter x for indices where y = yi and calc mean
        a=np.mean(X[y[:,0]==yi],0)
        
        #append mean to mean vector
        means = np.c_[means, a]
    
    #calc covariance
    covmat = np.cov(X,rowvar=0,bias=1)
    

    
    # IMPLEMENT THIS METHOD 
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    #same as lda learn except calc cov in for loop and append to list
    means = np.empty((2,0))
    covmats = []

    for i in range(len(np.unique(y))):
        yi=(np.unique(y)[i])
        a=np.mean(X[y[:,0]==yi],0)
        means = np.c_[means, a]
    
        cov = np.cov(X[y[:,0]==yi],rowvar=0,bias=1)

        covmats.append(cov)
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    cov = covmat
    def sqdist (x,mean,cov):
        diff = x - np.transpose(mean)
        inv_Sigma = np.linalg.inv(cov)
        sqdist1 = np.dot(np.dot(np.transpose(diff),inv_Sigma),diff)
        return sqdist1
    
    
    p = np.zeros((len(Xtest),len(means[0])))
    
    for i in range(len(Xtest),):
        x = Xtest[i]
        for j in range(len(means[0])):
            mean = means[:,j]
            sqdistij = sqdist(x,mean,cov)
            p[i,j] = np.exp(-sqdistij)
            pred = np.argmax(p, axis=1)+1
    
    acc = sum(pred==ytest[:,0])/len(ytest)
    ypred = pred

    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # function to calculate mah square distance
    def sqdist (x,mean,cov):
        diff = x - np.transpose(mean)
        diffT =np.transpose(diff)
        inv_Sigma = np.linalg.inv(cov)
        sqdist1 = np.dot(np.dot(diffT,inv_Sigma),diff)
        return sqdist1


    p = np.zeros((len(Xtest),len(means[0])))
    
    for i in range(len(Xtest),):
        xi = Xtest[i]
        for j in range(len(means[0])):
            covj = covmats[j]
            meanj = means[:,j]
            sqdistij = sqdist(xi,meanj,covj)
            p[i,j] = np.exp(-sqdistij/2)/sqrt(np.linalg.det(covj))
            pred = np.argmax(p, axis=1)+1
    
    acc = sum(pred==ytest[:,0])/len(ytest)
    ypred = pred
    # IMPLEMENT THIS METHOD
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # IMPLEMENT THIS METHOD 
    w=np.dot(inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
                                                  
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1    
    d=int(np.size(X,1))                                                            
    w=np.dot(inv(lambd*np.identity(d)+np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    N=len(Xtest)
    #mse=(np.dot(np.transpose(ytest-np.dot(Xtest,w)),(ytest-np.dot(Xtest,w))))/N
    mse = (np.sum((ytest-np.dot(Xtest,w))**2))/N
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda   
    w1=w.reshape((len(X[1]),1))
                                                              
    error=np.dot(np.transpose(y-np.dot(X,w1)),(y-np.dot(X,w1)))/2 + (lambd*(np.dot(np.transpose(w1),w1)))/2
                
    obj_grad = np.sum((np.dot(X_i,w1) -y)*X_i ,0 ) +np.array(lambd*w).flatten()  
    #obj_grad = np.sum(np.matrix(np.dot(np.transpose(X_i),np.sum(np.dot(X_i,w1)-y)))+ np.matrix(lambd*w1),1)
    error_grad = np.array(obj_grad).flatten()
    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
    r=len(x)
    c=p+1
    Xd=np.zeros((r,c)) 
    for i in range(0,c):
        Xd[:,i]=x**i
# IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')
plt.xlabel('attribute 1')
plt.ylabel('attribute 2')
plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')
plt.xlabel('attribute 1')
plt.ylabel('attribute 2')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_tr = testOLERegression(w,X,y)  # training mse without intercept
mle = testOLERegression(w,Xtest,ytest)  # test mse without intercept

w_i = learnOLERegression(X_i,y)
mle_i_tr = testOLERegression(w_i,X_i,y) # training mse with intercept
mle_i = testOLERegression(w_i,Xtest_i,ytest) # test mse with intercept


print('MSE without intercept on testing data = '+str(mle))# test mse without intercept
print('MSE with intercept on testing data = '+str(mle_i)) # test mse with intercept
print('MSE without intercept on training data = '+str(mle_tr)) # training mse without intercept
print('MSE with intercept on training data = '+str(mle_i_tr))  # training mse with intercept

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.xlabel('$\lambda$')
plt.ylabel('Mean Squared Error')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
lambda_opt_3 = lambdas[np.argmin(mses3)]
print('optimal value for lambda using Ridge Regression '+str(lambda_opt_3))

plt.xlabel('$\lambda$')
plt.ylabel('Mean Squared Error')

plt.show()
print(' OLE without intercept='+str(np.linalg.norm(w))) # OLE without intercept
print('OLE with intercept='+str(np.linalg.norm(w_i)))  # OLE with intercept
print('ridge regression='+str(np.linalg.norm(w_l))) # ridge regression 
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.xlabel('$\lambda$')
plt.ylabel('Mean Squared Error')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.xlabel('$\lambda$')
plt.ylabel('Mean Squared Error')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

#Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
#lambda_opt = lambdas[np.argmin(mses3)] # 0.059999999999999998
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.xlabel('Degree of Non-linearity(p)')
plt.ylabel('Mean Squared Error')
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.xlabel('Degree of Non-linearity(p)')
plt.ylabel('Mean Squared Error')
plt.show()

