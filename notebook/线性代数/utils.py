import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

def loadDataSet(filename):
    X=[]
    Y=[]
    with open(filename,'rb') as f:
        for idx,line in enumerate(f):
            line = line.decode('utf-8').strip()
            if not line:
                continue
            entity = line.split()
            print(entity)
            if idx==0:
                numFea=len(entity)
                
            entity=list(map(float,entity))
            
            X.append(entity[:-1])
            Y.append([entity[-1]])
    return np.array(X),np.array(Y)

def h(theta,X):
    return np.dot(X,theta)

def J(theta,X,Y):
    m=len(X)
    return np.sum(np.dot(h(theta,X)-Y,(h(theta,X)-Y).T)/(2*m))
	
def standarize(X):
    """特征标准化处理
    
    Args:
        X 样本集
    Returns:
        标准后的样本集
    """
    m,n = X.shape
    #归一化每一个特征
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std !=0:
            X[:,j] = (features-meanVal)/std
        else:
            X[:,j] = 0
    return X

def bgd(alpha,maxloop,epsilon,X,Y):
    m,n = X.shape # m是样本数, n是特征数，也就是参数theta个数
    theta = np.zeros((n,1))
    
    count=0
    converged=False
    error=np.inf  #当前的代价函数值
    errors=[J(theta,X,Y),]
    
    thetas={}
    for i in range(n):
        thetas[i]=[theta[i,0],]#记录每一个theta j的历史更新
    
    
    while count <= maxloop:
        if(converged):
            break;
        count += 1
        
        #这里，我们的梯度计算统一了
        for j in range(n):
            deriv = np.dot(X[:,j].T,(h(theta,X)-Y)).sum() / m
            thetas[j].append(theta[j,0]-alpha*deriv) #同步更新theta，先放入thetas，下面同步更新
        
        #同步更新
        for j in range(n):
            theta[j,0] = thetas[j][-1]
        
        error = J(theta,X,Y)
        errors.append(error)  
               
        if(abs(errors[-1] - errors[-2]) < epsilon):
            converged = True
    
    return theta,errors,thetas    