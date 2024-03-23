import numpy as np
import scipy.stats as stats 
from numba import jit 

#%% Gaussian mixture 

#List of location 
moy_list=[np.array([0,0]),np.array([1,1]),np.array([3,2])]

#List of covariance matrix (only non degenerate)
matrix1=np.matrix([[1,0],[0,1]])
matrix2=np.matrix([[2,0.5],[0.5,2]])
matrix3=matrix1
cov_list=[matrix1,matrix2,matrix3]
     
#Probability density function 
def gausmix(x,moy=moy_list,cov=cov_list):
    nb_gaussian=len(moy)
    toreturn=0
    for i in range(nb_gaussian):
        gaussian=stats.multivariate_normal.pdf(x,mean=moy[i],cov=cov[i])
        toreturn+=gaussian
    return toreturn/nb_gaussian 

#Marginals probabilities 

def marg1(x,moy=moy_list,cov=cov_list):
    nb_gaussian=len(moy)
    toreturn=0
    for i in range(nb_gaussian):
        gaussian=stats.norm.pdf(x,loc=moy[i][0],scale=cov[i][0,0])
        toreturn+=gaussian
    return toreturn/nb_gaussian

def marg2(x,moy=moy_list,cov=cov_list):
    nb_gaussian=len(moy)
    toreturn=0
    for i in range(nb_gaussian):
        gaussian=stats.norm.pdf(x,loc=moy[i][1],scale=cov[i][1,1])
        toreturn+=gaussian
    return toreturn/nb_gaussian

#Conditional probability 

def proba1(x,y,moy=moy_list,cov=cov_list):
    X=np.array([x,y])
    return gausmix(X,moy,cov)/marg2(y,moy,cov)

def proba2(y,x,moy=moy_list,cov=cov_list):
    X=np.array([x,y])
    return gausmix(X,moy,cov)/marg1(x,moy,cov)

#Computing the Gradient

def df1(x,moy,cov,det):
    return -1/det*(cov[0,0]*(x[0]-moy[0])-cov[0,1]*(x[1]-moy[1]))*stats.multivariate_normal.pdf(x,mean=moy,cov=cov)
def df2(x,moy,cov,det):
    return -1/det*(cov[0,0]*(x[1]-moy[1])-cov[0,1]*(x[0]-moy[0]))*stats.multivariate_normal.pdf(x,mean=moy,cov=cov)

def delta_log_gauss(x,moy=moy_list,cov=cov_list):
    nb_gaussian=len(moy)
    toreturn=np.zeros_like(x)
    for i in range(nb_gaussian):
        det=np.linalg.det(cov[i])
        toreturn[0]+=df1(x,moy[i],cov[i],det)
        toreturn[1]+=df2(x,moy[i],cov[i],det)
    return toreturn/(3*gausmix(x,moy,cov))

#Potential energy

def Ugaussmix(q,moy=moy_list,cov=cov_list):
    return -np.log(gausmix(q,moy,cov))

#Gradient 

def delta_Ugaussmix(x,moy=moy_list,cov=cov_list):
    return -delta_log_gauss(x,moy,cov)


#%% Ring-shapped distribution

center = np.array([0, 0])  # Change this to the desired center of the ring
radius = 3


def ring(x, center=center, radius=radius):
    distance = np.linalg.norm(x - center, axis=-1)
    ring_distribution = np.exp(-1/2 * ((distance - radius)) ** 2)
    return ring_distribution

def delta_log_ring(x, center=center, radius=radius):
    distance = np.linalg.norm(x - center, axis=-1)
    toreturn=np.zeros_like(x)
    if distance>0:
        toreturn[0]=-(distance-radius)*(x[0]-center[0])/distance
        toreturn[1]=-(distance-radius)*(x[1]-center[1])/distance
    return toreturn


#Potential energy

def Uring(x, center=center, radius=radius):
    return -np.log(ring(x, center=center, radius=radius))

#Gradient 

def delta_Uring(x, center=center, radius=radius):
    return -delta_log_ring(x,center,radius)