'''
Created on 17.02.2015

@author: serg
'''
from cmath import sqrt
from sympy.core.symbol import _range
import numpy as np
from numpy import matrix
from win32con import NUMCOLORS
import matplotlib.pyplot as plt
from nltk.featstruct import FeatureValueUnion
import os,sys

import Image
import os

import matplotlib.pyplot as plt


def _calcMean(a):
    '''
    Calculate the mean of the float array
    '''
     
    _mean=float(0.0)
    for i in a:
        _mean+=i
        
    return _mean/len(a)    



def _calcVariance(a):
    '''
    Calculate the Variance of the float array
    '''
    
    mean=_calcMean(a)
    _variance=float(0.0)
    for i in a:
        _variance+= (i-mean)*(i-mean)   
    _variance=_variance/(len(a)-1)
    return _variance



def _calcStandard_Deviation(a):
    '''
    Calculate the Standard Deviation of the float array
    '''
    
    return sqrt(_calcVariance(a))    
     
 
 
def _calcCovariance(a,b):
    '''
    Calculate the Covariance of two float arrays
    '''
    meanA=_calcMean(a)
    meanB=_calcMean(b)
    _covariance=float(0.0)
    _sizeAB=len(a)
    for i in range(_sizeAB):
        _covariance+= (a[i]-meanA)*(b[i]-meanB)   
    _covariance=_covariance/(_sizeAB-1)
    return _covariance  
def _calcCovarianceMatrix(datMat):
    '''
    Calculate the Covariance Matric of two float arrays
    '''
    numRows=len(datMat)  
    covMat=[[0 for x in range(numRows)] for x in range(numRows)] 
    for i in range(numRows):
        a=datMat[i]
        for j in range(numRows):            
            b=datMat[j]
            covMat[i][j]=_calcCovariance(a, b)
    return covMat
              
def _subtrTheMean(datMat):
    '''
    Subtracts the Mean of the data
    '''
    numRows=len(datMat) 
    numCols=len(datMat[0]) 
    for i in range(numRows):
        mean=_calcMean(datMat[i])
        for j in range(numCols):
            datMat[i][j]=datMat[i][j]-mean;
            
def _selecFeatureVector(eigVecMat,N):
    '''
    Subtracts the Mean of the data (eigVecMat is the row matrix of eigenVectors, N is the number of the first features)
    '''
    
    if N>len(eigVecMat):
        print("Too much number for feature selection (N>>number of eigenVectors)")
        return
    else:
             
  
        numCols=len(eigVecMat[0]) 
        FeatureVector= [[0 for x in range(N)] for x in range(numCols)]  
        for i in range(N):
            for j in range(numCols):
                FeatureVector[j][i]=eigVecMat[i][numCols-1-j]
                
        return FeatureVector   
def getRowImageMatrix(path):
    '''
    Making image Row Image( Image1 (NxN to N^2)
                            Image2
                            Image3
                            ...
                            ImageN)
    '''
    rowImageMatrix=[]
    for files in os.walk(path):
        imCursor=0
        rowImageMatrix=[[0 for y in range(1024) ] for x in range(10)] #str
                 
        for f in files[2]:
                 
                    jpIm=Image.open("Images\\"+f)
                    imarr=np.array(jpIm)
                    
                    
                    
                    #going throw the files, reading each files (photos) each pixel(only 1 component as the image is in greyscale) and saving it in a row image Matric
                    index=0
                    for i in range( len (imarr) ):
                        for j in range( len (imarr[0]) ):
                            rowImageMatrix[imCursor][index]=imarr [i] [j] [0]
                                                      
                            index+=1
                  
                    imCursor+=1
                    if imCursor==10:
                        return rowImageMatrix
                 
    return rowImageMatrix
def doPCA(mat,N):
    '''
    realization of PCA algorithm
    '''
    _subtrTheMean(mat)
  
    cov=_calcCovarianceMatrix(mat)
    print("SAD")
    eigValue,eigVector=np.linalg.eig(cov)
    FeatureVector=_selecFeatureVector(eigVector,N)
    RowFeatureVector=np.transpose(FeatureVector)
    transformedData=np.dot(RowFeatureVector,mat)
    
    return transformedData

    
def toArray(transformedData,w,h):
    '''
    As we have Row Image Data, and eacht pixel is represented with 1 Value(because Greyscale) we need to tranform to usual Image Matrix
    Matr
    '''   
    
    vecSize=len(transformedData)
    imarr= [ [ [ [0 for color in range(3)] for z in range(w)]  for y in range(h)]  for x in range(vecSize)]  
    for ind in range(vecSize):
        indexofRowImage=0
        for i in range(h):
            for j in range(w):
                imarr[ind][i][j][0]=transformedData[ind][indexofRowImage]
                imarr[ind][i][j][1]=transformedData[ind][indexofRowImage]
                imarr[ind][i][j][2]=transformedData[ind][indexofRowImage]
                indexofRowImage+=1       
        
        plt.imshow(imarr[ind]) #Needs to be in row,col order
        plt.savefig("training\\eigFace"+str(ind))
                
                
            
                  
                    
       

'''  
          
mat = [[0 for x in range(10)] for x in range(2)] 
            
a=[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]
b=[2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]


mat[0]=a
mat[1]=b

_subtrTheMean(mat)
cov=_calcCovarianceMatric(mat)




FeatureVector=_selecFeatureVector(eigVector,1)
RowFeatureVector=np.transpose(FeatureVector)

transformedData=np.dot(RowFeatureVector,mat)
'''


imRowMat =getRowImageMatrix("Images")
Mat=doPCA(imRowMat, 3)
toArray(Mat,32,32)

       
