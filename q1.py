# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:29:11 2019

@author: pulap
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def activation_function(x):#RELU function
    if(x<0):
        return 0
    else:
        return x


        
    
def correlation(img,l,w,kernel):
    length=np.size(kernel[:,0,0])
    width=np.size(kernel[0,:,0])
    depth=np.size(kernel[0,0,:])
    sum=0
    for i in range(depth):
        for j in range(length):
            for k in range(width):
                sum=sum+(img[l+j,w+k,i]*kernel[j,k,i])
    return sum
    
def convolution(img,kernel,padding,stride,bias):
    img_width=np.size(img[0,:,0])
    img_length=np.size(img[:,0,0])
    img_depth=np.size(img[0,0,:])
    
    k_l=np.size(kernel[:,0,0])
    k_w=np.size(kernel[0,:,0])
    k_d=np.size(kernel[0,0,:])
    if(padding=="valid"):
        
        l=int((img_length-k_l)/stride)+1
        w=int((img_width-k_w)/stride)+1
        
        pad_length=k_l-(img_length-((l-1)*stride+k_l))
        pad_width=k_w-(img_width-((w-1)*stride+k_w))
        
        img_padded=np.zeros((img_length+pad_length,img_width+pad_width,img_depth));
        img_padded[:-pad_length,:-pad_width,:]=img
        
        
        img_padded_width=np.size(img_padded[0,:,0])
        img_padded_length=np.size(img_padded[:,0,0])
        img_padded_depth=np.size(img_padded[0,0,:]) 
        
        
        l=int((img_padded_length-k_l)/stride)+1
        w=int((img_padded_width-k_w)/stride)+1
        output=np.zeros((l,w))
        for i in range(l):
            for j in range(w):
                output[i,j]=activation_function(correlation(img_padded,i*stride,j*stride,kernel)+bias)
                
    if(padding=="same"):
        required_length=(img_length-1)*stride+k_l
        required_width=(img_width-1)*stride+k_w
        required_padd_length=required_length-img_length
        required_padd_width=required_width-img_width
        r_l=int(required_padd_length/2)
        r_w=int(required_padd_width/2)
        img_padded=np.zeros((required_length,required_width,img_depth))
        img_padded[r_l:r_l+img_length,r_w:r_w+img_width,:]=img
     
        img_padded_width=np.size(img_padded[0,:,0])
        img_padded_length=np.size(img_padded[:,0,0])
        img_padded_depth=np.size(img_padded[0,0,:]) 
        
        l=int((img_padded_length-k_l)/stride)+1
        w=int((img_padded_width-k_w)/stride)+1
        output=np.zeros((l,w))
        for i in range(l):
            for j in range(w):
                output[i,j]=activation_function(correlation(img_padded,i*stride,j*stride,kernel)+bias) 
                
                
    return output


img = cv2.imread('image1.jpg')
#plt.imshow(img)

depth=np.size(img[0,0,:]);
kernel= np.zeros((10,10,depth))
k_l=np.size(kernel[:,0,0])
k_w=np.size(kernel[0,:,0])
bias=np.random.normal(0, 0.5, 1)
for i in range(k_l):
    for j in range(k_w):
        for k in range(depth):
            kernel[i,j,k]= np.random.normal(0, 2, 1)
            
        
test=convolution(img,kernel,"valid",5,bias)
#test_pool=pooling(kernel,2,2)
plt.imshow(test, cmap = 'gray')
plt.show()
                  
                    
            
        
