# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:15:32 2019

@author: pulap
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def activation_function(x):
    if(x<0):
        return 0
    else:
        return x
##
def correlation(img,l,w,kernel,d):
    length=np.size(kernel[:,0,0,d])
    width=np.size(kernel[0,:,0,d])
    depth=np.size(kernel[0,0,:,d])
    sum=0
    for i in range(depth):
        for j in range(length):
            for k in range(width):
                sum=sum+(img[l+j,w+k,i]*kernel[j,k,i,d])
    return sum

def convolution(img,kernel,d,padding,stride,bias):
    img_width=np.size(img[0,:,0])
    img_length=np.size(img[:,0,0])
    img_depth=np.size(img[0,0,:])

    k_l=np.size(kernel[:,0,0,d])
    k_w=np.size(kernel[0,:,0,d])
    k_d=np.size(kernel[0,0,:,d])
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
                 output[i,j]=activation_function(correlation(img_padded,i*stride,j*stride,kernel,d)+bias[d,0])


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
                output[i,j]=activation_function(correlation(img_padded,i*stride,j*stride,kernel,d)+bias[d,0])


    return output

def convolution_util(img,kernels,stride,padding,activation_function):
    depth=np.size(kernels[0,0,0,:]);
    bias1=np.zeros((depth,1))

    for i in range(depth):
        if(i==0):
             out=convolution(img,kernels,i,"valid",stride,bias1)
             n=np.size(out[:,0])
             m=np.size(out[0,:])
             output=np.zeros((n,m,depth))
             output[:,:,i]=out

        output[:,:,i]=convolution(img,kernels,i,"valid",stride,bias1)
    return output


#plt.imshow(img)
img = cv2.imread('image1.jpg')
#plt.imshow(img)


depth=np.size(img[0,0,:]);
kernel= np.zeros((10,10,depth,3))
k_l=np.size(kernel[:,0,0,0])
k_w=np.size(kernel[0,:,0,0])
number_filter=np.size(kernel[0,0,0,:])

for i in range(k_l):
    for j in range(k_w):
        for k in range(depth):
            for l in range(number_filter):
                kernel[i,j,k,l]= np.random.normal(0, 1, 1)



test=convolution_util(img,kernel,5,"valid",activation_function)

plt.imshow(test[:,:,2], cmap = 'gray')
plt.show()
