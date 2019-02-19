# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:23:16 2019

@author: pulap
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def max_pooling_function(inp,start_length,start_width,pool_length,pool_width,d):
    ans=-10000
    for i in range(pool_length):
        for j in range(pool_width):
            ans=max(ans,inp[start_length+i,start_width+j,d])
    return ans
def avg_pooling_function(inp,start_length,start_width,pool_length,pool_width,d):
    ans=0
    for i in range(pool_length):
        for j in range(pool_width):
            ans=ans+inp[start_length+i,start_width+j,d]
    ans=ans/(pool_length*pool_width)
    return ans
##
def pooling(inp,pool_length,pool_width,pooling_function):
    inp_length=np.size(inp[:,0,0])
    inp_width=np.size(inp[0,:,0])
    inp_depth=np.size(inp[0,0,:])
    temp_l=inp_length-pool_length*int(inp_length/pool_length)
    temp_w=inp_width-pool_width*int(inp_width/pool_width)
    inp_length_new=inp_length
    inp_width_new=inp_width
    if(temp_l!=0):
        pad_length=pool_length-(inp_length-pool_length*int(inp_length/pool_length))
        inp_length_new=inp_length+pad_length
    if(temp_w!=0):
        pad_width=pool_width-(inp_width-pool_width*int(inp_width/pool_width))
        inp_width_new=inp_width+pad_width

    img=np.zeros((inp_length_new,inp_width_new,inp_depth))
    img[0:inp_length,0:inp_width,:]=inp
    l=int(inp_length_new/pool_length)
    w=int(inp_width_new/pool_width)
    output=np.zeros((l,w,inp_depth))
    for i in range(inp_depth):
        for j in range(l):
            for k in range(w):
                output[j,k,i]=pooling_function(img,j*pool_length,k*pool_width,pool_length,pool_width,i)

    return output
depth=5
inp= np.zeros((100,100,depth))
inp_l=np.size(inp[:,0,0])
inp_w=np.size(inp[0,:,0])
bias=np.random.normal(0, 0.5, 1)
for i in range(inp_l):
    for j in range(inp_w):
        for k in range(depth):
            inp[i,j,k]= np.random.normal(0, 2, 1)
test_pool1=pooling(inp,3,2,avg_pooling_function)
