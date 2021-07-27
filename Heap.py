# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:22:58 2021

@author: faust
"""


from heapq import heappush, heappop, heapify, nlargest, nsmallest, heappushpop

def insertHeap(H, v):
    H.append(v)
    i = len(H)-1
    while H[i//2] > H[i]:
        H[i//2], H[i] = H[i], H[i//2]
        i = i//2
    return H

# Function to heapify the node at pos i
def bubbleD(h,i):
    n = len(h)-1
    if 2*i+1 > n:
        pass
    elif 2*i+2 > n:
        if h[i] > h[2*i+1]:
            h[2*i+1], h[i] = h[i], h[2*i+1]
            bubbleD(h,2*i+1)
    else:
        if h[2*i+1] < h[2*i+2] and h[2*i+1] < h[i]:
            h[2*i+1], h[i] = h[i], h[2*i+1]
            bubbleD(h,2*i+1)
        elif h[2*i+1] > h[2*i+2] and h[2*i+2] < h[i]:
            h[2*i+2], h[i] = h[i], h[2*i+2]
            bubbleD(h,2*i+2)
        
def extMin(h):
    n = len(h)
    print(h[0])
    h[n-1],h[0] = h[0],h[n-1]
    h.pop(n-1)
    bubbleD(h,0)
    

#%% 1.- k largest(or smallest) elements in an array :(

""" Write an efficient program for printing k largest elements in an array. Elements in array 
can be in any order. For example, if given array is [1, 23, 12, 9, 30, 2, 50] and you are 
asked for the largest 3 elements i.e., k = 3 then your program should print 50, 30 and 23. """


def fcn(h,n,k): # lo mismo que la funcion nlargest XD
    x = h[:k]
    heapify(x)
    for i in range(k,n):
        if h[i] > x[0]:
            x.pop(0) 
            heappush(x,h[i])
    return x
    
    
# O((n-k)*log(k)) El orden de sorted(h,reverse=True) return h[:k] es n*log(n)+k
    
    
#%% 2.- Connect n ropes with minimum cost :(

""" There are given n ropes of different lengths, we need to connect these ropes into one 
rope. The cost to connect two ropes is equal to the sum of their lengths. We need to connect 
the ropes with minimum cost. """

import heapq
    
def minCRopes(H):
    ans = 0
    heapq.heapify(H)
    while len(H) != 1:
        x = heapq.heappop(H)
        y = heapq.heappop(H)
        z = x+y
        ans += z
        heapq.heappush(H, z)
    return ans
    

#%% 3.- K maximum sum combinations from two arrays :|

""" Given two equally sized arrays (A, B) and N (size of both arrays). A sum combination is 
made by adding one element from array A and another element of array B. Display the maximum 
K valid sum combinations from all the possible sum combinations """

def sumCombination(a, b, n, k):
    H = a[:k]
    for i in range(n):
        for j in range(n):
            x = a[i]+b[j]
            if x > H[0]:
                heappop(H)
                heappush(H, x)
    return H

def sumCombination2(a, b, n, k): # regresa los valores con sus pares (se puede mejorar)
    A = []
    for i in range(n):
        for j in range(n):
            x = a[i]+b[j]
            A.append((x,a[i],b[j]))
    heapify(A)
    H = A[:k]
    for i in range(k,n*n):
        if A[i][0] > H[0][0]:
            heappop(H)
            heappush(H, A[i])
    return H
    

#%% 4.- Maximum distinct elements after removing k elements


def distElem(arr,n, k):
    H = {} # almacena los valores con su frecuencia
    # separa los valores con frecuencia mayor a 1 en un heap
    m = k
    for i in range(n):
        if arr[i] not in H:
            H[arr[i]] = 1
        else:            
            H[arr[i]] += 1
            m -= 1
            
#%% 19.- Merge Sort Tree for Range Order Statistics

""" Find the Kth smallest number in the range from array index 'start' to 'end'. """

#from heapq import heapify, heappop

def kthSmallest(arr, start, end, k):
    heapify(arr[start:end+1])
    for i in range(k):
        x = heappop(arr)
    return x

#%% 25.- Maximum difference between two subsets of m elements

""" Given an array of n integers and a number m, find the maximum possible difference between 
two sets of m elements chosen from given array. """

#from heapq import nlargest, nsmallest

def maxDiference(arr, m):
    heapify(arr)
    x = sum(nlargest(m, arr))
    y = sum(nsmallest(m, arr))
    return x-y
    

# Este funciona mejor
def find_difference(arr, n, m):
    max = 0; min = 0
    arr.sort();
    j = n-1
    for i in range(m):
        min += arr[i]
        max += arr[j]
        j = j - 1 
    return (max - min)



    
    