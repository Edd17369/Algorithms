# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:35:18 2020

@author: faust
"""


#%%

""" DYNAMIC PROGRAMMING """

#%%

# :) mio
# :| con ayuda
# :( copiado


#%% 2.- Longest Increasing Subsequence :(

"""Given an array of random numbers. Find longest increasing subsequence (LIS) in the array."""


def lisCopy(arr): # este fur i intento creo
    n = len(arr) 
    A = [1]*n 
    for i in range (1 , n): 
        for j in range(0 , i): 
            if arr[i] > arr[j] and A[i] < A[j]+1: 
                A[i] = A[j]+1 
    return max(A)

# O(n^2)


""" O(n*log n) """

def CeilIndex(A, l, r, key): # Binary Search
	while (r - l > 1): 
		m = l + (r - l)//2
		if (A[m] >= key): 
			r = m 
		else: 
			l = m 
	return r 
def LongestIncreasingSubsequenceLength(A, size):
	tailTable = [0 for i in range(size + 1)] 
	len = 0 # always points empty slot 
	tailTable[0] = A[0] 
	len = 1
	for i in range(1, size): 
		if (A[i] < tailTable[0]): 
			tailTable[0] = A[i] 
		elif (A[i] > tailTable[len-1]):
			tailTable[len] = A[i] 
			len+= 1
		else: 
			tailTable[CeilIndex(tailTable, -1, len-1, A[i])] = A[i] 
	return len

#%% 26.- Min Cost Path :) (Es diferente en el PRACTICE de GfG)

"""Given a cost matrix cost[][] and a position (m, n) in cost[][], write a function that returns
 cost of minimum cost path to reach (m, n) from (0, 0). Each cell of the matrix represents a 
 cost to traverse through that cell. Total cost of a path to reach (m, n) is sum of all the 
 costs on that path
"""


def minCostPath2(cost,m,n):
    A = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(n+1):
        A[0][i] = sum(cost[0][j] for j in range(i+1))
    for i in range(m+1):
        A[i][0] = sum(cost[j][0] for j in range(i+1))
    for i in range(1,m+1):
        for j in range(1,n+1):
            A[i][j] = min(A[i-1][j], A[i][j-1], A[i-1][j-1]) + cost[i][j]
    return A[m][n]


#%% 15.- Coin Change :|

""" Given a value N, if we want to make change for N cents, and we have infinite supply of each
 of S = { S1, S2, .. , Sm} valued coins, how many ways can we make the change? The order of 
 coins doesn’t matter. """
 
 
def count(S,m,n):
    A = [[0 for i in range(m)] for j in range(n+1)]
    for j in range(1,n+1):
        A[j][0] = 1 if j%S[0]==0 else 0
    for i in range(m):
        if S[i]<=n:
            A[S[i]][i] = 1
    for i in range(1,m):
        for j in range(n+1):
            if S[i] > j: 
                A[j][i] += A[j][i-1]
            else:
                A[j][i] += A[j][i-1] + A[j-S[i]][i]
    return A[n][m-1]


""" Si el orden importa """
def countWays(arr, m, n):   
    count = [0 for i in range(n + 1)]       
    # base case 
    count[0] = 1 # any set has a set which sum is 0, the empty set
    # Count ways for all values up  
    # to 'n' and store the result 
    for i in range(1, n + 1): 
        for j in range(m):   
            # if i >= arr[j] then 
            # accumulate count for value 'i' as 
            # ways to form value 'i-arr[j]' 
            if i >= arr[j]: 
                count[i] += count[i - arr[j]]       
    # required number of ways  
    return count[n]                 


#%% 12.- Matrix Chain Multiplication :(

""" Given a sequence of matrices, find the most efficient way to multiply these matrices 
together. The problem is not actually to perform the multiplications, but merely to decide 
in which order to perform the multiplications. """

import sys 
# Matrix A[i] has dimension p[i-1] x p[i] 
# for i = 1..n 
def MatrixChainOrder(p, i, j): # i el indice de matriz inicial, j el indice de matriz final
  
    if i == j: 
        return 0  
    _min = sys.maxsize   
    # place parenthesis at different places between first and last matrix, 
    # recursively calculate count of multiplications for each parenthesis 
    # placement and return the minimum count 
    for k in range(i, j):   
        count = (MatrixChainOrder(p, i, k) 
                 + MatrixChainOrder(p, k + 1, j) 
                 + p[i-1] * p[k] * p[j])   
        if count < _min: 
            _min = count   
    # Return minimum count 
    return _min 


#%% 5.- Palindrom Partition :(

def minPalPartion(str): 	
	# Get the length of the string 
	n = len(str) 	
	# Create two arrays to build the solution in bottom up manner 
	# C[i][j] = Minimum number of cuts needed for palindrome 
	# partitioning of substring str[i..j] P[i][j] = true if substring str[i..j] 
	# is palindrome, else false. Note that C[i][j] is 0 if P[i][j] is true 
	C = [[0 for i in range(n)] 
			for i in range(n)] 
	P = [[False for i in range(n)] 
				for i in range(n)] 
	# different looping variables 
	j = 0
	k = 0
	L = 0
	# Every substring of length 1 is a palindrome 
	for i in range(n): 
		P[i][i] = True; 
		C[i][i] = 0; 		
	# L is substring length. Build the solution in bottom-up manner by 
	# considering all substrings of length starting from 2 to n. 
	# The loop structure is the same as Matrix Chain Multiplication problem 
	for L in range(2, n + 1): 
		# For substring of length L, set different possible starting indexes 
		for i in range(n - L + 1): 
			j = i + L - 1 # Set ending index If L is 2, then we just need to 
			# compare two characters. Else need to check two corner characters 
			# and value of P[i + 1][j-1] 
			if L == 2: 
				P[i][j] = (str[i] == str[j]) 
			else: 
				P[i][j] = ((str[i] == str[j]) and
							P[i + 1][j - 1]) 
			# IF str[i..j] is palindrome, then C[i][j] is 0 
			if P[i][j] == True: 
				C[i][j] = 0
			else: 
				# Make a cut at every possible location starting from i to j, 
				# and get the minimum cost cut. C[i][j] = 100000000
				for k in range(i, j): 
					C[i][j] = min (C[i][j], C[i][k] +
								C[k + 1][j] + 1) 									
	# Return the min cut value for complete string. i.e., str[0..n-1] 
	return C[0][n - 1] 


#%% 6.- Find the longest path in a matrix with given constraints :(

""" Given a n*n matrix where all numbers are distinct, find the maximum length path (starting
from any cell) such that all cells along the path are in increasing order with a difference 
of 1. We can move in 4 directions from a given cell (i, j), i.e., we can move to (i+1, j) 
or (i, j+1) or (i-1, j) or (i, j-1) with the condition that the adjacent cells have a
difference of 1. """


# Returns length of the longest path beginning with mat[i][j]. 
# This function mainly uses lookup table dp[n][n] 
def findLongestFromACell(i, j, mat, dp):
    n = len(mat)
	# Base case 
    if n <= i < 0 or n <= j < 0:
        return 0
	# If this subproblem is already solved 
    if dp[i][j] != -1:
        return dp[i][j]
	# To store the path lengths in all the four directions 
	x, y, z, w = -1, -1, -1, -1
	# Since all numbers are unique and in range from 1 to n * n, 
	# there is atmost one possible direction from any cell 
	if (j<n-1 and ((mat[i][j] +1) == mat[i][j + 1])): 
		x = 1 + findLongestFromACell(i, j + 1, mat, dp) 
	if (j>0 and (mat[i][j] +1 == mat[i][j-1])): 
		y = 1 + findLongestFromACell(i, j-1, mat, dp) 
	if (i>0 and (mat[i][j] +1 == mat[i-1][j])): 
		z = 1 + findLongestFromACell(i-1, j, mat, dp) 
	if (i<n-1 and (mat[i][j] +1 == mat[i + 1][j])): 
		w = 1 + findLongestFromACell(i + 1, j, mat, dp) 
	# If none of the adjacent fours is one greater we will take 1 
	# otherwise we will pick maximum from all the four directions 
	dp[i][j] = max(x, max(y, max(z, max(w, 1)))) 
	return dp[i][j] 

# Returns length of the longest path beginning with any cell 
def finLongestOverAll(mat): 
    n = len(mat)
	result = 1 # Initialize result 
	# Create a lookup table and fill all entries in it as -1 
	dp = [[-1 for i in range(n)] for i in range(n)] 
	# Compute longest path beginning from all cells 
	for i in range(n): 
		for j in range(n): 
			if (dp[i][j] == -1): 
				findLongestFromACell(i, j, mat, dp) 
			# Update result if needed 
			result = max(result, dp[i][j]); 
	return result 

# O(n^2)

#%% 7.- Subset Sum Problem

""" Given a set of non-negative integers, and a value sum, determine if there is a subset of the given set 
with sum equal to given sum. """

def isSubsetSum(set, n, s): # La misma idea que knapsack   
    # The value of subset[i][j] will be 
    # true if there is a
    # subset of set[0..j-1] with sum equal to i
    subset = [[False for i in range(s+1)] for i in range(n+1)]   
    # If s is 0, then answer is true. Empty subset
    for i in range(n+1):
        subset[i][0] = True         
    # If sum is not 0 and set is empty, 
    # then answer is false 
    for i in range(1, s+1):
         subset[0][i] = False             
    # Fill the subset table in botton up manner
    for i in range(1, n+1): 
        for j in range(1, s+1):
            if j < set[i-1]:
                subset[i][j] = subset[i-1][j]
            if j >= set[i-1]:
                subset[i][j] = (subset[i-1][j] or subset[i-1][j-set[i-1]])     
    # uncomment this code to print table 
    # for i in range(n + 1):
    # for j in range(sum + 1):
    # print (subset[i][j], end =" ")
    # print()
    return subset[n][s]

# O(n*s)

#%% 8.- Optimal Strategy for a Game

""" Consider a row of n coins of values v1 . . . vn, where n is even. We play a game 
against an opponent by alternating turns. In each turn, a player selects either the 
first or last coin from the row, removes it from the row permanently, and receives the 
value of the coin. Determine the maximum possible amount of money we can definitely win 
if we move first.

Note: The opponent is as clever as the user. """

def optimalStrategyOfGame(arr, n):       
    # Create a table to store  
    # solutions of subproblems  
    table = [[0 for i in range(n)] for i in range(n)]   
    # Fill table using above recursive  
    # formula. Note that the table is  
    # filled in diagonal fashion  
    # (similar to http://goo.gl / PQqoS), 
    # from diagonal elements to 
    # table[0][n-1] which is the result.  
    for gap in range(n): 
        for j in range(gap, n): 
            i = j - gap               
            # Here x is value of F(i + 2, j),  
            # y is F(i + 1, j-1) and z is  
            # F(i, j-2) in above recursive  
            # formula  
            x = 0
            if (i + 2) <= j: 
                x = table[i + 2][j] 
            y = 0
            if (i + 1) <= (j - 1): 
                y = table[i + 1][j - 1] 
            z = 0
            if (i <= (j - 2)): 
                z = table[i][j - 2] 
            table[i][j] = max(arr[i] + min(x, y), arr[j] + min(y, z)) 
    return table[0][n - 1] 

#%% 3.- Count number of ways to cover a distance :)

""" Given a distance, count total number of ways to cover the distance with 1, 
2 and 3 steps. """

 
def printCountRec(n):
    A = [0,1,2,4]
    if n < len(A):
        return A[n]
    else:
        for i in range(4,n+1):
            A.append(sum(A[i-3:i]))
    return A[n] 

# O(n)

#%% 1.- Longest common subsequence :'(

"""  Given two sequences, find the length of longest subsequence present in both of them. """

def lcs(X, Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in range(m+1)] 
    """Following steps build L[m+1][n+1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 

# O(n*m)

#%% 11.- Shortest common supersequence :') (mi subconsiente es un genio)

""" Given two strings str1 and str2, the task is to find the length of the shortest 
string that has both str1 and str2 as subsequences """
 
def scs(X, Y, m, n):
    A = [[0 for i in range(m+1)] for i in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            if Y[i-1] == X[j-1]:
                A[i][j] = A[i-1][j-1]
            else:
                A[i][j] = 1+min(A[i-1][j],A[i][j-1])
    return A[n-1][m-1]

# O(n*m)

#%% 16.- Word break problem :(

""" Given an input string and a dictionary of words, find out if the input string can be 
segmented into a space-separated sequence of dictionary words. """

def wordBreak(wordList, word): 
	if word == '': 
		return True
	else: 
		wordLen = len(word) 
		return any([(word[:i] in wordList) and wordBreak(wordList, word[i:]) for i in range(1, wordLen+1)]) 



#%% 3.- Edit distance :|

""" Given two strings str1 and str2 and below operations that can performed on str1. 
Find minimum number of edits (operations) required to convert ‘str1’ into ‘str2’.  
    Insert
    Remove
    Replace
All of the above operations are of equal cost. """

def ed(s1, s2):
    n, m = len(s1), len(s2)
    A = [[0 for i in range(m+1)] for j in range(n+1)]
    for i in range(n+1):
        A[i][0] = i
    for j in range(m+1):
        A[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            if s1[i-1] == s2[j-1]:
                A[i][j] = A[i-1][j-1]
            else:
                A[i][j] = 1+min(A[i-1][j-1],A[i-1][j],A[i][j-1])
    return A[n][m]

# O(n*m)
            

#%% 17.- Maximum Product Cutting :') (No use DP)

""" Given a rope of length n meters, cut the rope in different parts of integer lengths in 
a way that maximizes product of lengths of all parts. You must make at least one cut. Assume 
that the length of rope is more than 2 meters """

def mpc(n):
    k = l = 0
    if n/3 == int(n/3):
        l = int(n/3)
    elif (n-2)/3 == int((n-2)/3):
        k = 1
        l = int((n-2)/3)
    elif (n-4)/3 == int((n-4)/3):
        k = 2
        l = int((n-4)/3)
    return 2**k * 3**l

# O(1) ~ O(n)

#%% 14.- Cutting  a Road :)

""" Given a rod of length n inches and an array of prices that contains prices of all 
pieces of size smaller than n. Determine the maximum value obtainable by cutting up the 
rod and selling the pieces.  """

def cut_log(p, n):
    A = [0]*(n+1) 
    for i in range(1,n+1):
        A[i] = max(A[i-j]+p[j] for j in range(i+1)) 
    return A[n]

# O(n^2)

#%% 20.- Egg dropping puzzle :)

""" Suppose that we wish to know which stories in a 36-story building are safe to drop eggs 
from, and which will cause the eggs to break on landing. We make a few assumptions:

…..An egg that survives a fall can be used again.
…..A broken egg must be discarded.
…..The effect of a fall is the same for all eggs.
…..If an egg breaks when dropped, then it would break if dropped from a higher floor.
…..If an egg survives a fall then it would survive a shorter fall.
…..It is not ruled out that the first-floor windows break eggs, nor is it ruled out that 
    the 36th-floor do not cause an egg to break. """

def egg(n, k):
    if n == 1:
        return k
    A = [[float('inf') for i in range(n+1)] for i in range(k+1)]
    for j in range(0,3):
        for i in range(1,n+1):
            A[j][i] = j 
    for i in range(3,k+1):
        A[i][1] = i
        A[i][2] = max(1+min(j,A[i-j][2]) for j in range(i))
    for i in range(3, n+1):
        for j in range(3, k+1):
            A[j][i] = min(1+max(A[m][i-1], A[j-m-1][i]) for m in range(1,j))
    return A[k][n]

# O(n* k^2)

#%% 19.- Box Stacking Problem :')

""" You are given a set of n types of rectangular 3-D boxes, where the i^th box has 
height h(i), width w(i) and depth d(i) (all real numbers). You want to create a stack of 
boxes which is as tall as possible, but you can only stack a box on top of another box if 
the dimensions of the 2-D base of the lower box are each strictly larger than those of 
the 2-D base of the higher box. Of course, you can rotate a box so that any side functions 
as its base. It is also allowable to use multiple instances of the same type of box. """

from itertools import permutations

def maxHeight(height, width, length, n):
    tercios = []
    for i in range(n):
        tercios.append((height[i],width[i],length[i]))
    x = []
    for t in tercios:
        x += [i for i in permutations(t) if i[1] <= i[2]]
    x.sort(key=lambda val:val[1]*val[2])
    A = [0]*len(x)
    A[0] = x[0][0]
    for i in range(1,len(x)):
        try:
            A[i] = x[i][0]+max(A[j] for j in range(i) if x[j][1]<x[i][1] and x[j][2]<x[i][2])
        except:
            A[i] = x[i][0]
    return max(A)
    
# O(n^2)

#%% 25.- Fabergè Easter Eggs crush test :)

""" Write a function that takes 2 arguments - the number of eggs n and the number of trys 
m - you should calculate maximum scyscrapper height (in floors), in which it is guaranteed 
to find an exactly maximal floor from which that an egg won't crack it.

Which means,

…..You can throw an egg from a specific floor every try
…..Every egg has the same, certain durability - if they're thrown from a certain floor or
     below, they won't crack. Otherwise they crack.
…..You have n eggs and m tries
…..What is the maxmimum height, such that you can always determine which floor the target
     floor is when the target floor can be any floor between 1 to this maximum height? """

#from numpy import array

def height(n, m):
    #A = array([[0]*(n+1)]*(m+1), dtype='int64')
    A = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(1,m+1):
        A[i][1] = i
    for j in range(1,n+1):
        A[1][j] = 1
    for j in range(2,n+1):
        for i in range(2,m+1):
            A[i][j] = A[i-1][j-1]+A[i-1][j]+1
    return A[m][n] 

# O(n*m)

#%% 9.- Knapsack 

""" Given weights and values of n items, put these items in a knapsack of capacity W to get 
the maximum total value in the knapsack. In other words, given two integer arrays val[0..n-1]
and wt[0..n-1] which represent values and weights associated with n items respectively. Also
given an integer W which represents knapsack capacity, find out the items such that sum of
the weights of those items of given subset is smaller than or equal to W.  """

def knapSack(W, wt, val, n):
    A = [[0 for i in range(W+1)] for j in range(n+1)]
    for j in range(1,W+1):
        if j >= wt[0]:
            A[1][j] = val[0]
    for i in range(2,n+1):
        for j in range(1,W+1):
            if wt[i-1] > j:
                A[i][j] = A[i-1][j]
            else:
                A[i][j] = max(A[i-1][j],A[i-1][j-wt[i-1]]+val[i-1])
    #return A[n][C]
    S = []
    c = W
    for i in range(n,0,-1):
        if wt[i-1] < c and A[i-1][c-wt[i-1]]+val[i-1] >= A[i-1][c]:
            S.append(wt[i-1])
            c -= wt[i-1]
    return S

# O(n*C)

#%% 29.- Longest subsequence such that difference between adjacents is one :)

"""Given an array of n size, the task is to find the longest subsequence such that 
difference between adjacents is one.  
"""

def lsd(arr, n):
    A = [1 for i in range(n)]
    for i in range(1,n):
        for j in range(i):
            if arr[i] in (arr[j]+1, arr[j]-1) and A[i] < A[j]+1:
                A[i] = A[j]+1
    return max(A)

# O(n*2)

#%% 23.- Longest Zig-Zag Subsequence :) [Expected Time Limit 37.06sec :( ]

"""The longest Zig-Zag subsequence problem is to find length of the longest subsequence of 
given sequence such that all elements of this are alternating. If a sequence {x1, x2, .. xn}
is alternating sequence then its element satisfy one of the following relation :

  x1 < x2 > x3 < x4 > x5 < …. xn or 
  x1 > x2 < x3 > x4 < x5 > …. xn
  
Constraints: 
1 <= T <= 200
1 <= N <= 10**3
1 <= Ai <= 10**5 """

def zigZagNaive(n, arr):
    if n == 1: return 1
    A = [1 for i in range(n)]
    # Caso base
    b = None
    for i in range(1,n):
        if arr[i] < arr[i-1]:
            b = False
            break
        if arr[i] > arr[i-1]:
            b = True
            break
    B = [b for i in range(n)]
    # Recursión
    for i in range(1,n):
        for j in range(i):
            if arr[i] < arr[j] and A[i] < A[j]+1 and B[j] == True:
                A[i] = A[j]+1
                B[i] = False
            if arr[i] > arr[j] and A[i] < A[j]+1 and B[j] == False:
                A[i] = A[j]+1
                B[i] = True
    return max(A)+1

# O(n^2) # Con T=200, N=10**3 a zigZagNaive le toma 64.10sec

"""
if __name__ == '__main__': 
    t = int(input())
    for _ in range(t):
        n = int(input())
        arr = list(map(int, input().strip().split()))
        #ob = Solution()
        print(zigZag(n, arr))
"""
        
# Esto es un test under the constrains
#import numpy as np
#T = []
#N = 10**3
#for _ in range(201):
 #   T.append((np.random.randint(1,10**5,N), N))
#for i in range(201):
 #   print(zigZag(T[i][0],T[i][1]))

# Sin DP
def zigZag(seq, n): 
    def signum(n): 
        if (n != 0): 
            return 1 if n > 0 else -1
        else: 
            return 0
    if (n == 0): 
        return 0
    lastSign = 0
    # Length is initialized to 1 as that is  
    # minimum value for arbitrary sequence 
    length = 1
    for i in range(1, n): 
        Sign = signum(seq[i] - seq[i - 1])           
        # It qualifies 
        if (Sign != lastSign and Sign != 0):    
            # Updating lastSign 
            lastSign = Sign  
            length += 1  
    return length 

# O(n)

#%% 24.- Largest sum Zigzag sequence in a matrix :)

""" Given a matrix of size n x n, find sum of the Zigzag sequence with the largest sum. 
A zigzag sequence starts from the top and ends at the bottom. Two consecutive elements of 
sequence cannot belong to same column. """

def zigzagMatrix(n, matrix):
    A = [[0 for i in range(n)] for i in range(n)]
    for j in range(n):
        A[0][j] = matrix[0][j]
    for i in range(1,n):
        for j in range(n):
              A[i][j] = matrix[i][j]+max(A[i-1][k] for k in range(n) if k != j)
    return max(A[n-1][i] for i in range(n))

# O(n^3)

#%% 13.- Partition problem :| (en el caso base y en el return False)

""" Partition problem is to determine whether a given set can be partitioned into two 
subsets such that the sum of elements in both subsets is the same.  """

def partitionP(arr,n):
    s = sum(arr) // 2
    if s != (sum(arr)/2):
            return False
    A = [[True for i in range(s+1)] for j in range(n+1)]
    for i in range(s+1):
        A[0][i] = False
    for i in range(1,n+1):
        for j in range(1,s+1):
            if j >= arr[i-1]:
                A[i][j] = A[i-1][j] or A[i-1][j-arr[i-1]]
            else:
                A[i][j] = A[i-1][j]
    return A[n][s]

# O(n^2)

#%% 22.- Count number of ways to reach destination in a Maze :) (No esta en GfG)

"""Given a maze with obstacles, count number of paths to reach rightmost-bottommost cell from topmost-leftmost
cell. A cell in given maze has value -1 if it is a blockage or dead end, else 0. From a given cell, we are 
allowed to move to cells (i+1, j) and (i, j+1) only."""

def findWays(n, m, blocked_cells):
    if [1,1] in blocked_cells:
        return 0
    A = [[0 for i in range(m)] for j in range(n)]
    for i in range(m):
        if [1,i+1] not in blocked_cells:
            A[0][i] = 1
        else:
            break
    for i in range(1,n):
        if [i+1,1] not in blocked_cells:
            A[i][0] = 1
        else:
            break
    for i in range(1,m):
        for j in range(1,n):
            if [j+1,i+1] not in blocked_cells:
                A[j][i] = A[j][i-1]+A[j-1][i]
            else:
                continue
    return A[n-1][m-1] # % 1000000007

# O(n*m) (En GfG piden tiempo de O(n*m))

"""
T=1
for i in range(T):
    n, m, k= input().split()
    n = int(n); m = int(m); k = int(k);
    blocked_cells = []
    for i in range(k):
        a = list(map(int, input().split()))
        blocked_cells.append(a)"""


#%% 23.- Minimum insertions to form a palindrome :(

""" Given a string str, the task is to find the minimum number of characters to be inserted to convert 
it to palindrome """

def findMinInsertionsDP(str1, n):
 
    # Create a table of size n*n. table[i][j]
    # will store minimum number of insertions 
    # needed to convert str1[i..j] to a palindrome.
    table = [[0 for i in range(n)] 
                for i in range(n)]
    # Fill the table
    for gap in range(1, n):
        l = 0
        for h in range(gap, n):
            if str1[l] == str1[h]:
                table[l][h] = table[l + 1][h - 1]
            else:
                table[l][h] = (Min(table[l][h - 1], 
                                   table[l + 1][h]) + 1)
            l += 1 
    # Return minimum number of insertions 
    # for str1[0..n-1]
    return table[0][n - 1]

#%% 30.- Minimum number of squares whose sum equals to given number n :| [Expected Time Limit 7.31sec :( ] 

""" A number can always be represented as a sum of squares of other numbers. Note that 1 is a square and we 
can always break a number as (1*1 + 1*1 + 1*1 + …). Given a number n, find the minimum number of squares that
sum to X. """


from math import ceil, sqrt

def getMinSquaresK(n): # Funciona pero no es el mejor
    m = int(sqrt(n))
    A = [[float('inf') for i in range(n+1)] for j in range(m+1)]
    for j in range(n+1):
        A[1][j] = j
    for i in range(m+1):
        A[i][0] = 0
    for i in range(2,m+1):
        for j in range(1,n+1):
            if i**2 > j:
                A[i][j] = A[i-1][j]
            else:
                # A[i][j-i**2] en vez de A[i-1][j-i**2] ya que un elemento se pueden repetir muchas veces
                A[i][j] = min(A[i-1][j], A[i][j-i**2]+1) 
    return A[m][n]
        
def getMinSquares(n): # El mejor desempeño
    dp = [0, 1, 2, 3] 
    for i in range(4, n + 1):         
        dp.append(i) 
        for x in range(1, ceil(sqrt(i))+1):
            temp = x * x
            if temp > i:
                break
            else:
                dp[i] = min(dp[i], 1 + dp[i-temp]) 
    return dp[n]

# O(n*sqrt(n))


#%% 25.- Modify array to maximize sum of adjacent differences :(

""" Given an array, we need to modify values of this array in such a way that sum of absolute differences 
between two consecutive elements is maximized. If the value of an array element is X, then we can change it 
to either 1 or X. """

def maximumDifferenceSumNaive(arr,n):
    if n == 1:
        return arr[0]
    A = [0 for i in range(n)]
    A[1] = max(abs(1-arr[1]),abs(1-arr[0]))
    for i in range(2,n):
        A[i] = A[i-1] + max(abs(1-arr[i-1]), max(abs(arr[i]-arr[i-1]),abs(1-arr[i])))
    return A[n-1]


def maximumDifferenceSum(arr, N):       
    # Initialize dp[][] with 0 values.  
    dp = [[0, 0] for i in range(N)] 
    for i in range(N): 
        dp[i][0] = dp[i][1] = 0  
    for i in range(N - 1):           
        # for [i+1][0] (i.e. current modified  
        # value is 1), choose maximum from  
        # dp[i][0] + abs(1 - 1) = dp[i][0]  
        # and dp[i][1] + abs(1 - arr[i])  
        dp[i + 1][0] = max(dp[i][0], dp[i][1] + abs(1 - arr[i]))    
        # for [i+1][1] (i.e. current modified value  
        # is arr[i+1]), choose maximum from  
        # dp[i][0] + abs(arr[i+1] - 1) and  
        # dp[i][1] + abs(arr[i+1] - arr[i]) 
        dp[i + 1][1] = max(dp[i][0] + abs(arr[i + 1] - 1), dp[i][1] + abs(arr[i + 1] - arr[i]))   
    return max(dp[N - 1][0], dp[N - 1][1]) 

# O(n)
    
#%% 31.- Minimum Cost To Make Two Strings Identical :)

""" Given two strings X and Y, and two values costX and costY, the task is to find the minimum cost required 
to make the given two strings identical. You can delete characters from both the strings. The cost of deleting 
a character from string X is costX and from Y is costY.  """

def findMinCost(X, Y, costX, costY):
	n, m = len(X), len(Y)
	A = [[0 for i in range(m+1)] for i in range(n+1)]
	for i in range(1,m+1):
	    A[0][i] = costY*i
	for i in range(1,n+1):
	    A[i][0] = costX*i
	for i in range(1,n+1):
	    for j in range(1,m+1):
	        if X[i-1] == Y[j-1]:
	            A[i][j] = A[i-1][j-1]
	        else:
	            A[i][j] = min(A[i][j-1]+costY,A[i-1][j]+costX)
	return A[n][m]

# O(n^2)

#%% 27.- Sum of average of all subsets :( 

""" Given an array arr of N integer elements, the task is to find sum of average of all subsets of this array. """

from math import factorial as fac

def cofBin(n,k):
    return fac(n)/(fac(k)*fac(n-k))
    
def sumAverageSubset(arr,n): # Mio, no sirve
    A = [0 for i in range(n+1)]
    A[1] = arr[0]
    for i in range(2,n+1):
        A[i] = 2*A[i-1] + arr[i-1]*sum(cofBin(n,i)/(i+1) for i in range(n))
    return A[n]

# O(n^2)

def resultOfAllSubsets(arr, N): 
    result = 0.0 # Initialize result   
    # Find sum of elements 
    s = 0
    for i in range(N): 
        s += arr[i]   
    # looping once for all subset of same size 
    for n in range(1, N + 1):   
        # each element occurs nCr(N-1, n-1) times while 
        # considering subset of size n */ 
        result += (s*cofBin(N-1,n-1)) / n   
    return result 
    
# O(n)

#%% 28.- Wildcard Pattern Matching :| (Considerar los casos sobre la longitud de '*****' y los espacios vacios. No esta en GfG) 

""" Given a text and a wildcard pattern, implement wildcard pattern matching algorithm that finds if wildcard 
pattern is matched with text. The matching should cover the entire text (not partial text).
The wildcard pattern can include the characters ‘?’ and ‘*’ 

‘?’ – matches any single character 
‘*’ – Matches any sequence of characters (including the empty sequence) """

def wildcardMatch(txt, pat):
    n, m= len(txt), len(pat)
    A = [[False for i in range(m+1)] for i in range(n+1)]
    for i in range(1,m+1):
        if pat[i-1] == '*' or pat[i-1] == '?':
            A[0][i] = True
    for i in range(1,n+1):
        A[i][0] = False
    A[0][0] = True
    for i in range(1,n+1):
        for j in range(1,m+1):
            if txt[i-1] == pat[j-1] or pat[j-1] == '*' or pat[j-1] == '?':
                A[i][j] = A[i-1][j-1]
            else:
                A[i][j] = True
    return A[n][m]
    
    
#%% 21.- 3 strings

#code
def lcs3(str1,str2,str3):
    n = len(str1)
    m = len(str2)
    l = len(str3)
    A = [[[0 for i in range(n)] for j in range(m)] for k in range(l)]
    for i in range(1,n):
        for j in range(1,m):
            for k in range(1,l):
                if str1[i] == str2[j] == str3[k]:
                    A[i][j][k] = A[i-1][j-1][k-1]
                else:
                    A[i][j][k] = max(A[i][j-1][k-1],A[i-1][j][k-1],A[i-1][j-1][k])
    return A[n-1][m-1][l-1]

#%% 32.- Size of array after repeated deletion of LIS :)

""" Given an array arr[0..n-1] of the positive element. The task is to print the remaining 
elements of arr[] after repeated deletion of LIS (of size greater than 1). If there are
multiple LIS with the same length, we need to choose the LIS that ends first. """

def minimize(arr):
    x = [i for i in arr]
    while len(x) > 1:
        n = len(x)
        A = [0 for i in range(n)]
        for i in range(1,n):
            for j in range(i):
                if x[i] > x[j] and A[i] < A[j]+1:
                    A[i] = A[j] + 1
        m = A.index(max(A))
        if m > 0:
            y = [m]
            for i in range(m,-1,-1):
                a = y[-1]
                if A[i] == A[a]-1:
                    y.append(i)
            y.reverse()
            z = [x[i] for i in y]
            for i in z:
                x.remove(i)
        else:
            return x
    return x if x else [-1]
    

#%% 33.- Maximum sum increasing subsequence from a prefix and a given element after prefix is must :)

""" Given an array of n positive integers, write a program to find the maximum sum of 
increasing subsequence from prefix till ith index and also including a given kth element 
which is after i, i.e., k > i .  """

def fcn(arr,i,k):
    n = len(arr)
    A = [0 for j in range(n)]
    A[0] = arr[0]
    for l in range(1,i+1):
        for j in range(l):
            if arr[k] > arr[l] > arr[j] and A[l] < A[j]+arr[l]:
                A[l] = A[j]+arr[l]
    return max(A)+arr[k]


#%% 34 .- Longest Palindromic Substring :|

""" Given a string, find the longest substring which is palindrome. """

def fcn(st):
    n = len(st)
    A = [[False for i in range(n)] for j in range(n)]
    maxLength = 1
    for i in range(n-1):
        A[i][i] = True
        if st[i] == st[i+1]:
            A[i][i+1] = True
    A[n-1][n-1] = True
    for L in range(3,n+1):
        for i in range(n-L+1):
            j = i+L-1
            if A[i+1][j-1] and st[i] == st[j]:
                A[i][j] = True
                maxLength = L
    return maxLength

""" Given a string S, return the longest palindromic substring in S. Incase of conflict, 
return the substring which occurs first ( with the least starting index ). """     

def fcn(st):
    n = len(st)
    longPal = st[0]
    A = [[False for i in range(n)] for j in range(n)]
    for i in range(n-1):
        A[i][i] = True
        if st[i] == st[i+1]:
            A[i][i+1] = True
            if len(st[i:i+2]) > len(longPal):
                longPal = st[i:i+2]
    A[n-1][n-1] = True
    for l in range(3,n+1):
        for i in range(n-l+1):
            j = i+l-1
            if A[i+1][j-1] and st[i] == st[j]:
                A[i][j] = True
                if len(st[i:j+1]) > len(longPal):
                    longPal = st[i:j+1]
    return longPal


#%% 35.- Maximum sum alternating subsequence :(

""" Given an array, the task is to find sum of maximum sum alternating subsequence starting 
with first element. Here alternating sequence means first decreasing, then increasing, then 
decreasing, … For example 10, 5, 14, 3 is an alternating sequence.
Note that the reverse type of sequence (increasing – decreasing – increasing -…) is not 
considered alternating here. """

def fcn(a): # Incorrecto
    n = len(a)
    A = [0 for i in range(n)]
    def sgn(k):
        if k > 0:
            return 1
        elif k < 0:
            return -1
        else:
            return 0
    ls = 1
    for i in range(1,n):
        for j in range(i):
            s = sgn(a[i]-a[j])
            if s != ls and s != 0 and A[i] < A[j]+a[i]:
                A[i] = A[j]+a[i]
                ls = s
    return max(A)

def zigZag(seq, n): 
    def signum(n): 
        if (n != 0): 
            return 1 if n > 0 else -1
        else: 
            return 0
    if (n == 0): 
        return 0
    lastSign = 0
    # Length is initialized to 1 as that is  
    # minimum value for arbitrary sequence 
    length = 1
    for i in range(1, n): 
        Sign = signum(seq[i] - seq[i - 1])           
        # It qualifies 
        if (Sign != lastSign and Sign != 0):    
            # Updating lastSign 
            lastSign = Sign  
            length += 1  
    return length 
    

def maxAlternateSum(arr, n):   
    if (n == 1): 
        return arr[0]   
    # Create two empty array that 
    # store result of maximum sum 
    # of alternate sub-sequence   
    # Stores sum of decreasing and  
    # increasing sub-sequence 
    dec = [0 for i in range(n + 1)]   
    # store sum of increasing and 
    # decreasing sun-sequence 
    inc = [0 for i in range(n + 1)]   
    # As per question, first element  
    # must be part of solution. 
    dec[0] = inc[0] = arr[0]   
    flag = 0  
    # Traverse remaining elements of array 
    for i in range(1, n):       
        for j in range(i):           
            # IF current sub-sequence is decreasing the 
            # update dec[j] if needed. dec[i] by current 
            # inc[j] + arr[i] 
            if (arr[j] > arr[i]):               
                dec[i] = max(dec[i], inc[j] + arr[i])   
                # Revert the flag, if first  
                # decreasing is found 
                flag = 1  
            # If next element is greater but flag should be 1 
            # i.e. this element should be counted after the 
            # first decreasing element gets counted 
            elif (arr[j] < arr[i] and flag == 1):   
                # If current sub-sequence is  
                # increasing then update inc[i] 
                inc[i] = max(inc[i], dec[j] + arr[i])   
    # Find maximum sum in b/w inc[] and dec[] 
    result = -2147483648
    for i in range(n):       
        if (result < inc[i]): 
            result = inc[i] 
        if (result < dec[i]): 
            result = dec[i]   
    # Return maximum sum 
    # alternate sun-sequence 
    return result


#%% 36.- Longest Geometric Progression

""" Given a set of numbers, find the Length of the Longest Geometrix Progression (LLGP) in 
it. The common ratio of GP must be an integer. """


def lenOfLongestGP(sett, n):
	# Base cases
	if n < 2:
		return n
	if n == 2:
		return 2 if (sett[1] % sett[0] == 0) else 1
	# let us sort the sett first
	sett.sort()
	# An entry L[i][j] in this 
	# table stores LLGP with
	# sett[i] and sett[j] as first 
	# two elements of GP
	# and j > i.
	L = [[0 for i in range(n)] for i in range(n)]
	# Initialize result (A single 
	# element is always a GP)
	llgp = 1
	# Initialize values of last column
	for i in range(0, n-1):
		if sett[n-1] % sett[i] == 0:
			L[i][n-1] = 2
			if 2 > llgp:
				llgp = 2
		else:
			L[i][n-1] = 1
	L[n-1][n-1] = 1
	# Consider every element as second element of GP
	for j in range(n-2, 0, -1):
		# Search for i and k for j
		i = j - 1
		k = j + 1
		while i >= 0 and k <= n - 1:
			# Two cases when i, j and k don't form
			# a GP.
			if sett[i] * sett[k] < sett[j] * sett[j]:
				k += 1
			elif sett[i] * sett[k] > sett[j] * sett[j]:
				if sett[j] % sett[i] == 0:
					L[i][j] = 2
				else:
					L[i][j] = 1
				i -= 1
			# i, j and k form GP, LLGP with i and j as
			# first two elements is equal to LLGP with
			# j and k as first two elements plus 1.
			# L[j][k] must have been filled before as
			# we run the loop from right side
			else:
				if sett[j] % sett[i] == 0:
					L[i][j] = L[j][k] + 1

					# Update overall LLGP
					if L[i][j] > llgp:
						llgp = L[i][j]
				else:
					L[i][j] = 1
				# Change i and k to fill more L[i][j]
				# values for current j
				i -= 1
				k += 1
		# If the loop was stopped due to k becoming
		# more than n-1, set the remaining entries
		# in column j as 1 or 2 based on divisibility
		# of sett[j] by sett[i]
		while i >= 0:
			if sett[j] % sett[i] == 0:
				L[i][j] = 2
			else:
				L[i][j] = 1
			i -= 1
	return llgp


#%% 37.- Remove array end element to maximize the sum of product :| [La idea esta bien pero el code no XD]

""" Given an array of N positive integers. We are allowed to remove element from either of 
the two ends i.e from the left side or right side of the array. Each time we remove an element, 
score is increased by value of element * (number of element already removed + 1). The task 
is to find the maximum score that can be obtained by removing all the element. """

def fcn(arr,n):
    dp = [[0 for i in range(n)] for i in range(n)]
    for i in range(n-1):
        dp[i][i] = arr[i]
        dp[i][i+1] = max(arr[i],arr[i+1])
    dp[n-1][n-1] = arr[n-1]
    for l in range(3,n+1):
        for i in range(n-l+1):
            j = l+i-1
            dp[i][j] = max(dp[i+1][j]+(l*arr[i]), dp[i][j-1]+(l*arr[j]))
    return dp[0][n-1]
    
" I:[1, 3, 1, 5, 2], O:43"

#%% 102.- Maximum sum path in a matrix from top to bottom :)

""" We can go from each cell in row i to a diagonally higher cell in row i+1 only i.e from 
cell(i, j) to cell(i+1, j-1) and cell(i+1, j+1) only. Find the path from the top row to the 
bottom row  """

def maxSumPath(mat, n):
    A = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        A[0][i] = mat[0][i]      
    for i in range(1,n):
        A[i][0] = A[i-1][1]+mat[i][0]
        A[i][n-1] = A[i-1][n-2]+mat[i][n-1]
        for j in range(1,n-1):
            A[i][j] = max(A[i-1][j-1], A[i-1][j+1])+mat[i][j]
    return max(A[n-1])
            
    
    
#%% 103.- N meetings in one room :|

""" There is one meeting room in a firm. There are N meetings in the form of (S[i], F[i]) where
S[i] is start time of meeting i and F[i] is finish time of meeting i. What is the maximum 
number of meetings that can be accommodated in the meeting room when only one meeting can be 
held in the meeting room at a particular time? Also note start time of one chosen meeting 
can't be equal to the end time of the other chosen meeting. """

def maximumMeetings(start,end): # DP - Time complexity O(n^2)
    n = len(start)
    meets = []
    for i in range(n):
        meets.append((start[i],end[i]))
    meets.sort(key= lambda val: val[1])
    A = [1 for i in range(n)]
    for i in range(n):
        for j in range(i):
            if meets[i][0] >= meets[j][1]+1 and A[i] < A[j]+1:
                A[i] = A[j]+1
    return max(A)


def maxMeeting(l, n): # Greedy - Time complexity O(n*lon(n))
    # Initialising an arraylist
    # for storing answer
    ans = []
    # Sorting of meeting according to
    # their finish time.
    l.sort(key = lambda x: x.end)
    # Initially select first meeting
    ans.append(l[0].pos)
    # time_limit to check whether new
    # meeting can be conducted or not.
    time_limit = l[0].end
    # Check for all meeting whether it
    # can be selected or not.
    for i in range(1, n):
        if l[i].start > time_limit:
            ans.append(l[i].pos)
            time_limit = l[i].end             
    # Print final selected meetings
    return len(ans)
    
    

#%% 104.- Largest Independent Set Problem :|

""" Given a Binary Tree, find size of the Largest Independent Set(LIS) in it. A subset of all 
tree nodes is an independent set if there is no edge between any two nodes of the subset. """

memo = {None:0} # La memo afuera mejora time complexity

def lis(root):
    if root in memo: # Caso base en memo
        return memo[root]
    else: # Recurcion
        x = lis(root.left) + lis(root.right)
        y = 1
        if root.left:
            y += lis(root.left.left) + lis(root.left.right) 
        if root.right:
            y += lis(root.right.left) + lis(root.right.right)
        memo[root] = max(x,y)
    return memo[root]
        
    
#%% 105.- Find all distinct subset sums of an array :|

""" Given a set of integers, find distinct sum that can be generated from the subsets of the 
given sets. """

def DistinctSum(nums): # Space complexity O(2^n), pero es mas rapido xd
    A = {}
    A[0] = set([0, nums[0]])
    n = len(nums)
    for i in range(1,n):
        A[i] = set()
        for num in A[i-1]:
            A[i].add(num)
            A[i].add(num+nums[i])
    return sorted(A[n-1])   


def printDistSum(arr):
    n = len(arr)
    Sum = sum(arr)
    # dp[i][j] would be true if arr[0..i-1]
    # has a subset with Sum equal to j.
    dp = [[False for i in range(Sum + 1)]
                 for i in range(n + 1)]
    # There is always a subset with 0 Sum
    for i in range(n + 1):
        dp[i][0] = True
    # Fill dp[][] in bottom up manner
    for i in range(1, n + 1):
        dp[i][arr[i - 1]] = True
        for j in range(1, Sum + 1):
            # Sums that were achievable
            # without current array element
            if (dp[i - 1][j] == True):
                dp[i][j] = True
                dp[i][j + arr[i - 1]] = True 
    # Print last row elements
    output = []
    for j in range(Sum + 1):
        if (dp[n][j] == True):
            output.append(j)
    return output


#%%

def shortestSeq(S, T):
    m = len(S)
    n = len(T)
    # declaring 2D array of m + 1 rows and
    # n + 1 columns dynamically
    dp = [[0 for i in range(n + 1)]
             for j in range(m + 1)]
    # T string is empty
    for i in range(m + 1):
        dp[i][0] = 1
    # S string is empty
    for i in range(n + 1):
        dp[0][i] = float('inf')
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ch = S[i - 1]
            k = j - 1
            while k >= 0:
                if T[k] == ch:
                    break
                k -= 1
            # char not present in T
            if k == -1:
                dp[i][j] = 1
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][k] + 1)
    ans = dp[m][n]
    if ans >= MAX:
        ans = -1
    return ans


#%% 106.- Ways to write n as sum of two or more positive integers :(

""" For a given number n > 0, find the number of different ways in which n can be written as a 
sum of at two or more positive integers. """

def countWays(n):
    mod = float('inf')
    #dp[i] will be storing the number of solutions for value i.
    #We need n+1 rows as the table is consturcted in bottom up
    #manner using the base case (n = 0).
    ##initializing all dp values as 0.
    dp = [0 for i in range(n+1)]
    #base case
    dp[0]=1 
    #picking all integer one by one and updating the dp[] values
    #from index j to the index less than or equal to n.
    for i in range(1,n):
        for j in range(i,n+1):
            dp[j] = (dp[j]%mod + dp[j-i]%mod)%mod
    #returning the result.
    return dp[n] 










