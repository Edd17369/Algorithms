# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:33:39 2021

@author: faust
"""


#%%

""" HASHING """

#%%

# :) mio
# :| con ayuda
# :( copiado


#%% 1.- Find whether an array is subset of another array

""" Given two arrays: arr1[0..m-1] and arr2[0..n-1]. Find whether arr2[] is a subset of arr1[] or not. """

def isSubset(arr1, m, arr2, n):       
    # Using STL set for hashing 
    hashset = set()   
    # hset stores all the values of arr1 
    for i in range(0, m): 
        hashset.add(arr1[i])   
    # Loop to check if all elements 
    # of arr2 also lies in arr1 
    for i in range(0, n): 
        if arr2[i] in hashset: 
            continue
        else: 
            return False  
    return True

#%% 3.-Given an array A and a number x, check for pair in A with sum as x 

def printPairs(arr, arr_size, x):
    s = set()     
    for i in range(0, arr_size):
        temp = x-arr[i]
        if (temp in s):
            return (arr[i], temp)
        s.add(arr[i])
    return False

# O(n)

#%% 5.- Find four elements a, b, c and d in an array such that a+b = c+d :|

""" Given an array of distinct integers, find if there are two pairs (a, b) and (c, d) 
such that a+b = c+d, and a, b, c and d are distinct elements. If there are multiple answers,
 then print any of them. """

def findPairs(arr, n): 
    A = {}
    for i in range(n):
        for j in range(i+1,n):
            a = arr[i]+arr[j]
            if a in A:
                b = A.get(a)
                return ((arr[i], arr[j]),b)
            else:
                A[a] = (arr[i],arr[j])
    return False

# O(n^2)

#%% 6.- Find the length of largest subarray with 0 sum :(

""" Given an array of integers, find the length of the longest sub-array with sum equals to 0. """

def maxLen(arr):       
    # NOTE: Dictonary in python was implemented as Hash Maps 
    # Create an empty hash map (dictionary) 
    hash_map = {}   
    # Initialize result 
    max_len = 0  
    # Initialize sum of elements 
    curr_sum = 0  
    # Traverse through the given array 
    for i in range(len(arr)):           
        # Add the current element to the sum 
        curr_sum += arr[i]   
        if arr[i] == 0 and max_len == 0: 
            max_len = 1  
        if curr_sum == 0: 
            max_len = i + 1  
        # NOTE: 'in' operation in dictionary to search  
        # key takes O(1). Look if current sum is seen  
        # before 
        if curr_sum in hash_map: 
            max_len = max(max_len, i - hash_map[curr_sum] ) 
        else:   
            # else put this sum in dictionary 
            hash_map[curr_sum] = i   
    return max_len 

# O(n)

#%% 7.- Count distinct elements in every window of size  :(

""" Given an array of size n and an integer k, return the count of distinct numbers in all windows of size k. """
    
def countDistinctNaive(arr, n, k):
    A = []
    for i in range(n-k+1):
        A.append(len(set(arr[i:i+k])))
    return A

# O((n-k)*k) ?

from collections import defaultdict 
def countDistinct(arr, k, n): 
    mp = defaultdict(lambda :0)
	# initialize distinct element 
	# count for current window 
    dist_count = 0
	# Traverse the first window and store count 
	# of every element in hash map 
    for i in range(k):
        if mp[arr[i]] == 0: # Si aun no se ha registrado
            dist_count += 1
        mp[arr[i]] += 1 # Si ya hay registro
	# Traverse through the remaining array 
    for i in range(k,n):
		# Remove first element of previous window 
		# If there was only one occurrence, 
		# then reduce distinct count.
        if mp[arr[i-k]] == 1:
          dist_count -= 1
        mp[arr[i]] -= 1
        # Add new element of current window 
    	# If this element appears first time, 
    	# increment distinct element count 
        if mp[arr[i]] == 0:
            dist_count += 1
        mp[arr[i]] += 1 
    return dist_count

# O(n)

#%% 8.- Find smallest range containing elements from k lists :(

"""Given k sorted lists of integers of size n each, find the smallest range that includes at least 
element from each of the k lists. If more than one smallest ranges are found, print any one of them."""


def findSmallest(arr):
    minRange = float('inf')
    res = [0,0]
    k = len(arr)
    while len(arr[0]) > 0:
        arr.sort()
        currRange = arr[k-1][0] - arr[0][0]
        if currRange < minRange:
            minRange = currRange
            res = [arr[0][0] ,arr[k-1][0]]
            arr[0].pop(0)
    return res

#%%  22.- Count divisible pairs in an array :(

""" Given an array, whiout repetitions, count pairs in the array such that one element of pair divides other. """

def divPairsNaive(arr):
    arr.sort()
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i+1,n):
            if not arr[j] % arr[i]:
                count += 1
    return count

def countDivisibles(arr): # el peor de todos 
    n = len(arr)
    res = 0 
    # Iterate through all pairs  
    for i in range(0, n) : 
        for j in range(i+1, n) :              
            # Increment count if one divides  
            # other  
            if (arr[i] % arr[j] == 0 or
            arr[j] % arr[i] == 0) : 
                res+=1 
    return res  

def divPairsHash(arr): # es el mejor si los int in arr no son muy grandes en comparacion a len(arr)
    n = len(arr)
    A = arr
    m = max(A)
    count = 0
    for i in range(n):
        k =  m // arr[i]
        for j in range(2,k+1):
            if arr[i] * j in A:
                count += 1
    return count

#%% 17.- Longest Consecutive Subsequence :(

""" Given an array of integers, find the length of the longest sub-sequence such that elements in the 
subsequence are consecutive integers, the consecutive numbers can be in any order.  """

def findLongestConseqSubseq(arr, n): 
    s = set() 
    ans = 0
    # Hash all the array elements 
    for ele in arr: 
        s.add(ele)   
    # check each possible sequence from the start 
    # then update optimal length 
    for i in range(n):   
        # if current element is the starting 
        # element of a sequence 
        if (arr[i]-1) not in s:   
            # Then check for next elements in the 
            # sequence 
            j = arr[i] 
            while(j in s): 
                j += 1  
            # update  optimal length if this length 
            # is more 
            ans = max(ans, j-arr[i]) 
    return ans 

#%% 18.- Print all subarrays with 0 sum  :| (hay que corregir  el viernes)

""" Given an array, print all subarrays in the array which has sum 0. """

#from collections import defaultdict
def arraySum0(arr):
    A = defaultdict(lambda:[])
    s = 0
    for i in range(len(arr)):
        s += arr[i]
        if arr[i] == 0:
            print('Subarray found from %s to %s' %(i,i))
        if s == 0:
            print('Subarray found from 0 to %s' %(i))
        if s in A:
            for x in A[s]:
                print('Subarray found from %s to %s' %(x,i))
            A[s] += [i+1]
        else:
            A[s] = [i+1]

"""
You are given an array A[] of size N. Find the total count of sub-arrays having their sum equal to 0.


Example 1:

Input:
N = 6
A[] = {0,0,5,5,0,0}
Output: 6
Explanation: The 6 subarrays are 
[0], [0], [0], [0], [0,0], and [0,0].


Example 2:
I: arr = [10, 14, 15, 10, -19, 2, 14, -12, -9, 10, 7, -16, 19, -1, 11, 8, -7, -20, 2, -2, -15, -8, 6, -15, 3, 8, -5, 12, 16, 11]
O: 4
"""
#from collections import defaultdict
def findSubArrays(arr,n):
    A = defaultdict(lambda:[])
    s = 0
    count = 0
    for i in range(n):
        s += arr[i]
        if arr[i] == 0:
            count += 1
            #print('Subarray found from %s to %s' %(i,i))
        if s == 0:
            count += 1
            #print('Subarray found from 0 to %s' %(i))
        if s in A:
            for x in A[s]:
                count += 1
                #print('Subarray found from %s to %s' %(x,i))
            A[s] += [i+1]
        else:
            A[s] = [i+1]
    return count    

#%% 12.- Check if an array can be divided into pairs whose sum is divisible by k :(

"""Given an array of integers and a number k, write a function that returns true if given array can be
divided into pairs such that sum of every pair is divisible by k."""

#from collections import defaultdict

def canPairs(arr, n, k):	
	# An odd length array cannot 
	# be divided into pairs
	if (n & 1):
		return 0		
	# Create a frequency array to
	# count occurrences of all 
	# remainders when divided by k.
	freq = defaultdict(lambda : 0)
	# Count occurrences of all remainders
	for i in range(0, n):
		freq[((arr[i] % k) + k) % k] += 1
	# Traverse input array and use 
	# freq[] to decide if given array 
	# can be divided in pairs
	for i in range(0, n):		
		# Remainder of current element
		rem = ((arr[i] % k) + k) % k
		# If remainder with current element 
		# divides k into two halves.
		if (2 * rem == k):			
			# Then there must be even occurrences
			# of such remainder
			if (freq[rem] % 2 != 0):
				return 0
		# If remainder is 0, then there
		# must be two elements with 0 remainde
		elif (rem == 0):
			if (freq[rem] & 1):
				return 0
			# Else number of occurrences of 
			# remainder must be equal to 
			# number of occurrences of
			# k - remainder
			elif (freq[rem] != freq[k - rem]):
				return 0				
	return 1

#%% 21.- Given an array of pairs, find all symmetric pairs in it :(

""" Two pairs (a, b) and (c, d) are said to be symmetric if c is equal to b and a is equal to d. For 
example, (10, 20) and (20, 10) are symmetric. Given an array of pairs find all symmetric pairs in it.
It may be assumed that the first elements of all pairs are distinct. """

def findSymPairs(arr, row):   
    # Creates an empty hashMap hM 
    hM = dict()   
    # Traverse through the given array 
    for i in range(row):           
        # First and second elements  
        # of current pair 
        first = arr[i][0] 
        sec = arr[i][1]   
        # If found and value in hash matches with first 
        # element of this pair, we found symmetry 
        if (sec in hM.keys() and hM[sec] == first): 
            print("(", sec,",", first, ")")   
        else: # Else put sec element of 
              # this pair in hash 
            hM[first] = sec 
            
#%% 23.- Minimum insertions to form a palindrome with permutations allowed :)

""" Given a string of lowercase letters. Find minimum characters to be inserted in string so that it can 
become palindrome. We can change positions of characters in string. """

#from collections import defaultdict

def makePalindrome(s):
    H = defaultdict(lambda:0)
    for i in range(len(s)):
        H[s[i]] += 1
    ins = 0
    for k in H:
        if H[k] & 1:
            ins += 1
    return ins-1

#%% 24.- Find the largest d in array such thay a+b+c=d :|

""" Given a set S (all distinct elements) of integers, find the largest d such that a + b + c = d
where a, b, c, and d are distinct elements of S. """

def largetD(arr,n):
    H = {}
    d = min(arr)
    for i in range(n):
        for j in range(i+1,n):
            H[arr[i]+arr[j]] = (arr[i],arr[j])
    for i in range(n):
        for j in range(i+1,n):
            abs_diff = abs(arr[i] - arr[j]) 
            #if arr[i]-arr[j] in H and arr[i] > d:
            #    d = arr[i]
            if abs_diff in H.keys(): 
                p = H[abs_diff] 
                if (p[0] != i and p[0] != j and p[1] != i and p[1] != j): 
                    d = max(d, max(arr[i], arr[j])) 
    return d

#%% 25.- Find all triplets with zero sum :(

""" Given an array of distinct elements. The task is to find triplets in the array whose sum is zero. """

def findTripletsNaive(arr,n): # Time execution out
    H = {}
    for i in range(n):
        for j in range(i+1,n):
            H[arr[i]+arr[j]] = (i,j)
    s = set()
    for i in range(n):
        if -arr[i] in H:
            b,c = H.get(-arr[i])
            if len({arr[i], arr[b], arr[c]}) == 3:
                s.add(arr[i])
                s.add(arr[b])
                s.add(arr[c])
    return 1 if s else 0

def findTriplets(arr, n): 
    found = False
    for i in range(n - 1): 
        # Find all pairs with sum  
        # equals to "-arr[i]"  
        s = set() 
        for j in range(i + 1, n): 
            x = -(arr[i] + arr[j]) 
            if x in s: 
                print(x, arr[i], arr[j]) 
                found = True
            else: 
                s.add(arr[j]) 
    if found == False: 
        print("No Triplet Found") 
        

#%% 26.- Count Substrings with equal number of 0s, 1s and 2s :(

""" Given a string which consists of only 0s, 1s or 2s, count the number of substrings that have equal 
number of 0s, 1s and 2s. """
 
def getSubstringWithEqual012(string): 
    n = len(string)   
    # map to store, how many times a difference 
    # pair has occurred previously 
    mp = dict() 
    mp[(0, 0)] = 1  
    # zc (Count of zeroes), oc(Count of 1s) 
    # and tc(count of twos) 
    # In starting all counts are zero 
    zc, oc, tc = 0, 0, 0  
    # looping into string 
    res = 0 # Initialize result 
    for i in range(n):   
        # increasing the count of current character 
        if string[i] == '0': 
            zc += 1
        elif string[i] == '1': 
            oc += 1
        else: 
            tc += 1 # Assuming that string doesn't contain 
                    # other characters   
        # making pair of differences (z[i] - o[i], 
        # z[i] - t[i]) 
        tmp = (zc - oc, zc - tc)   
        # Count of previous occurrences of above pair 
        # indicates that the subarrays forming from 
        # every previous occurrence to this occurrence 
        # is a subarray with equal number of 0's, 1's 
        # and 2's 
        if tmp not in mp: 
            res += 0
        else: 
            res += mp[tmp]   
        # increasing the count of current difference 
        # pair by 1 
        if tmp in mp: 
            mp[tmp] += 1
        else: 
            mp[tmp] = 1  
    return res 

#%% 27.- Count subarrays having total distinct elements same as original array :(

""" Given an array of n integers. Count total number of sub-array having total distinct elements same as 
that of total distinct elements of original array.  """

def countDistictSubarray(arr, n):   
    # Count distinct elements in whole array 
    vis = dict() 
    for i in range(n): 
        vis[arr[i]] = 1
    k = len(vis)   
    # Reset the container by removing 
    # all elements 
    vid = dict()   
    # Use sliding window concept to find 
    # count of subarrays having k distinct 
    # elements. 
    ans = 0
    right = 0
    window = 0
    for left in range(n):       
        while (right < n and window < k):   
            if arr[right] in vid.keys(): 
                vid[ arr[right] ] += 1
            else: 
                vid[ arr[right] ] = 1  
            if (vid[ arr[right] ] == 1): 
                window += 1  
            right += 1          
        # If window size equals to array distinct  
        # element size, then update answer 
        if (window == k): 
            ans += (n - right + 1)   
        # Decrease the frequency of previous  
        # element for next sliding window 
        vid[ arr[left] ] -= 1  
        # If frequency is zero then decrease  
        # the window size 
        if (vid[ arr[left] ] == 0): 
            window -= 1      
    return ans 
        
# O(n)    
        
        
#%% 28.- Smallest subarray with k distinct numbers [Tecnicamente no es hashing]

""" We are given an array consisting of n integers and an integer k. We need to find the 
minimum range in array [l, r] (both l and r are inclusive) such that there are exactly k 
different numbers. """

def fcn(arr,k):
    n = len(arr)
    x = 0
    while (k + x) < n:
        for i in range(n-(k+x)+1):
            y = len(set(arr[i:i+k+x]))
            if y == k:
               return [i,i+k+x]
        x += 1
    return False


#%% 29.- All unique triplets that sum up to a given value :)

""" Given an array and a sum value, find all possible unique triplets in that array whose 
sum is equal to the given sum value. If no such triplets can be formed from the array, then 
print “No triplets can be formed” """

def printTriplets(arr, n, x):
    s = set()
    for i in range(n):
        for j in range(i+1, n):
            temp = x-arr[i]-arr[j]
            if temp in arr[j+1:]:
                s.add((arr[i], arr[j], temp))
    if s:
        for i in range(len(s)):
            print(list(s)[i])
    else:
        print("No triplets can be formed.")

        
#%% 43.- Maximize elements using another array

""" Given two arrays with size n, maximize the first array by using the elements from the 
second array such that the new array formed contains n greatest but unique elements of both the 
arrays giving the second array priority  """

import heapq

arr1 = [7, 4, 8, 0, 1]
arr2 = [9, 7, 2, 3, 6]

def maximizeArray(arr1, arr2):
    n = len(arr1)
    heapq.heapify(arr1)
    for i in range(n):
        if arr2[i] >= arr1[0] and arr2[i] not in arr1:
            heapq.heappop(arr1)
            heapq.heappush(arr1, arr2[i])
    return arr1
            
        
#%% 18.- Count subarrays with same even and odd elements :|

""" Given an array of N integers, count number of even-odd subarrays. An even – odd subarray is 
a subarray that contains the same number of even as well as odd integers. """

def even_odd(arr, n):
    H = {0:1}
    ec, oc = 0, 0
    ans = 0
    for i in range(n):
        if arr[i] % 2:
            ec += 1
        else:
            oc += 1
        r = ec-oc
        if r not in H:
            ans += 0
        else:
            ans += H[r]
        if r not in H:
            H[r] = 1
        else:
            H[r] += 1
    return ans
        
        
        
        
        
        
        
        
        
        
        
        