# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:13:14 2021

@author: faust
"""


def isThree(self, n: int) -> bool:
    divisors = []
    for i in range(1, n//2 + 1):
        if n%i == 0:
            divisors.extend([i,n//i])
    return len(set(divisors)) == 3


#%%

# Time Limit Exceeded 
def numberOfWeeks(self, milestones: list[int]) -> int:
    s = sum(milestones)
    n = len(milestones)
    A = [[False for i in range(s+1)] for j in range(n+1)]
    for i in range(n):
        A[i+1][milestones[i]] = True
    for i in range(1,n+1):
        for j in range(1,s+1):
            if A[i][j]:
                continue
            if j >= milestones[i-1]:
                A[i][j] = A[i-1][j] or A[i-1][j-milestones[i-1]]
    if s%2 == 0 and A[n][s//2]:
        return s
    r = 0
    for k in range(s//2,-1,-1):
        if A[n][k]:
            r = k
            break
    return r*2 +1
    

#%% Making A Large Island

"""
You are given an n x n binary matrix grid. You are allowed to change at most one 0 to be 1.
Return the size of the largest island in grid after applying this operation. An island is a 
4-directionally connected group of 1s.


Example 1:
Input: grid = [[1,0],[0,1]]
Output: 3
Explanation: Change one 0 to 1 and connect two 1s, then we get an island with area = 3.
"""


# Status: Time Limit Exceeded. 70 / 75 test cases passed. 

class Solution:
    def largestIsland(self, grid):
        n = len(grid)
        con_comp = self.conectedComponents(grid, n)
        if len(con_comp) == 1 and len(con_comp[0]) == n*n:
            return n*n
        x = 0
        for i in range(n):
            for j in range(n):
                val = 0
                if grid[i][j] == 0:
                    neighbors = set([(i-1,j), (i+1,j), (i,j-1), (i,j+1)])
                    for cc in con_comp:
                        if neighbors.intersection(set(cc)):
                            val += len(cc)
                    x = max(x,val)
        return x+1
    
    def dsf(self, grid, temp, v, visited, n):
        (i, j) = v 
        visited.add((i,j))
        temp.append((i,j))
        for (k,l) in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
            if (-1 < k < n and -1 < l < n) and (k,l) not in visited and grid[k][l] == 1:
                visited.add((k,l))
                temp = self.dsf(grid, temp, (k,l), visited, n)
        return temp

    def conectedComponents(self, grid, n): 
        visited = set()
        cc = []
        for i in range(n):
            for j in range(n):
                if (i,j) not in visited and grid[i][j] == 1:
                    visited.add((i,j))
                    temp = []
                    cc.append(self.dsf(grid, temp, (i,j), visited, n))
        return cc
    

# For each 0 in the grid, let's temporarily change it to a 1, then count the size of the group from that square.
class SolutionCorrectNaive(object):
    def largestIsland(self, grid): # O(n**4)
        N = len(grid)

        def check(r, c):
            seen = {(r, c)} # Esto es como la cc
            stack = [(r, c)]
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r-1, c), (r, c-1), (r+1, c), (r, c+1)):
                    if (nr, nc) not in seen and 0 <= nr < N and 0 <= nc < N and grid[nr][nc]:
                        stack.append((nr, nc))
                        seen.add((nr, nc))
            return len(seen)

        ans = 0
        has_zero = False
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == 0:
                    has_zero = True
                    grid[r][c] = 1
                    ans = max(ans, check(r, c))
                    grid[r][c] = 0

        return ans if has_zero else N*N
    
    
"""
As in the previous solution, we check every 0. However, we also store the size of each group, so that we do not have to use depth-first search to repeatedly calculate the same size.
However, this idea fails when the 0 touches the same group. For example, consider grid = [[0,1],[1,1]]. The answer is 4, not 1 + 3 + 3, since the right neighbor and the bottom neighbor of the 0 belong to the same group.
We can remedy this problem by keeping track of a group id (or index), that is unique for each group. Then, we'll only add areas of neighboring groups with different ids.

Algorithm
For each group, fill it with value index and remember it's size as area[index] = dfs(...).
Then for each 0, look at the neighboring group ids seen and add the area of those groups, plus 1 for the 0 we are toggling. This gives us a candidate answer, and we take the maximum.
To solve the issue of having potentially no 0, we take the maximum of the previously calculated areas.
"""

class SolutionBest(object):
    def largestIsland(self, grid): # O(n**2)
        N = len(grid)

        def neighbors(r, c):
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < N and 0 <= nc < N:
                    yield nr, nc

        def dfs(r, c, index):
            ans = 1
            grid[r][c] = index
            for nr, nc in neighbors(r, c):
                if grid[nr][nc] == 1:
                    ans += dfs(nr, nc, index)
            return ans

        area = {}
        index = 2
        for r in range(N):
            for c in range(N):
                if grid[r][c] == 1:
                    area[index] = dfs(r, c, index)
                    index += 1

        ans = max(area.values() or [0])
        for r in range(N):
            for c in range(N):
                if grid[r][c] == 0:
                    seen = {grid[nr][nc] for nr, nc in neighbors(r, c) if grid[nr][nc] > 1}
                    ans = max(ans, 1 + sum(area[i] for i in seen))
        return ans
    
    
#%% Subset II

"""
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
"""

def aux(arr):
    candidates = set(arr)
    out = []
    for c in candidates:
        temp = arr.copy()
        temp.remove(c)
        out.append(temp)
    return out
        
def fcn(arr, power_set):
    for subset in aux(arr):
        if subset not in power_set:
            power_set.append(subset)
            fcn(subset, power_set)

def subsetsWithDup(arr):
    arr.sort()
    power_set = []
    fcn(arr, power_set)
    return power_set


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


#%% Minimum Garden Perimeter to Collect Enough Apples

"""
In a garden represented as an infinite 2D grid, there is an apple tree planted at every integer coordinate. The apple tree planted at an integer coordinate (i, j) has |i| + |j| apples growing on it.
You will buy an axis-aligned square plot of land that is centered at (0, 0).
Given an integer neededApples, return the minimum perimeter of a plot such that at least neededApples apples are inside or on the perimeter of that plot.
"""

class SolutionApples:
    def minimumPerimeter(self, neededApples: int) -> int:
        A = 12
        n = 1
        while A < neededApples:
            n += 1
            A = A + 12*(n)*(n-1) + 12*n 
        return 8*n

    
        
            
#%% Path Sum II

"""
Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where each path's sum equals targetSum.

Example 1
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]
"""

# Definition for a binary tree node.
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
         
class Solution4: # Backtracking 85 time 5 memory
    def pathSum(self, root: TreeNode, targetSum: int) -> list(list([int])):
        solutions = []
        if not root:
            return []
        state = [root]
        self.search(state, targetSum, root, solutions)
        return solutions
        
    def is_valid(self, state, targetSum):
        s = sum([node.val for node in state if node])
        return s == targetSum
    
    def get_candidates(self, node):
        candidates = []
        if node.left:
            candidates.append(node.left)
        if node.right:
            candidates.append(node.right)
        return candidates
    
    def search(self, state, targetSum, node, solutions):
        if self.is_valid(state, targetSum) and not node.left and not node.right:
            state_values = [node.val for node in state.copy() if node]
            solutions.append(state_values)
            
        for candidate in self.get_candidates(node):
            state.append(candidate)
            self.search(state, targetSum, candidate, solutions)
            state.pop()
            


class Solution4_2: # DFS 50 time 43 memory
    def pathSum(self, root: TreeNode, targetSum: int) -> list(list([int])):
        solutions = []
        path = []
        if not root:
            return []
        return self.aux_fcn(targetSum, path, root, solutions)
        
    def aux_fcn(self, targetSum, path, node, solutions): 
        path = path + [node.val]
        if not node.right and not node.left:
            if sum(path) == targetSum:
                solutions.append(path.copy()) 
        if node.left:
            self.aux_fcn(targetSum, path, node.left, solutions)
        if node.right:
            self.aux_fcn(targetSum, path, node.right, solutions)
        return solutions
        


class Solution4_3: # DFS 99 time No es mio
    def dfs(self, node, num):
        if not node:
            return
        self.pathNodes.append(node.val)
        num += node.val
        if  num == self.TS and not node.left and not node.right:
            self.ans.append(list(self.pathNodes))
        else:
            self.dfs(node.left, num)
            self.dfs(node.right, num)
        self.pathNodes.pop()
        
    def pathSum(self, root: TreeNode, targetSum: int) -> list(list([int])):
        self.TS = targetSum
        self.pathNodes = []
        self.ans = []
        
        self.dfs(root, 0)

        return self.ans
    

#%%


class Solution5: # DP
    def stoneGame(self, piles) -> bool:
        n = len(piles)
        s = sum(piles)
        A = [[0 for i in range(n)] for i in range(n)]
        for i in range(n-1):
            A[i][i] = piles[i]
            A[i][i+1] = max(piles[i], piles[i+1])
        A[n-1][n-1] = piles[n-1]
        for i in range(n-2,-1,-1):
            for j in range(i+2,n):
                A[i][j] = max(piles[i]+min(A[i+2][j], A[i+1][j-1]), piles[j]+min(A[i][j-2], A[i+1][j-1]))
        return A[0][n-1] > (s - A[0][n-1])


class Solution5_2:
    def stoneGame(self, piles) -> bool: #Este es mejor en tiempo y memoria
        sz = len(piles)
        ## dp[j]: at the Lth step piles[j-L, j], max diff between Alice and Bob
        dp = [0]*sz 
        ## initialize l = 1
        for i in range(1,sz):
            dp[i] = piles[i]
        for L in range(2,sz):
            for j in reversed(range(L, sz) ):
                dp[j] = max(piles[j-L] - dp[j], piles[j] - dp[j-1] )               
        return dp[sz-1] > 0


#%%

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

class Solution6:
    def levelOrder(self, root: 'Node') -> list(list([int])):
        if not root:
            return []
        q = [root]
        output = [[root.val]]
        while len(q):
            n = len(q)
            for i in range(1,n+1):
                temp = q.pop(0)
                if i == 1:
                    temp_arr = []
                temp_arr.extend([node.val for node in temp.children if node])
                q.extend(temp.children)
            output.append(temp_arr)   
        return output[:-1]
    
    
#%% Palindrome Partitioning II

"""
Given a string s, partition s such that every substring of the partition is a palindrome.
Return the minimum cuts needed for a palindrome partitioning of s.
"""


class Solution7: # DP O(n**3)
    def minCut(self, s: str) -> int: 
        print(s)
        n = len(s)
        C = [[0 for i in range(n)] for i in range(n)]
        P = [[False for i in range(n)] for i in range(n)]
        
        for i in range(n-1):
            P[i][i] = True
            P[i][i+1] = s[i] == s[i+1]
            if not P[i][i+1]:
                C[i][i+1] = 1
                
        P[n-1][n-1] = True
        if not P[n-1][n-1]:
                C[n-1][n-1] = 1
                
        for L in range(3,n+1):
            for i in range(n-L+1):
                j = i+L-1
                P[i][j] = (s[i] == s[j]) and P[i+1][j-1]
                if P[i][j] == True:
                    C[i][j] = 0
                else:
                    C[i][j] = float('inf')
                    for k in range(i,j):
                        C[i][j] = min(C[i][j], C[i][k]+C[k+1][j]+1)
        return C[0][n-1]

    
class Solution7_2: # DP O(n**2)
    def minCut(self, a: str) -> int:
        cut = [0 for i in range(len(a))]
        palindrome = [[False for i in range(len(a))] for j in range(len(a))]
        for i in range(len(a)):
            minCut = i;
            for j in range(i + 1):
                if (a[i] == a[j] and (i - j < 2 or palindrome[j + 1][i - 1])):      
                    palindrome[j][i] = True;
                    minCut = min(minCut, 0 if  j == 0 else (cut[j - 1] + 1));
            cut[i] = minCut;  
        return cut[len(a) - 1]
    
    
class Solution7_3: # The best
    def minCut(self, s: str) -> int:
        # acceleration
        if s == s[::-1]: return 0
        for i in range(1, len(s)):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1
        # algorithm
        cut = [x for x in range(-1,len(s))]  # cut numbers in worst case (no palindrome)
        for i in range(len(s)):
            r1, r2 = 0, 0
            # use i as origin, and gradually enlarge radius if a palindrome exists
            # odd palindrome
            while i-r1 >= 0 and i+r1 < len(s) and s[i-r1] == s[i+r1]:
                cut[i+r1+1] = min(cut[i+r1+1], cut[i-r1]+1)
                r1 += 1
            # even palindrome
            while i-r2 >= 0 and i+r2+1 < len(s) and s[i-r2] == s[i+r2+1]:
                cut[i+r2+2] = min(cut[i+r2+2], cut[i-r2]+1)
                r2 += 1
        return cut[-1]

#%% Flip String to Monotone Increasing

"""
A binary string is monotone increasing if it consists of some number of 0's (possibly none), followed by some number of 1's (also possibly none).
You are given a binary string s. You can flip s[i] changing it from 0 to 1 or from 1 to 0.
Return the minimum number of flips to make s monotone increasing.

Example 1:
Input: s = "00110"
Output: 1
Explanation: We flip the last digit to get 00111.
"""

class Solution10:
    def minFlipsMonoIncr(self, s: str) -> int:
        n = len(s)
        A = [0,1] # Memo
        for i in range(1,n):
            x, y = 0, 0
            # no switch i
            a = []
            if s[i] >= s[i-1]:
                a.append(A[0])
            if int(s[i]) >= 1-int(s[i-1]):
                a.append(A[1])
            x = min(a)
            # switch i
            b = []
            if 1-int(s[i]) >= int(s[i-1]):
                b.append(A[0])
            if 1-int(s[i]) >= 1-int(s[i-1]):
                b.append(A[1])
            y = min(b)+1
            A = [x,y]
        return min(A)
    
    
    
class Solution10_2: # El mejor en tiempo
    def minFlipsMonoIncr(self, s: str) -> int:
        sl = list(s)
        l = len(s)
        c = 0
        si = sl.index('1')
        c1 = 0
        c0 = 0
        c = 0
        for i in range(si, l):
            if sl[i] == '1':
                if c1 < c0:
                    c += c1
                    c1 = 0
                    c0 = 0
                c1 += 1
            else:
                c0 += 1
        
                
        return  c + min(c1, c0)
    
    
class Solution10_3: # El mejor en memoria
    def minFlipsMonoIncr(self, s: str) -> int:
        o = z = 0
        for c in s:
            if c == '0': 
                z = min(z + 1, o)
            else:
                o += 1
        return z
