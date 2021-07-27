# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:01:15 2021

@author: faust
"""


# Python program to print topological sorting of a DAG
from collections import defaultdict

# Class to represent a graph

class Graph:
	def __init__(self, vertices):
		self.graph = defaultdict(list) # dictionary containing adjacency List
		self.V = vertices # No. of vertices
	# function to add an edge to graph
	def addEdge(self, u, v):
		self.graph[u].append(v)
	# A recursive function used by topologicalSort
	def topologicalSortUtil(self, v, visited, stack):
		# Mark the current node as visited.
		visited[v] = True
		# Recur for all the vertices adjacent to this vertex
		for i in self.graph[v]:
			if visited[i] == False:
				self.topologicalSortUtil(i, visited, stack)
		# Push current vertex to stack which stores result
		stack.append(v)

	# The function to do Topological Sort. It uses recursive
	# topologicalSortUtil()
	def topologicalSort(self):
		# Mark all the vertices as not visited
		visited = [False]*self.V
		stack = []
		# Call the recursive helper function to store Topological
		# Sort starting from all vertices one by one
		for i in range(self.V):
			if visited[i] == False:
				self.topologicalSortUtil(i, visited, stack)
		# Print contents of the stack
		print(stack[::-1]) # return list in reverse order



#%%
def spiralN(matrix):
    n = len(matrix)
    A = []
    d = 1
    lb = 0
    ub = n-1
    while len(A) < (n)*(n):
        for k in range(lb, ub, d):
            A.append(matrix[lb][k])
        for h in range(lb, ub, d):
            A.append(matrix[h][ub])
        d *= -1
        for k in range(ub, lb, d):
            A.append(matrix[ub][k])
        for h in range(ub, lb, d):
            A.append(matrix[h][lb])
        lb += 1
        ub -= 1
        d *= -1
    return A


a = [[ 1, 2, 3, 4 ],
    [ 5, 6, 7, 8 ],
    [ 9, 10, 11, 12 ],
    [ 13, 14, 15, 16 ]]
print(spiral(a))


#%%

def spiralNM(matrix):
    n = len(matrix)
    m = len(matrix[0])
    A = []
    d = 1
    lbh, lbv = 0, 0
    ubh, ubv = m-1, n-1
    while len(A) < n*m:
        for k in range(lbh, ubh, d):
            A.append(matrix[lbv][k])
        for h in range(lbv, ubv, d):  # Este no mete los ultimos
            A.append(matrix[h][ubh])
        d *= -1
        for k in range(ubh, lbh, d): # Los meten este o el siguiente
            A.append(matrix[ubv][k])
        for h in range(ubv, lbv, d):
            A.append(matrix[h][lbh])
        lbh += 1
        lbv += 1
        ubh -= 1
        ubv -= 1
        d *= -1
    return A

matrix = [[1 ,2 ,3 ,4 ,5 ],
          [6 ,7 ,8 ,9 ,10],
          [11,12,13,14,15],
          ]

print(spiralNM(matrix))


#%%




#%%

""" Elimina las islas (vertices que no tienen un camino a la orilla) """

matrix = [[1, 0, 0, 0, 0, 0],
          [0, 1, 0, 1, 1, 1],
          [0, 0, 1, 0, 1, 0],
          [1, 1, 0, 0, 1, 0],
          [1, 0, 1, 1, 0, 0],
          [1, 0, 1, 0, 0, 1]]


output = [[1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 1],
          [0, 0, 0, 0, 1, 0],
          [1, 1, 0, 0, 1, 0],
          [1, 0, 1, 1, 0, 0],
          [1, 0, 1, 0, 0, 1]]


def utilfcn(matrix, visited, copy, i, j, n, m):
    visited.add((i,j))
    neighbors = [x for x in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)] if x[0]<n and x[1]<m]
    for vtx in neighbors:
        if vtx not in visited and matrix[vtx[0]][vtx[1]] == 1:
            copy[vtx[0]][vtx[1]] = 1
            utilfcn(matrix, visited, copy, vtx[0], vtx[1], n, m)
        

def iland(matrix):
    n = len(matrix)
    m = len(matrix[0])
    copy = [[0 for i in range(m)] for j in range(n)]
    visited = set()
    for i in [0,n-1]:
        for j in range(m):
            if matrix[i][j] == 1 and (i,j) not in visited:
                copy[i][j] = 1
                utilfcn(matrix, visited, copy, i, j, n, m)
    for j in [0,m-1]:
        for i in range(m):
            if matrix[i][j] == 1 and (i,j) not in visited:
                copy[i][j] = 1
                utilfcn(matrix, visited, copy, i, j, n, m)
    return copy


