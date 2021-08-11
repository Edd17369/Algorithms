# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:39:18 2021

@author: faust
"""


#%%

""" Backtrackikng """

#%%

" Template backtracking "

def is_valid_state(state):
    # check if it is a valid solution
    return True

def get_candidates(state):
    return []

def search(state, solutions):
    if is_valid_state(state):
        solutions.append(state.copy())
        # return

    for candidate in get_candidates(state):
        state.add(candidate)
        search(state, solutions)
        state.remove(candidate)

def solve():
    solutions = []
    state = set()
    search(state, solutions)
    return solutions

#%% Combination sum :)


candidates = [2,3,6,7]
target = 7


def combinationSum(candidates, target):
    solutions = []
    state = []
    search(state, solutions, candidates, target)
    return solutions


def is_valid_state(state, target):
    return sum(state) == target

def get_candidates(state, candidates, target):
    s = sum(state)
    new_candidates = set(candidates)
    if not s:
        return new_candidates
    indx = candidates.index(state[-1])
    for c in candidates:
        if s+c > target or candidates.index(c) < indx:
            new_candidates.discard(c)
    return new_candidates

def search(state, solutions, candidates, target):
    if is_valid_state(state, target):
        solutions.append(state.copy())
        
        
    for candidate in get_candidates(state, candidates, target):
        state.append(candidate)
        is_valid = search(state, solutions, candidates, target)
        if is_valid:
            return True
        else:
            state.pop()
    return False
        
print(combinationSum(candidates, target))


#%% Sudoku solver :|


from itertools import product

def is_valid_board(board):
    x = [0]*9
    for i in range(9):
        if '.' in board[i]:
            x[i] = False
    return False not in x 

def get_candidates2(board, k, j): 
    candidates = set(range(9))
    row = set(board[k])
    candidates = candidates.difference(row)
    col = set([board[i][j] for i in range(9)])
    candidates = candidates.difference(col)
    # definir el cuadrado de 3x3 en el que esta la row y la col
    a1, a2 = j - j%3, k - k%3
    square = set([board[x][y] for x, y in product(range(a2,a2+3), range(a1,a1+3))]) 
    candidates = candidates.difference(square)
    return candidates
    
def search2(board):
    #if is_valid_board(board): # Esto se puede quitar, lines 67-69 comprueban que no hay espacios en blanco,
    #    return True           # y lines 78-79 deluelven un boleano. 
    
    # No tiene mucho sentido revisar si cumple las condiciones de sudoku ya que los candidatos
    # se escojen de modo que cumplan estas condiciones por eso solo hay de revisar que no hay 
    # espacios en blanco
    
    col =  0
    for row in range(9):
        if '.' in board[row]:
            col = board[row].index('.')
            for candidate in get_candidates2(board, row, col):
                board[row][col] = candidate
                is_solved = search2(board) # ya que search2 returns a boolean
                if is_solved:
                    return True
                else:
                    board[row][col] = '.'
            return False # return False si hay espacios en blanco y no quedan candidatos
    return True # Este entra cuando el tablero esta completo, es decir, se encontro una solucion
            
        
board = [ [4,   2 , '.', '.',  6,   '.', '.', '.', '.'],
          ['.',  '.', '.',  0,   8,   4,   '.', '.', '.'],
          ['.', '.',   7,  '.', '.',  '.', '.', '.',   '.'],
          [7,   '.', '.', '.', 5,   '.', '.', '.',  2 ],
          ['.',   '.', '.', 7,   '.', '.',   '.', '.',  0 ],
          ['.',   '.', '.', '.', 1,   '.', '.', '.',  5 ],
          ['.', 5,   '.', '.', '.', '.', 1,   7,   '.'],
          ['.', '.', '.', 3,   '.',   '.',   '.', '.',  4 ],
          ['.', '.', '.', '.', 7,   '.', '.', '.',    '.' ]] 
    
search2(board)

print(board)

#%% N-Queens :(

class Solution:
    """
    example on the left: [1, 3, 0, 2]
    example on the right: [2, 0, 3, 1]
    """
    def solveNQueens(self, n) 
        solutions = []
        state = []
        self.search(state, solutions, n)
        return solutions
        
    def is_valid_state(self, state, n):
        # check if it is a valid solution
        return len(state) == n

    def get_candidates(self, state, n):
        if not state:
            return range(n)
        
        # find the next position in the state to populate
        position = len(state)
        candidates = set(range(n))
        # prune down candidates that place the queen into attacks
        for row, col in enumerate(state):
            # discard the column index if it's occupied by a queen
            candidates.discard(col)
            dist = position - row
            # discard diagonals
            candidates.discard(col + dist)
            candidates.discard(col - dist)
        return candidates

    def search(self, state, solutions, n):
        if self.is_valid_state(state, n):
            state_string = self.state_to_string(state, n)
            solutions.append(state_string)
            return

        for candidate in self.get_candidates(state, n):
            # recurse
            state.append(candidate)
            self.search(state, solutions, n)
            state.pop()
    
    def state_to_string(self, state, n):
        # ex. [1, 3, 0, 2]
        # output: [".Q..","...Q","Q...","..Q."]
        ret = []
        for i in state:
            string = '.' * i + 'Q' + '.' * (n - i - 1)
            ret.append(string)
        return ret



#%% 24 Game X|


import operator
state = ''
cards = [2,4,5,6]

integers = [str(i) for i in range(1,10)]
ops = { '+': operator.add, '-': operator.sub, '*':operator.mul, '/':operator.truediv}

def is_valid_state(state):
    try:
        eval(state)
        return eval(state) == 24
    except:
        return False

def get_candidates(state, cards):
    candidates = set([str(i) for i in cards])
    no_candidates = set([i for i in state if i in integers])
    return candidates.difference(no_candidates)
            

def search(state, cards):
    for candidate in get_candidates(state, cards):
        if not state:
            state = candidate
            search(state,cards)
            state = ''
        for op in ops:
            old_state = state
            state = '(' + state + op + candidate + ')'
            search(state, cards)
            state = old_state
            
    if is_valid_state(state):  # si lo encuentra pero no lo devuelve
        print(state)
        return state


def solve(cards):
    state = ''
    search(state, cards)
    return False if not state else True
    


#%% Word Search

"Error: Maximum recursion"

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCCED"

state = []



n = len(board)
m = len(board[0])

def is_valid_state(board, state, word):
    # check if it is a valid solution
    word_state = [board[i][j] for i,j in state]
    return ''.join(word_state) == word

def get_candidates(state, i, j):
    if len(state) >= len(word):
        return []
    neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
    candidates = [x for x in neighbors if -1 < x[0] < n and -1 < x[1] < m and x not in state]
    return candidates


def search(state, solutions, row, col):
    if is_valid_state(board, state, word):
        solutions.append(state.copy())
        return 


    for x, y in get_candidates(state, row, col):
        if board[x][y] in word:
            state.append((x,y))
            print(state)
        search(state, solutions, x, y)
        state.pop()
        
def exist(board, word):
    state = [(0,0)]
    solutions = []
    search(state, solutions, 0, 0)
               

#%% Increasing Subsequences :')


nums = [4,6,7,7]

def search1(solutions, state, nums, indx):
    if len(state) > 1:
        # Los conjuntos y diccionarios no pueden almacenar listas pero 
        # si pueden almacenar tuplas
        solutions.add(tuple(state.copy())) 
    

    for i in range(indx+1, len(nums)):
        # Esto es practico si no hay state, state[-1] no tira error
        if not state or nums[i] >= state[-1]: 
            state.append(nums[i])
            search1(solutions, state, nums, i)
    if state:
        state.pop()
    
def solve1(nums):
    state = []
    solutions = set()
    search1(solutions, state, nums, -1)
    return solutions
    
"""
Runtime: 232 ms, faster than 54.92% of Python3 online submissions for Increasing Subsequences.
Memory Usage: 22.7 MB, less than 37.43% of Python3 online submissions for Increasing 
"""

