# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:18:36 2021

@author: faust
"""


#%%

""" Trees """

#%%


class Node:
    def __init__(self,key):
        self.left = None
        self.right = None
        self.data = key
        
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left  = Node(4)
        

""" 
Main applications of trees include: 
1. Manipulate hierarchical data. 
2. Make information easy to search (see tree traversal). 
"""


#%% 1.- Left View of Binary Tree 

"""Given a Binary Tree, print Left view of it. Left view of a Binary Tree is set of nodes
 visible when tree is visited from Left side. """ 

def LeftView(root):
    output = []
    if not root:
        return []
    q = []
    q.append(root)
    while len(q): # No puedes poner for node in q: cause q is dinamic so for loop throw error, you need a while loop  
        # number of nodes at current level
        n = len(q)
        ## Traverse all nodes of current level
        for i in range(1, n + 1):
            temp = q.pop(0)
            # Print the left most element at the level
            if i == 1:
                output.append(temp.data)
            # Add left node to queue
            if temp.left != None:
                q.append(temp.left)
            # Add right node to queue
            if temp.right != None:
                q.append(temp.right)
    return output


#%% 2.- Check for BST 

""" Given a binary tree. Check whether it is a BST or not. """

INT_MAX = float('-inf')
INT_MIN = float('inf')

def isBST(root):
    return (isBSTUtil(root, INT_MIN, INT_MAX))
# Retusn true if the given tree is a BST and its values
# >= min and <= max
def isBSTUtil(node, mini, maxi):     
    # An empty tree is BST
    if node is None:
        return True 
    # False if this node violates min/max constraint
    if node.data < mini or node.data > maxi:
        return False
    # Otherwise 
    ### check the subtrees recursively
    # tightening the min or max constraint
    return (isBSTUtil(node.left, mini, node.data -1) and
          isBSTUtil(node.right, node.data+1, maxi))

        
        
#%% 3.- Bottom View of Binary Tree :|


""" Given a binary tree, print the bottom view from left to right. A node is included in bottom 
 view if it can be seen when we look at the tree from bottom. """

def util_fcn(node, vd, hd, d): # this utility fcn runs through the tree and register the last node on the horizontal level 
    if not node:
        pass
    if node.left:
        if hd-1 not in d:
            d[hd-1] = (vd+1, node.left.data)
        else:
            x = d[hd-1]
            if x[0] <= vd+1:
                d[hd-1] = (vd+1, node.left.data)
        util_fcn(node.left, vd+1, hd-1, d)
    if node.right:
        if hd+1 not in d:
            d[hd+1] = (vd+1, node.right.data)
        else:
            x = d[hd+1]
            if x[0] <= vd+1:
                d[hd+1] = (vd+1, node.right.data)
        util_fcn(node.right, vd+1, hd+1, d)
    
def bottomView(root):
    H = {0:(0, root.data)}
    util_fcn(root, 0, 0, H)
    output = []
    for i in sorted(H.keys()):
        output.append(H[i][1])
    return output  

#%% 4.- Vertical Traversal of Binary Tree :)

""" Given a Binary Tree, find the vertical traversal of it starting from the leftmost level to 
the rightmost level. If there are multiple nodes passing through a vertical line, then they 
should be printed as they appear in level order traversal of the tree. """

def fcn1(node, vd, hd, h):
    if not node:
        return
    if node.left:
        h[hd-1].append((vd+1, node.left.data))
        fcn1(node.left, vd+1, hd-1, h)
    if node.right:
        h[hd+1].append((vd+1, node.right.data))
        fcn1(node.right, vd+1, hd+1, h)

def verticalOrder(root): 
    H = defaultdict(lambda : [])
    H[0].append((0, root.data))
    fcn1(root, 0, 0, H)
    output = []
    for i in sorted(H.keys()):
        a = H[i] # a is lines of nodes with the same hd
        a.sort(key=lambda v: v[0]) # sort the a's nodes by their vd
        output.extend([i[1] for i in a])
    return output
    
    
#%% 5.- Level order traversal in spiral form :)

""" Complete the function to find spiral order traversal of a tree. """


from collections import defaultdict

def fcn(node, vd, hd, h):
    if not node:
        pass
    if node.left:
        h[vd+1].append((hd-1, node.left.data))
        fcn(node.left,vd+1, hd-1, h)
    if node.right:
        h[vd+1].append((hd+1, node.right.data))
        fcn(node.right,vd+1, hd+1, h)
        
        
def findSpiral(root):
    if not root:
        return []
    h = defaultdict(lambda : [])
    h[0].append((0,root.data))
    fcn(root,0,0,h)
    output = []
    for i in sorted(h.keys()):
        if i % 2 != 0:
            a = [j[1] for j in h[i]]
            output.extend(a)
        else:
            a = []
            for j in range(len(h[i])-1,-1,-1):
                a.append(h[i][j][1])
            output.extend(a)
    return output
    

#%% 7.- Lowest Common Ancestor in a BST 

""" Given a Binary Search Tree (with all values unique) and two node values. Find the Lowest 
Common Ancestors of the two nodes in the BST. """

def LCA(root, n1, n2):
    if root is None:
        return 
    if root.data > n1 and root.data > n2:
        return LCA(root.left, n1, n2)
    if root.data < n1 and root.data < n2:
        return LCA(root.right, n1, n2)
    return root
    
    
    
#%% 9 .- Determine if Two Trees are Identical 

""" Pues eso """

def identicalTrees(a, b):
    if a is None and b is None:
        return True 
    if a is not None and b is not None:
        return ((a.data == b.data) and 
                identicalTrees(a.left, b.left) and
                identicalTrees(a.right, b.right))
    return False
    
    
#%% 10.- Symmetric Tree

""" Pues eso """

def isSymmetric(root):
    if root is None:
        return True
    return mirror(root, root)
    
def mirror(a, b):
    if a is None and b is None:
        return True 
    if a is not None and b is not None:
        if a.data == b.data:
            return mirror(a.left, b.right) and mirror(a.right, b.left)
    return False


#%% 11.- Height of Binary Tree :|

""" Pues eso """

def height(root):
    def son(node, vd, a):
        if node is None:
            return 0
        if node.left:
            a.add(vd+1)
            son(node.left, vd+1, a)
        if node.right:
            a.add(vd+1)
            son(node.right, vd+1, a)
    a = set()
    if root:
        a.add(1)
    else:
        a.add(0)
    son(root,1,a)
    return max(a)


def maxDepth(node): # es la mejor
    if node is None:
        return 0 ;
    else :
        # Compute the depth of each subtree
        lDepth = maxDepth(node.left)
        rDepth = maxDepth(node.right)
        # Use the larger one
        return max(lDepth, rDepth)+1
       
        
def height2(root): # otra no tan buena
    # base condition when binary tree is empty
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    return max(height(root.left), height(root.right))
 
        
        
#%% 12.- Maximum Path Sum between 2 Leaf Nodes 

""" Given a binary tree in which each node element contains a number. Find the maximum possible 
 sum from one leaf node to another. """

INT_MIN = float('-inf')

def maxPathSumUtil(root, res): 
    # Base Case
    if root is None:
        return 0 
    # Find maximumsum in left and righ subtree. Also
    # find maximum root to leaf sums in left and righ
    # subtrees ans store them in ls and rs
    ls = maxPathSumUtil(root.left, res)
    rs = maxPathSumUtil(root.right, res) 
    # If both left and right children exist
    if root.left is not None and root.right is not None: 
        # update result if needed
        res = max(res, ls + rs + root.data) 
        # Return maximum possible value for root being
        # on one side
        return max(ls, rs) + root.data 
    # If any of the two children is empty, return
    # root sum for root being on one side
    if root.left is None:
        return rs + root.data
    else:
        return ls + root.data 
# The main function which returns sum of the maximum
# sum path betwee ntwo leaves. THis function mainly
# uses maxPathSumUtil() 
def maxPathSum(root):
    res = INT_MIN
    maxPathSumUtil(root, res)
    return res


#%% 13.- Diameter of a Binary Tree

""" The diameter of a tree (sometimes called the width) is the number of nodes on the longest 
path between two end nodes.  """

def height4(node):
    # Base Case : Tree is empty
    if node is None:
        return 0
    # If tree is not empty then height = 1 + max of left
    # height and right heights
    return 1 + max(height(node.left), height(node.right))
 
# Function to get the diameter of a binary tree
def diameter(root): 
    # Base Case when tree is empty
    if root is None:
        return 0 
    # Get the height of left and right sub-trees
    lheight = height4(root.left)
    rheight = height4(root.right) 
    # Get the diameter of left and right sub-trees
    ldiameter = diameter(root.left)
    rdiameter = diameter(root.right) 
    # Return max of the following tree: # PIENZA ANTES DE ACTUAR NMM
    # 1) Diameter of left subtree
    # 2) Diameter of right subtree
    # 3) Height of left subtree + height of right subtree +1 (the root)
    return max(lheight + rheight + 1, max(ldiameter, rdiameter))



#%% 14.- Count Leaves in Binary Tree :)

""" Pues eso """

def countLeaves(root):
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    else:
        return countLeaves(root.left)+countLeaves(root.right)

#%% 15.- Check for Balanced Tree 


""" Given a binary tree, find if it is height balanced or not. A tree is height balanced if 
difference between heights of left and right subtrees is not more than one for all nodes of tree. """ 

def height3(root): 
    if root is None:
        return 0
    return max(height3(root.left), height3(root.right)) + 1
 
# function to check if tree is height-balanced or not
def isBalanced(root):
    # Base condition
    if root is None:
        return True
    # for left and right subtree height
    lh = height3(root.left)
    rh = height3(root.right) 
    # allowed values for (lh - rh) are 1, -1, 0
    if (abs(lh - rh) <= 1) and isBalanced(root.left) and isBalanced( root.right):
        return True 
    # if we reach here means tree is not
    # height-balanced tree
    return False


