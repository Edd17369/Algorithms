# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:19:42 2021

@author: faust
"""


class Node: 
    # Function to initialise the node object 
    def __init__(self, data): 
        self.data = data  # Assign data 
        self.next = None  # Initialize next as null 
  
  
# Linked List class contains a Node object 
class LinkedList: 
    # Function to initialize head 
    def __init__(self): 
        self.head = None
        
    def print(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next
    
    def push(self, new_data): # push at the begining of the llist
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node
    
    def pushBetween(self, prev_node, new_data): # prev_node tiene que ser un nodo
        if prev_node is None:  
            print("The given previous node must be in LinkedList.")
            pass
        new_node = Node(new_data)
        new_node.next = prev_node.next
        prev_node.next = new_node
        
    def pushEnd(self,new_data):
        new_node =  Node(new_data)
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node
        
    def popKey(self, key):
        temp = self.head
        if temp.data == key: 
            self.head = temp.next
            temp = None
            return
        while temp: 
            if temp.data == key: 
                break
            prev = temp 
            temp = temp.next
        if (temp == None): 
            return
        prev.next = temp.next
        temp = None
    
    def popPos(self, pos):
        p = 1
        temp = self.head
        if pos == p: # si p = 1
            self.head = temp.next
            return 
        while temp: # esta parte
            if p == pos:
                break
            prev = temp
            temp = temp.next
            p += 1
        if not temp: # esto es si nunca llegua a la posicion
            return
        prev.next = temp.next
        
    def deletList(self):
        current = self.head
        while current:
            prev = current.next
            del current.data
            current = prev
            
    def length(self):
        l = 1
        current = self.head
        while current.next:
            current = current.next
            l += 1
        return l
    
    def reverse(self):
        prev = None 
        current = self.head
        while current:
            next = current.next
            current.next = prev
            prev = current
            current = next
        self.head = prev
            
            
        
llist = LinkedList() 
   

first = Node(1)
second = Node(2) 
third = Node(3) 
fourth = Node(4)

llist.head = first  
llist.head.next = second
second.next = third 
third.next = fourth
fourth.next = Node(5)
  

#%% 1.- Finding middle element in a linked list 

""" Given a singly linked list of N nodes. The task is to find the middle of the linked list. 
If there are even nodes, then there would be two middle nodes, we need to print the second 
middle element.  """

import math

def findMid(head): # Espacio O(n)
    if head is None:
        return 
    current = head
    arr = [current.data]
    l = 0
    while current.next:
        arr.append(current.next.data)
        l += 1
        current = current.next
    return arr[math.ceil(l/2)] 

def findMid2(head): # Espacio O(1)
    if head is None:
        return None    
    ptr1 = head
    ptr2 = head
    while(ptr2 is not None and ptr2.next is not None):
        ptr1 = ptr1.next
        # this pointer moves 1 nodes ahead everytime loop is run        
        ptr2 = ptr2.next.next
        # this pointer moves 2 nodes ahead everytime loop is run    
    return ptr1.data
    # since slow was moving with half speed, it is there at halfway point


#%% 2.- Reverse a linked list

""" Given a linked list of N nodes. The task is to reverse this list. The function returns new 
head after reversing the list. """

def reverseList(head):
    prev = None # If node is the end of a llist then node.next = None  
    current = head
    while current:
        next = current.next
        current.next = prev
        prev = current
        current = next
    head = prev
    return head

#%% 4.- Reverse a Linked List in groups of given size

""" Given a linked list of size N. The task is to reverse every k nodes (where k is an input 
to the function) in the linked list. """

def rotate(head_ref, k): # Incorrecto, pero deberia de funcionar mejor
    if (k == 0):
        return
    current = head_ref
    new_head = head_ref
    l = 1
    while (current.next != None): # El loop no termina
        current = current.next
        l += 1
        if l == k+1:
            new_head = current
    current.next = head_ref
    #current.next = None # necesita algo asi
    head_ref = new_head
    return head_ref.data


def rotate2(head_ref, k):
    if (k == 0):
        return
    current = head_ref
    while (current.next != None):
        current = current.next  
    current.next = head_ref
    current = head_ref
    # Traverse the linked list to k-1
    # position which will be last element
    # for rotated array.
    for i in range(k - 1):
        current = current.next
    # Update the head_ref and last
    # element pointer to None
    head_ref = current.next
    current.next = None
    return head_ref


#%% 6.- Detect Loop in linked list 

""" Pues eso """

def detectLoop(head): # Espacio O(n) Tiempo O(n)
    curr = head
    H = set()
    while curr:
        if curr in H:
            return True
        H.add(curr)
        curr = curr.next
    return False

# Floyd’s Cycle-Finding Algorithm
def detectLoopFloyd(head): # Espacio O(1) Tiempo O(n)
        slow_p = head
        fast_p = head
        while (slow_p and fast_p and fast_p.next):
            slow_p = slow_p.next
            fast_p = fast_p.next.next
            if slow_p == fast_p:
                return True
        return False
    
    
#%% 7.- Remove loop in Linked List 

""" You are given a linked list of N nodes. Remove the loop from the linked list, if present. """

def detectAndRemoveLoop(self):
    slow_p = fast_p = self.head
    while(slow_p and fast_p and fast_p.next):
        slow_p = slow_p.next
        fast_p = fast_p.next.next
        # If slow_p and fast_p meet at some point then
        # there is a loop
        if slow_p == fast_p:
            self.removeLoop(slow_p)     
            # Return 1 to indicate that loop is found
            return 1     
    # Return 0 to indicate that there is no loop
    return 0
 
# Function to remove loop
# loop_node --> pointer to one of the loop nodes
# head --> Pointer to the start node of the linked list
def removeLoop(self, loop_node):
    ptr1 = loop_node
    ptr2 = loop_node
    # Count the number of nodes in loop
    k = 1
    while(ptr1.next != ptr2):
        ptr1 = ptr1.next
        k += 1
    # Fix one pointer to head
    ptr1 = self.head
    # And the other pointer to k nodes after head
    ptr2 = self.head
    for i in range(k):
        ptr2 = ptr2.next
    # Move both pointers at the same place
    # they will meet at loop starting node
    while(ptr2 != ptr1):
        ptr1 = ptr1.next
        ptr2 = ptr2.next
    # Get pointer to the last node
    while(ptr2.next != ptr1):
        ptr2 = ptr2.next
    # Set the next node of the loop ending node
    # to fix the loop
    ptr2.next = None
    
#%% 8.- Nth node from end of linked list 

""" Given a linked list consisting of L nodes and given a number N. The task is to find the Nth 
node from the end of the linked list. """

def length(head):
    l = 1
    current = head
    while current.next:
        current = current.next
        l += 1
    return l

def getNthFromLast(head,n):
    l = length(head)
    if n > l:
        return -1
    curr = head
    for i in range(l-n):
        curr = curr.next
    return curr.data


def getNthFromLast2(head,n): # Using pointers
    p1 = head
    p2 = head
    for i in range(n):
        p2 = p2.next
        if p2 is None and i < n-1:
            return -1
    while p2 is not None:
        p1 = p1.next
        p2 = p2.next
    return p1.data

#%% 10.- Merge two sorted linked lists 

""" Given two sorted linked lists consisting of N and M nodes respectively. The task is to 
merge both of the list (in-place) and return head of the merged list. """

def sortedMerge(head1, head2):
    # code here
    p1 = head1
    p2 = head2
    if p1.data <= p2.data:
        head3 = Node(p1.data)
        p1 = p1.next
    else:
        head3 = Node(p2.data)
        p2 = p2.next
    curr = head3
    while p1 is not None and p2 is not None:
        if p1.data <= p2.data:
            curr.next = Node(p1.data)
            p1 = p1.next
        else:
            curr.next = Node(p2.data)
            p2 = p2.next
        curr = curr.next
        
    if p1 or p2:
        temp = p1 or p2
    while temp:
        curr.next = Node(temp.data)
        temp = temp.next
        curr = curr.next
    return head3


#%% 11.- Intersection Point in Y Shapped Linked Lists

""" Given two singly linked lists of size N and M, write a program to get the point where two 
linked lists intersect each other. """

def intersetPoint(head1,head2): # Hay que usar punteros no hash
    H = set()
    curr = head1
    while curr:
        H.add(curr)
        curr = curr.next
    curr2 = head2
    while curr2:
        if curr2 in H:
            return curr2.data
        curr2 = curr2.next
    return

