import numpy as np
from IPython.core.debugger import set_trace

class SumTree():
    # capacity should be power of two
    def __init__(self, capacity):
        assert (capacity % 2) == 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype='object')
        self.nextIdx = 0
        
    def insert(self, val, data):
        self.data[self.nextIdx] = data
        treeIdx = self.capacity - 1 + self.nextIdx
        change = val - self.tree[treeIdx]
        while treeIdx > 0:
            self.tree[treeIdx] += change 
            treeIdx = (treeIdx - 1 )// 2
        self.tree[treeIdx] += change
            
        self.nextIdx = (self.nextIdx + 1) % self.capacity
    
    def find_val_idx(self,val):
        i = 0
#         set_trace()
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            if val <= self.tree[l]:
                i = l
            else:
                i = r
                val -= self.tree[l]
            if i >= self.capacity - 1:
                break
        return i - self.capacity + 1
