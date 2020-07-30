"""
 A tree data structure which stores a list of degrees and can quickly retrieve the min degree element,
 or modify any of the degrees, each in logarithmic time. It works by creating a binary tree with the
 given elements in the leaves, where each internal node stores the min of its two children.
"""

import math


class MinTree:
    def __init__(self, degrees):
        self.input_length = len(degrees)
        self.height = int(math.ceil(math.log(len(degrees), 2)))
        self.numLeaves = 2 ** self.height
        self.numBranches = self.numLeaves - 1
        self.n = self.numBranches + self.numLeaves
        self.nodes = [float('inf')] * self.n
        for i in range(len(degrees)):
            self.nodes[self.numBranches + i] = degrees[i]
        for i in reversed(range(self.numBranches)):
            self.nodes[i] = min(self.nodes[2 * i + 1], self.nodes[2 * i + 2])

    def getMin(self):
        cur = 0
        for i in range(self.height):
            cur = (2 * cur + 1) if self.nodes[2 * cur + 1] <= self.nodes[2 * cur + 2] else (2 * cur + 2)
        # print "found min at %d: %d" % (cur, self.nodes[cur])
        return (cur - self.numBranches, self.nodes[cur])

    def changeVal(self, idx, delta):
        cur = self.numBranches + idx
        self.nodes[cur] += delta
        for i in range(self.height):
            cur = (cur - 1) // 2
            nextParent = min(self.nodes[2 * cur + 1], self.nodes[2 * cur + 2])
            if self.nodes[cur] == nextParent:
                break
            self.nodes[cur] = nextParent

    def dump(self):
        print ("numLeaves: %d, numBranches: %d, n: %d, nodes: " % (self.numLeaves, self.numBranches, self.n))
        cur = 0
        for i in range(self.height + 1):
            for j in range(2 ** i):
                print (self.nodes[cur])
                cur += 1
            print ('')
            
    def print_leaves(self):
        for i in range(self.input_length):
            print (self.nodes[self.numBranches + i])
