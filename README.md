# Blobtrees
Blob-trees are a way of connecting a set of points in the plane by a mixture of enclosing them by cycles (as in the convex hull) and   connecting them by edges (as in a spanning tree). The program computes a minimum-cost blob tree for n points in O(n^3) time.

In the default configuration, blobtree.py generates 23 random points in the unit square and computes the optimum blobtree. Output is provided graphically as a file blobsol.ipe, while can be processed by 
Otfried Cheong's extensible drawing editor ipe, https://ipe.otfried.org/. In addition to the optimal solution, the result file shows the solutions of some selected subproblems.

Alternatively, the name of an XML-file created by ipe can be given as a parameter on the command-line, and then the program reads points from that file. This feature is currently only rudimentary and not fool-proof.

The algorithm is described in the paper "Minimum spanning blob-trees" by Katharina Klost, Marc van Kreveld, Daniel Perz, Günter Rote, and Josef Tkadlec.
