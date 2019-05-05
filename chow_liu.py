import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from read_data import load_skeleton_data
from skeleton import Skeleton

class ChowLiu:
    def __init__(self, adj_mat):
        # Adjacency matrix whose entry is mutual information of two vertices
        # adj_mat can be symmetric or upper triangular
        self.adj_mat = adj_mat

    def max_span(self):
        # return upper triangular matrix
        # zero for non-edge; non-zero for edge
        X = -self.adj_mat
        # Kruskal algorithm
        Tcsr = minimum_spanning_tree(X)
        self.T = -Tcsr.toarray()
        return self.T

    def sum_mutual_info(self):
        return np.sum(self.T)

    def extract_edges(self):
        # return an array containing start nodes and end nodes
        try:
           self.T
        except NameError:
           print 'T not defined'

        self.joint = np.zeros((2, self.adj_mat.shape[0]-1))
        idx = 0
        for i in range(self.adj_mat.shape[0]):
            for j in range(i, self.adj_mat.shape[0]):
                if self.T[i,j] != 0:
                    self.joint[0,idx] = i
                    self.joint[1,idx] = j
                    idx += 1
        return self.joint

def main():
    mat = load_skeleton_data()
    skeleton = Skeleton(mat)

    # draw average skeleton
    skeleton.plot_mean_skeleton()

    # run chow-liu algorithm
    cl = ChowLiu(skeleton.mutual_info)
    cl.max_span()
    print("Maximum mutual information: %f" % (cl.sum_mutual_info()))
    joint = cl.extract_edges()
    # print joint

    # plot joint learnt by chow-liu algorithm
    skeleton.plot_custom_skeleton(joint, 'chowliu_skeleton')
    

if __name__ == '__main__':
    main()

