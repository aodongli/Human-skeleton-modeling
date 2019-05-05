from itertools import combinations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from read_data import load_skeleton_data

class ChowLiu:
    def __init__(self, adj_mat):
        # Adjacency matrix whose entry is mutual information of two vertices
        # adj_mat can be symmetric or upper triangular
        self.adj_mat = adj_mat

    def max_span(self):
        # return upper triangular matrix
        X = -self.adj_mat
        # Kruskal algorithm
        Tcsr = minimum_spanning_tree(X)
        return Tcsr.toarray()

class Skeleton:
    def __init__(self, data):
        # data: (num_point, num_dim, num_image)
        self.data = data
        self.shape = data.shape

        self.mu = np.zeros((self.shape[0], self.shape[1]))
        # compute covariance for each node
        self.det_cov_1d = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            node_data = self.extract_one_node(i)
            mu, sigma = self.emp_distribution(node_data)
            self.mu[i,:] = mu
            self.det_cov_1d[i] = np.linalg.det(sigma)

        # compute covariance for each two nodes
        self.det_cov_2d = np.zeros((self.shape[0], self.shape[0]))
        self.mutual_info = np.zeros((self.shape[0], self.shape[0]))
        for (idx1, idx2) in combinations(range(self.shape[0]), 2):
            node_data = self.extract_two_nodes(idx1, idx2)
            _, sigma = self.emp_distribution(node_data)
            self.det_cov_2d[idx1, idx2] = np.linalg.det(sigma)

            self.mutual_info[idx1, idx2] = -np.log(self.det_cov_2d[idx1, idx2]
                                            / self.det_cov_1d[idx1]
                                            / self.det_cov_1d[idx2]) / 2

    def extract_one_node(self, idx):
        # extract all the data for node idx 
        # return (num_image, num_dim)
        return np.transpose(self.data[idx,:,:])

    def extract_two_nodes(self, idx1, idx2):
        # extract all the data for nodes idx1 and idx2
        # return (num_image, num_dim*2)
        return np.transpose(np.concatenate((self.data[idx1,:,:], 
                                            self.data[idx2,:,:]), axis=0))

    def emp_distribution(self, data):
        # model with full Gaussian
        # data: (num_image, num_dim)
        (n, p) = data.shape
        mu = np.mean(data, axis=0)
        norm_data = data - mu
        sigma = np.matmul(norm_data.T, norm_data) / n
        return (mu, sigma)

    def plot_mean_skeleton(self):
        # joint pair
        joint = np.array([[1, 2, 3, 2, 5, 6, 7, 2, 9,  10, 11, 
                4, 13, 14, 15, 4,  17, 18, 19], [2, 3, 4, 5, 6, 7, 8, 9, 
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        joint = joint - 1

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        xs = self.mu[:,0]
        ys = self.mu[:,2]
        zs = self.mu[:,1]
        # print(np.max(xs), np.min(xs))
        # print(np.max(ys), np.min(ys))
        # print(np.max(zs), np.min(zs))

        for i in range(joint.shape[1]):
            j1 = joint[0,i]
            j2 = joint[1,i]
            ax.plot((xs[j1], xs[j2]), (ys[j1], ys[j2]), (zs[j1], zs[j2]), 'b')

        ax.scatter(xs, ys, zs)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
        ax.set_xlim3d(-0.5, 0.55)
        ax.set_ylim3d(-2.2, 3.5)
        ax.set_zlim3d(-1.5, 1.0)
        # ax.set_xlim3d(np.min(xs), np.max(xs))
        # ax.set_ylim3d(np.min(ys), np.max(zs))
        # ax.set_zlim3d(np.min(zs), np.max(zs))
        plt.savefig('./figs/ave_skeleton.jpg')


def main():
    mat = load_skeleton_data()
    skeleton = Skeleton(mat)
    # print skeleton.det_cov_1d
    # print skeleton.det_cov_2d
    # print skeleton.det_cov_2d[0,1], skeleton.det_cov_2d[0,7]
    # print skeleton.mutual_info[0,1], skeleton.mutual_info[0,7]
    cl = ChowLiu(skeleton.mutual_info)
    span_mat = cl.max_span()

    skeleton.plot_mean_skeleton()
    # joint pair
    # joint = np.array([[1, 2, 3, 2, 5, 6, 7, 2, 9,  10, 11, 
    #         4, 13, 14, 15, 4,  17, 18, 19], [2, 3, 4, 5, 6, 7, 8, 9, 
    #         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    # joint = joint - 1

    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='3d')
    # xs = skeleton.mu[:,0]
    # ys = skeleton.mu[:,2]
    # zs = skeleton.mu[:,1]
    # print(np.max(xs), np.min(xs))
    # print(np.max(ys), np.min(ys))
    # print(np.max(zs), np.min(zs))

    # for i in range(joint.shape[1]):
    #     j1 = joint[0,i]
    #     j2 = joint[1,i]
    #     ax.plot((xs[j1], xs[j2]), (ys[j1], ys[j2]), (zs[j1], zs[j2]), 'b')

    # ax.scatter(xs, ys, zs)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Z Label')
    # ax.set_zlabel('Y Label')
    # ax.set_xlim3d(-0.5, 0.55)
    # ax.set_ylim3d(-2.2, 3.5)
    # ax.set_zlim3d(-1.5, 1.0)
    # plt.savefig('./figs/ave_skeleton.jpg')
    


if __name__ == '__main__':
    main()

