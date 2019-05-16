from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_data import load_skeleton_data

class Skeleton:
    def __init__(self, data):
        # data: (num_point, num_dim, num_image)
        self.data = data
        self.shape = data.shape
        self.mu = np.mean(self.data, axis=2)

    def cov_and_mutual_info(self):
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

    def stack_data(self):
        # vectorize each image
        return self.data.transpose(2,0,1).reshape((self.shape[2],-1))

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
        plt.savefig('./figs/ave_skeleton.jpg')

    def plot_custom_skeleton(self, joint, name, title=""):
        # joint: [2, num_edges] where the 1st dim represents 
        # start_node and end_node
        # if np.size(joint) == 0:
        #     raise Exception('joint is empty.')
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        xs = self.mu[:,0]
        ys = self.mu[:,2]
        zs = self.mu[:,1]

        if np.size(joint) != 0:
            for i in range(joint.shape[1]):
                j1 = int(joint[0,i])
                j2 = int(joint[1,i])
                ax.plot((xs[j1], xs[j2]), (ys[j1], ys[j2]), (zs[j1], zs[j2]), 'b')

        ax.scatter(xs, ys, zs)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
        ax.set_xlim3d(-0.5, 0.55)
        ax.set_ylim3d(-2.2, 3.5)
        ax.set_zlim3d(-1.5, 1.0)
        plt.title(title)
        plt.savefig('./figs/' + name + '.jpg')

def main():
    mat = load_skeleton_data()
    skeleton = Skeleton(mat)
    # print skeleton.data
    # print skeleton.stack_data()
    print skeleton.data.shape
    print skeleton.stack_data()[6, :]
    print skeleton.data[:,:,6]

if __name__ == '__main__':
    main()