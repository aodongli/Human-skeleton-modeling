import os
import numpy as np
import scipy.io

DATA_FILE = "./data/Skeleton/"

def load_skeleton_data(data_dir=DATA_FILE):
    # data_dir: directory contains skeleton data
    # return with 3D array: num_points, num_coordinates, num_images
    mat_files = os.listdir(data_dir)
    all_mat = []
    for mat_file in mat_files:
        all_mat.append(load_skeleton_mat(data_dir + mat_file))
    return np.concatenate(all_mat, axis=2)

def load_skeleton_mat(mat_file_name):
    # return with 3D array: num_points, num_coordinates, num_frames
    # print(mat_file_name)
    mat = scipy.io.loadmat(mat_file_name)
    return mat['d_skel']

def main():
    all_mat = load_skeleton_data()
    print all_mat.shape, all_mat.dtype

if __name__ == '__main__':
    main()