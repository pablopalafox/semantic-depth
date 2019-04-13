from open3d import *
import time

def display_inlier_outlier(cloud, ind):
    inlier_cloud = select_down_sample(cloud, ind)
    outlier_cloud = select_down_sample(cloud, ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    start = time.time()
    # pcd = read_point_cloud("results/stuttgart_video/result_sequence_ply/"
    #     "stuttgart_02_000000_005256_leftImg8bit_naive.ply")
    pcd = read_point_cloud("test_road.ply")
    end = time.time()
    print(end - start)

    # print("Downsample the point cloud with a voxel of 0.02")
    # start = time.time()
    # voxel_down_pcd = voxel_down_sample(pcd, voxel_size = 0.02)
    # end = time.time()
    # print(end - start)
    # #draw_geometries([voxel_down_pcd])
    # print()

    print("Statistical oulier removal")
    start = time.time()
    cl,ind = statistical_outlier_removal(pcd,
            nb_neighbors=20, std_ratio=0.5)
    end = time.time()
    print(end - start)
    display_inlier_outlier(pcd, ind)
    print()

    pcd = select_down_sample(pcd, ind)

    print("Radius oulier removal")
    start = time.time()
    cl,ind = radius_outlier_removal(pcd,
            nb_points=80, radius=0.5)
    end = time.time()
    print(end - start)
    display_inlier_outlier(pcd, ind)

    # pcd = select_down_sample(pcd, ind)
    # pcd.paint_uniform_color([0, 1, 0])
    # draw_geometries([pcd])
