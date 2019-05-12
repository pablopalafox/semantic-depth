import numpy as np
from open3d import *
import glob
import os
import matplotlib.pyplot as plt

def render_plys(ply_file, ply_name):
    pcd = read_point_cloud(ply_file)

    vis = Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    
    param = read_pinhole_camera_parameters("top.json")
    
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)

    ##########################################
    ## UNCOMMENT TO SAVE INTRINSICS AS JSON
    # vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # write_pinhole_camera_parameters("frontal.json", param)
    # exit()
    ##########################################

    
    image = vis.capture_screen_float_buffer(True)
    plt.imsave(ply_name, np.asarray(image), dpi = 1)    

    vis.destroy_window()

    

if __name__ == "__main__":
    base_folder = "../results/stuttgart_video/"

    ply_files = glob.glob(base_folder + "result_sequence_ply/*")
    output_folder = base_folder + "rendered_sequence/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 

    view_point_is_set = False

    for ply_file in ply_files:

        print(ply_file)

        ply_name = os.path.basename(ply_file)
        ply_name = os.path.splitext(ply_name)[0] + '.png'
        ply_name = output_folder + ply_name

        render_plys(ply_file, ply_name)
        

     

