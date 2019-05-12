# This file is licensed under a GPLv3 License.
#
# GPLv3 License
# Copyright (C) 2018-2019 Pablo R. Palafox (pablo.rodriguez-palafox@tum.de)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


'''
Hand-made Point Cloud Library to deal with the most basic operations
one could want to apply to a 3D Point Cloud 
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.linalg


def remove_from_to(points3D, colors, axis, from_meter, to_meter):

    inliers_indices = []
    min_value_in_axis = min(points3D[:,axis])

    for p in range(points3D.shape[0]):
        if (points3D[p][axis]) < -to_meter:
            inliers_indices.append(p)


    points3D = points3D[inliers_indices]
    colors   = colors[inliers_indices]  

    return points3D, colors


def remove_noise_by_mad(points3D, colors, axis, threshold=15.0):
    ''' 
    The criteria is to apply Median Absolute Deviation in one of the 
    dimensions of the point cloud
    ``axis``: x (0), y(1), z(2)
    '''
    
    # print("\nRemoving noise from the axis '{}' by applying MAD...".format(axis))
    
    # First, we'll get the ``axis`` coordinates of the 3D Point Cloud we want to denoise
    points3D_axis = points3D[:, axis]
    #print(points3D_axis)
    # We compute the Median Absolute Deviation of the set of 'y' coordinates
    abs_diffs, mad_axis = mad(points3D_axis)
    #print(abs_diffs)
    #print(mad_axis)
    # We compute the penalty of each element
    penalty = 0.6745 * abs_diffs / mad_axis
    #print(penalty)
    # Now, we get the indices of the points whose 'y' has a penalty lower than
    # ``threshold``, that is, the indices of the inliers
    inliers_indices = np.where(penalty < threshold)
    #print("Indices of inliers", inliers_indices)
    # Finally, remove noisy points from original 3D Point Cloud (``poins3D``)
    # and remove also corresponding data from the colors array
    points3D = points3D[inliers_indices]
    colors   = colors[inliers_indices]
    return points3D, colors


def mad(points1D):
    ''' Computes the Median Absolute Deviation '''
    median = np.median(points1D)
    abs_diffs = abs(points1D - median)
    mad = np.median(abs_diffs)
    return abs_diffs, mad


def remove_noise_by_fitting_plane(points3D, colors, axis=0, threshold=1.0, plane_color=[255,255,255]):
    ''' 
    Removes noise from a 3D Point Cloud by fitting a plane to it
    and then removing all points that are not in this plane.

    The plane will be perpendicular to ``axis`` 
    
    Returns the corresponding denoised 3D Point Cloud.
    
    The criteria will be to remove any point which is not situated in the same
    plane as the road, that is, the points whose 'y' value differs significantly
    from the rest.

    threshold = 1.0 is good
    '''

    grid_size = 0.05

    #print("\nRemoving noise from 3D Point Cloud by fitting a plane...")

    if axis == 0: # Plane perpendicular to 'x' axis, which points RIGHT in our world
        
        # For visualization #
        y_min = np.amin(points3D[:,1])
        y_max = np.amax(points3D[:,1])
        z_min = np.amin(points3D[:,2])
        z_max = np.amax(points3D[:,2])
        Y, Z = np.meshgrid(np.arange(y_min, y_max, grid_size), np.arange(z_min, z_max, grid_size))
        YY = Y.flatten()
        ZZ = Z.flatten()

        # 1. We start by fitting a plane to the points and obtaining its coefficients
        #    The planes equation is: --> C[0]*X + C[1]*Y - Z + C[2] = 0 <--
        #    So we solve by least-squares the equation Ax=b
        A = np.c_[ points3D[:,1], points3D[:,2], np.ones(points3D.shape[0]) ]
        b = points3D[:,0]
        C,_,_,_ = scipy.linalg.lstsq(A, b) # coefficients
        
        # For visualization #
        X = C[0]*Y + C[1]*Z + C[2]
        XX = X.flatten()
        plane3D = np.c_[XX, YY, ZZ]
        colors_plane = np.ones(plane3D.shape)*plane_color

        
        # 2. Denoise - For every point in ``points3D``, compute if it belongs to the plane
        a = C[0]*points3D[:, 1] + C[1]*points3D[:, 2] - points3D[:, 0] + C[2]
        inliers_indices = np.where(abs(a) < threshold) # 2.7 good

        # Re-order the coefficients in such a way that the plane equation is
        # C0 * x + C1 * y + C2 * z + C3 = 0
        coefficients = {'Cx': -1.0, 'Cy': C[0], 'Cz': C[1], 'C': C[2]}
        

    elif axis == 1: # Plane perpendicular to 'y' axis, which points UP in our world
        
        # For visualization #
        x_min = np.amin(points3D[:,0])
        x_max = np.amax(points3D[:,0])
        z_min = np.amin(points3D[:,2])
        z_max = np.amax(points3D[:,2])
        X, Z = np.meshgrid(np.arange(x_min, x_max, grid_size), np.arange(z_min, z_max, grid_size))
        XX = X.flatten()
        ZZ = Z.flatten()

        # 1. We start by fitting a plane to the points and obtaining its coefficients
        #    The planes equation is: --> C[0]*X + C[1]*Y - Z + C[2] = 0 <--
        #    So we solve by least-squares the equation Ax=b
        A = np.c_[ points3D[:,0], points3D[:,2], np.ones(points3D.shape[0]) ]
        b = points3D[:,1]
        C,_,_,_ = scipy.linalg.lstsq(A, b) # coefficients
        
        # For visualization #
        Y = C[0]*X + C[1]*Z + C[2]
        YY = Y.flatten()
        plane3D = np.c_[XX, YY, ZZ]
        colors_plane = np.ones(plane3D.shape)*plane_color
        
        # 2. Denoise - For every point in ``points3D``, compute if it belongs to the plane
        a = C[0]*points3D[:, 0] + C[1]*points3D[:, 2] - points3D[:, 1] + C[2]
        inliers_indices = np.where(abs(a) < threshold) # 2.7 good

        # Re-order the coefficients in such a way that the plane equation is
        # C0 * x + C1 * y + C2 * z + C3 = 0
        coefficients = {'Cx': C[0], 'Cy': -1.0, 'Cz': C[1], 'C': C[2]}

    elif axis == 2: # Plane perpendicular to 'z' axis, which points INTO THE SCREEN in our world
        
        # For visualization #
        x_min = np.amin(points3D[:,0])
        x_max = np.amax(points3D[:,0])
        y_min = np.amin(points3D[:,1])
        y_max = np.amax(points3D[:,1])
        X, Y = np.meshgrid(np.arange(x_min, x_max, grid_size), np.arange(y_min, y_max, grid_size))
        XX = X.flatten()
        YY = Y.flatten()

        # 1. We start by fitting a plane to the points and obtaining its coefficients
        #    The planes equation is: --> C[0]*X + C[1]*Y - Z + C[2] = 0 <--
        #    So we solve by least-squares the equation Ax=b
        A = np.c_[ points3D[:,0], points3D[:,1], np.ones(points3D.shape[0]) ]
        b = points3D[:,2]
        C,_,_,_ = scipy.linalg.lstsq(A, b) # coefficients
        
        # For visualization #
        Z = C[0]*X + C[1]*Y + C[2]
        ZZ = Z.flatten()
        plane3D = np.c_[XX, YY, ZZ]
        colors_plane = np.ones(plane3D.shape)*plane_color

        
        # 2. Denoise - For every point in ``points3D``, compute if it belongs to the plane
        a = C[0]*points3D[:, 0] + C[1]*points3D[:, 1] - points3D[:, 2] + C[2]
        inliers_indices = np.where(abs(a) < threshold) # 2.7 good

        # Re-order the coefficients in such a way that the plane equation is
        # C0 * x + C1 * y + C2 * z + C3 = 0
        coefficients = {'Cx': C[0], 'Cy': C[1], 'Cz': -1.0, 'C': C[2]}
    
    
    # Finally, remove noisy points from original 3D Point Cloud (``poins3D``)
    # and remove also corresponding data from the colors array
    points3D = points3D[inliers_indices]
    colors   = colors[inliers_indices]

    return points3D, colors, plane3D, colors_plane, coefficients


def planes_intersection_at_certain_depth(C_p1, C_p2, z):

    # The depth is provided in absolute value. However, in our world, 
    # the 'z' axis points into the screen. This means that increasing negative z values 
    # represent increasing depth values
    z = - z 

    # Now we solve a system of two equations and two variables,
    # since 'z' is known
    # C0*x + C1*y = - (C2*z + C2)
    # K0*x + K1*y = - (K2*z + K2)
    
    #print("Looking for intersection of two planes...")
    
    A = np.matrix([ [C_p1['Cx'], C_p1['Cy'] ], 
                    [C_p2['Cx'], C_p2['Cy'] ]])

    B = np.matrix([ [ - (C_p1['Cz']*z + C_p1['C']) ], 
                    [ - (C_p2['Cz']*z + C_p2['C']) ]])

    A_inverse = np.linalg.inv(A)
    X = A_inverse * B

    point = np.array( [[X[0], X[1], [z]]], np.float64)
    point = np.squeeze(point, axis=2)
    return point


def threshold_complete(points3D, colors, axis, threshold=15.0):
    
    #print("\nMaintain only points whose ``axis`` coordinate is"
    #       "smaller than ``threshold``")
    
    points3D_axis = points3D[:, axis]
    #print(points3D_axis)
    inliers_indices = np.where(abs(points3D_axis) < threshold)
    points3D = points3D[inliers_indices]
    colors = colors[inliers_indices]
    return points3D, colors


def extract_pcls(points3D, colors, axis=0):
    
    #print("\nExtract 2 smaller Point Clouds from ``points3D``")
    
    points3D_axis = points3D[:, axis]
    mean = np.mean(points3D_axis)

    left_indices = np.where(points3D_axis < mean)
    left = points3D[left_indices]
    left_colors = colors[left_indices]

    right_indices = np.where(points3D_axis > mean)
    right = points3D[right_indices]
    right_colors = colors[right_indices]

    return left, left_colors, right, right_colors


def get_end_points_of_road(points3D, depth):
    '''
    Returns the left and right ends of a 3D segment which is perpendicular 
    to the Z axis and situated at a 'z' value of ``depth``, 
    which must be a POSITIVE number.
        * ``depth``: depth at which the segment must be found
    '''

    # Get a numpy array with only the Z coordinates of the input 3D Point Cloud
    points3D_Z = points3D[:, 2]
    # Find the indices of the ``points3D_Z`` whose values are within a 
    # range of the input variable ``depth``
    indices = np.where(( (points3D_Z < -(depth-0.05)) & (points3D_Z > -(depth+0.05)) ))
    # Generate a 3D segment by getting the 3D points of the original
    # input Point Cloud whose 'z' components are situated at a depth of ``depth``
    points3D_Z_segment = points3D[indices]
    # Find the length of the segment
    left_pt_naive, right_pt_naive = get_end_points_of_segment(points3D_Z_segment)

    return left_pt_naive, right_pt_naive


def get_end_points_of_segment(segment):

    #print('\nComputing length of segment...')
    
    ''' Computes the length of a 3D segment '''
    # First, we must find end points of segment, taking into account that the segment
    # has a fixed 'y' and 'z' value, only varying in the 'x' dimension.
    # Consequently, we only take the 'x' components of every 3D point that forms the segment
    segment_X = segment[:, 0]

    if segment_X.size == 0:
        return None, None
        
    # Find the points whose 'x' coordinates are min and max, respectively
    left_end_index  = np.where(segment_X == np.amin(segment_X))
    right_end_index = np.where(segment_X == np.amax(segment_X))
    # Get those points
    left_end_pt  = segment[left_end_index]
    right_end_pt = segment[right_end_index]
     
    return left_end_pt, right_end_pt


def compute_distance_in_3D(pt3D_A, pt3D_B):
    ''' Computes euclidean distance between two 3D points '''
    return  np.linalg.norm(pt3D_A-pt3D_B)


def create_3Dline_from_3Dpoints(left_pt, right_pt, color):
    left_pt[0][1] += 0.01
    right_pt[0][1] += 0.01
    v = right_pt - left_pt
    t_values = np.arange(0.0, 1.0, 0.001)
    line = left_pt
    for t in t_values:
        line = np.append(line, left_pt + (t * v), axis=0)
    colors_line = np.ones(line.shape) * color

    return line, colors_line