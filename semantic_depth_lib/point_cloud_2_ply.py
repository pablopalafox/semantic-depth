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
Point cloud class generated from two numpy arrays, one being
the actual points and one the points' colors, which then saves theses
points into a ply file

Classes:

    * ``PointCloud2Ply`` - Point cloud with RGB colors

'''

import numpy as np

class PointCloud2Ply():

    """3D point cloud tools."""

    #: Header for exporting point cloud to PLY
    ply_header = (
    '''ply
    format ascii 1.0
    element vertex {vertex_count}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    ''')

    def __init__(self, points3D, colors, output_name):
        """
        Initialize point cloud with given coordinates and associated colors.

        ``coordinates`` and ``colors`` should be numpy arrays of the same
        length, in which ``coordinates`` is made of three-dimensional point
        positions (X, Y, Z) and ``colors`` is made of three-dimensional spectral
        data, e.g. (R, G, B).
        """
        self.points3D = points3D.reshape(-1, 3)
        self.colors = colors.reshape(-1, 3)
        self.output_name = output_name

    def write_ply(self, output_file):
        """Export ``PointCloud`` to PLY file for viewing in MeshLab."""
        points = np.hstack([self.points3D, self.colors])
        try:
            with open(output_file, 'w') as f:
                f.write(self.ply_header.format(vertex_count=len(points)))
                np.savetxt(f, points, '%f %f %f %d %d %d')
                print("Point Cloud file generated!")
        except Exception as e:
            raise

    def add_extra_point_cloud(self, points3D_extra, colors_extra):
        """
        If set, append the ``points3D_extra`` vector to the already existign 
        ``points3D`` (do the same with the colors)
        """
        self.points3D = np.append(self.points3D, points3D_extra, axis=0)
        self.colors   = np.append(self.colors, colors_extra, axis=0)

    def prepare_and_save_point_cloud(self):
        """
        Apply an inifitiy filter and, finally, save the points into a ply file.
        """
        # Apply a mask to remove points with an infinite depth
        infinity_mask = self.points3D[:, 2] > self.points3D[:, 2].min()
        self.points3D = self.points3D[infinity_mask]
        self.colors = self.colors[infinity_mask]

        output_ply = '{}.ply'.format(self.output_name)
        self.write_ply(output_ply)
