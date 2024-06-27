import numpy as np
import open3d as o3d
import pandas as pd
# voxelization
import pyntcloud # use for voxelization, transformation  in pcd, voxels and meshes
                 # Trimesh, PyVista
                 

point_cloud_data = np.load('data\\data_3d_ssc_test\\0002_0000016048.npy')
# points = np.asarray(point_cloud)
print(point_cloud_data.shape)
# ! pcd visiualization

# vis = o3d.visualization.Visualizer()
# vis.create_window(width=800, height=600)

# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

# vis.add_geometry(point_cloud)
# ctr = vis.get_view_control()
# ctr.rotate(180.0, 0.0)
# ctr.set_zoom(0.3)

# vis.run()
# vis.destroy_window()

# ! voxelization
points_df = pd.DataFrame(point_cloud_data, columns=['x', 'y', 'z'])
cloud = pyntcloud.PyntCloud(points_df)
# We use the imported point cloud to create a voxel grid of nxnxn.
voxelgrid_id = cloud.add_structure("voxelgrid", n_x=512, n_y=512, n_z=512)  # voxel size setting
# return <class 'str'>

# We use the calculated occupied voxel grid ids to create the voxel representation of the point cloud
voxelgrid = cloud.structures[voxelgrid_id]
# <class 'pyntcloud.structures.voxelgrid.VoxelGrid'>

# We extract the density feature for each occupied voxel that we will use for coloring the voxels
# the number of points in each voxels
density_feature_vector = voxelgrid.get_feature_vector(mode="density")
# print(density_feature_vector.shape)
# print(density_feature_vector[0,:,:])

# Calculate the maximum density to normalize the colors
max_density = density_feature_vector.max()
# We extract the shape of a voxel, as well as the position of each point in X, Y, Z in the voxel grid
# * so the data from the voxelization it the position of the each vocxels?
voxel_size = voxelgrid.shape
x_cube_pos = voxelgrid.voxel_x
y_cube_pos = voxelgrid.voxel_y
z_cube_pos = voxelgrid.voxel_z
# print("x_cube_pos:", x_cube_pos.shape)
# print(x_cube_pos)
# print("y_cube_pos:", y_cube_pos.shape)
# print("z_cube_pos:", z_cube_pos.shape)


# Initialize a open3d triangle mesh object
vox_mesh = o3d.geometry.TriangleMesh()

# go through all voxelgrid voxels
for idx in range(0, len(voxelgrid.voxel_n)):
    # get the id of the current voxel in the voxel grid
    curr_number = voxelgrid.voxel_n[idx]
    # get the center of the voxel grid voxel
    voxel_center = voxelgrid.voxel_centers[curr_number]
    # get the density value of the current voxel. Because the density matrix is in the shape X,Y,Z, where they are the coordinates in the voxel grid
    # we use the voxel grid positions we already
    curr_density = density_feature_vector[x_cube_pos[idx],y_cube_pos[idx],z_cube_pos[idx]]
    # we normalize the value using the maximum density
    curr_density_normalized = curr_density / max_density
    # create a box primitive in open3d
    primitive=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    # paint the box uniformly using the normalized density
    primitive.paint_uniform_color((curr_density_normalized,curr_density_normalized,curr_density_normalized))
    # scale the cube using the saved voxel size
    primitive.scale(voxel_size[0], center=primitive.get_center())
    # we translate the box to the center position of the voxel
    primitive.translate(voxel_center, relative=True)
    # add to the voxel mesh
    vox_mesh+=primitive

# Initialize a visualizer object
vis = o3d.visualization.Visualizer()
# Create a window, name it and scale it
vis.create_window(window_name='ShapeNet', width=800, height=600)
# add the voxel mesh to the visualizer
vis.add_geometry(vox_mesh)
vis.run()
# Once the visualizer is closed destroy the window and clean up
vis.destroy_window()