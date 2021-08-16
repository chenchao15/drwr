import numpy as np


def evaluate_voxel_prediction(preds, gt, thresh):
    preds_occupy = preds[:, 1, :, :] >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt[:, 1, :, :]))
    intersection = np.sum(np.logical_and(preds_occupy, gt[:, 1, :, :]))
    union = np.sum(np.logical_or(preds_occupy, gt[:, 1, :, :]))
    num_fp = np.sum(np.logical_and(preds_occupy, gt[:, 0, :, :]))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt[:, 1, :, :]))  # false negative
    return np.array([diff, intersection, union, num_fp, num_fn])


def voxel2mesh(voxels, surface_view):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > 0.3)
    voxels[positions] = 1 
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face 
        if not surface_view or np.sum(voxels[i-1:i+2, j-1:j+2, k-1:k+2]) < 27:
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)  
              
    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, surface_view = True):
    verts, faces = voxel2mesh(pred, surface_view)
    write_obj(filename, verts, faces)


def voxel2pc(voxels, threshold):
    voxels = np.squeeze(voxels)
    vox = voxels > threshold
    vox = np.squeeze(vox)
    vox_size = vox.shape[0]

    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-0.5, 0.5, vox_size)
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, x, x)
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(mesh_z, -1)

    occupancies = np.reshape(vox, -1)
    xyz = xyz[occupancies, :]
    return xyz, occupancies


def augment_mesh(verts, faces):
    new_points = np.zeros((0, 3), np.float32)
    for k1, k2 in [(0, 1), (1, 2), (0, 2)]:
        i1 = faces[:, k1]
        i2 = faces[:, k2]
        pts = 0.5*(verts[i1, :] + verts[i2, :])
        new_points = np.concatenate((new_points, pts), axis=0)
    return np.concatenate((verts, new_points), axis=0)


def extract_surface(voxels, iso_level, dense=False):
    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxels, iso_level)
    if dense:
        return augment_mesh(verts, faces)
    else:
        return verts
