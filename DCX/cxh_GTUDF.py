import trimesh
import torch
import torch.nn as nn
# import cc3d
import cubvh
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage.measure import marching_cubes
import os
import numpy as np
import open3d as o3d




class DiffUdfFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bvh, all_verts, all_tris):
        x_flat = x.view(-1, 3).contiguous()

        udf, tri_id, uvw = bvh.unsigned_distance(x_flat, return_uvw=True)  # [N], [N], [N, 3]
        uvw = torch.clamp(uvw, 1e-9, 1-1e-9)
        idx0 = all_tris[tri_id, 0]
        idx1 = all_tris[tri_id, 1]
        idx2 = all_tris[tri_id, 2]

        v0 = all_verts[idx0]
        v1 = all_verts[idx1]
        v2 = all_verts[idx2]

        nearest_points = uvw[:, 0:1] * v0 + uvw[:, 1:2] * v1 + uvw[:, 2:3] * v2

        # # floodfill to get SDF
        # resolution = int(x_flat.shape[0] ** (1 / 3) + 0.5)
        # udf = udf.view(resolution, resolution, resolution).contiguous()
        # occ = udf < 2 / resolution # tolerance 2 voxel
        # floodfill_mask = cubvh.floodfill(occ)
        # empty_label = floodfill_mask[0, 0, 0].item()
        # empty_mask = (floodfill_mask == empty_label)
        # occ_mask = ~empty_mask
        # sdf = udf - 1e-6
        # inner_mask = occ_mask & (sdf > 0)
        # # sdf[inner_mask] *= -1
        # # sdf = -sdf.view(x.shape[:-1])
        # nearest_points[inner_mask.view(-1)] = x_flat[inner_mask.view(-1)]

        ctx.save_for_backward(x_flat, nearest_points)
        return udf.view(x.shape[:-1])[...,None]

    @staticmethod
    def backward(ctx, grad_output):
        x, nearest_points = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            diff = x - nearest_points
            norm = diff.norm(dim=-1, keepdim=True) + 1e-8
            grad_udf = grad_output.view(-1, 1)
            grad_input = grad_udf * (diff / norm)

            grad_input = grad_input.view_as(x)

        return grad_input, None, None, None

class DiffUdfCubvh(nn.Module):
    def __init__(self, bvh, all_verts, all_tris):
        super().__init__()
        self.bvh = bvh
        self.all_verts = all_verts
        self.all_tris = all_tris

    def forward(self, x):
        return DiffUdfFunction.apply(x, self.bvh, self.all_verts, self.all_tris)



def apply_transform(mesh, center, scale):
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(2.0 / scale * 0.5)
    return mesh


def normalize_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, process=False, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        scene = trimesh.load(mesh_path, process=False, force='scene')
        meshes = []
        for node_name in scene.graph.nodes_geometry:
            geom_name = scene.graph[node_name][1]
            geometry = scene.geometry[geom_name]
            transform = scene.graph[node_name][0]
            if isinstance(geometry, trimesh.Trimesh):
                geometry.apply_transform(transform)
                meshes.append(geometry)

        mesh = trimesh.util.concatenate(meshes)

    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    mesh.apply_scale(2.0 / scale * 0.45)

    return mesh, center, scale


def extract_mesh(BVH, resolution):
    
    N = 32
    X = torch.linspace(-0.5, 0.5, resolution).split(N)
    Y = torch.linspace(-0.5, 0.5, resolution).split(N)
    Z = torch.linspace(-0.5, 0.5, resolution).split(N)

    udf = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                sub_udf, _, _ = BVH.unsigned_distance(pts, return_uvw=False)
                val = sub_udf.detach().cpu().numpy().reshape(len(xs), len(ys), len(zs))
                udf[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

    # query dense UDF
    # udf, _, _ = BVH.unsigned_distance(grid_points.view(-1, 3), return_uvw=False)


    # floodfill to get SDF
    # occ = ~(udf < 2 / resolution)
    occ = udf < 2 / resolution # tolerance 2 voxel
    occ = torch.from_numpy(occ).cuda()
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    occ_mask = ~empty_mask
    # empty_label = labels[0, 0, 0]
    # mask = (labels == empty_label)
    # occ_mask = ~mask
    occ_mask = occ_mask.cpu().numpy()
    plt.imsave("occ.png", occ_mask[:, :, resolution // 2])


    sdf = udf
    sdf[occ_mask] *= -1
    verts, faces, normals, values = marching_cubes(sdf, 0,spacing=(1/resolution,1/resolution,1/resolution))
    verts = verts - 0.5 + (0.5/(resolution-1))
    verts = verts * (resolution/(resolution-1))
    mesh = trimesh.Trimesh(verts,faces)
    return mesh
