# -*- coding: utf-8 -*-

import time
import mesh2sdf.core
import torch
import math
import numpy as np
import trimesh
import open3d as o3d
import os
from V2MUDF.VectorAdam import VectorAdam
import warnings
import igl
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import mesh2sdf
import skimage
warnings.filterwarnings('ignore')


class V2MUDF:
    """SPUDF mesh extraction

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            query_func:       differentiable function, input tensor[batch_size, 3] and output tensor[batch_size]
            resolution:
            threshold_ceil:
            max_iter:
            normal_step:      end of step one
            laplacian_weight:
            bound_min:
            bound_max:
            is_cut:           if model is a open model, set to True to cut double cover
            region_rate:      region of seed and sink in mini-cut
            max_batch:        higher batch_size will have quicker speed. If you GPU memory is not enough, decrease it.
            learning_rate:
            warm_up_end:
            report_freq:      report loss every {report_freq}

        """
    def __init__(self,query_func,resolution,threshold_ceil,dest_dir,
                 iter_max=400, bound_min=None,bound_max=None, 
                 max_batch=100000, learning_rate=0.0005, 
                 warm_up_end=25, report_freq=1):
        self.u = None
        self.mesh = None
        self.device = torch.device('cuda')

        # Evaluating parameters
        # self.iter_1st = iter_1st
        # self.iter_2nd = iter_2nd
        self.iter_max = iter_max
        self.max_batch = max_batch
        self.report_freq = report_freq
        self.warm_up_end = warm_up_end
        self.learning_rate = learning_rate
        self.resolution = 1024
        # self.resolution = resolution
        self.threshold_ceil = threshold_ceil
        self.dest_dir = dest_dir

        self.optimizer = None

        self.query_func = query_func
        # a = 0

    def compute_GTUDF(vertices: np.ndarray, faces: np.ndarray, size: int = 128,
            fix: bool = False, level: float = 0.015, return_mesh: bool = False):

        # compute sdf
        sdf = mesh2sdf.core.compute(vertices, faces, size)
        if not fix:
            return (sdf, trimesh.Trimesh(vertices, faces)) if return_mesh else sdf

        # NOTE: the negative value is not reliable if the mesh is not watertight
        sdf = np.abs(sdf)
        vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

        # keep the max component of the extracted mesh
        mesh = trimesh.Trimesh(vertices, faces)
        components = mesh.split(only_watertight=False)
        bbox = []
        for c in components:
            bbmin = c.vertices.min(0)
            bbmax = c.vertices.max(0)
            bbox.append((bbmax - bbmin).max())
        max_component = np.argmax(bbox)
        mesh = components[max_component]
        mesh.vertices = mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]

        # re-compute sdf
        sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, size)
        return (sdf, mesh) if return_mesh else sdf

    def optimize(self):
        query_func = self.query_func

        bound = 150 / self.resolution + 0.5
        count = self.resolution + 301
        # count = 256
        base_0 = torch.linspace(-bound, bound, count)
        base_x = base_0.repeat(count).reshape(1, -1)
        base_y = base_0.reshape(-1, 1).repeat(1, count).reshape(1, -1)
        base_z = torch.zeros(1, count * count)
        points_init = torch.cat([base_x, base_z, base_y], 0).t()
        # points_init = torch.cat([base_x, base_z, base_y], 0).t() / 40 + torch.tensor([[-0.395833, 0, -0.0791667]])
        points_init.requires_grad = True

        dest_dir_txt = self.dest_dir + 'txt/'
        dest_dir_ply = self.dest_dir + 'ply/'
        if not os.path.exists(dest_dir_txt):
            os.makedirs(dest_dir_txt)
        if not os.path.exists(dest_dir_ply):
            os.makedirs(dest_dir_ply)

        batch_size = 50000
        head = 0
        dist_all = torch.empty(0, dtype=torch.float32)
        while head < points_init.shape[0]:
            batch = points_init[head:min((head + batch_size), points_init.shape[0])]
            dist = query_func(batch).reshape(-1).abs()
            dist_all = torch.cat([dist_all, dist])

            loss = dist.mean()
            loss.backward()

            head += batch_size

        # dist_all /= 10
        grad = torch.nn.functional.normalize(points_init.grad)

        alpha_diff = np.zeros(0, dtype=np.float32)
        # a = torch.dot(grad, udf_grads_normalized)
        for idx in range(count * count):
            grad_a = grad[idx]
            grad_b = udf_grads_normalized[idx]
            dot = torch.dot(grad_a, grad_b).item()
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            alpha = math.acos(dot)
            if math.isnan(alpha):
                print()
            alpha_diff = np.append(alpha_diff, alpha)
        
        # # # # alpha_diff_gray = (((alpha_diff - alpha_diff.min()) / (alpha_diff.max() - alpha_diff.min())) * 255).astype(np.uint8)
        # # # # alpha_diff_gray = np.reshape(alpha_diff_gray, (count, count))
        # # # # color_img_alpha = cv2.applyColorMap(alpha_diff_gray, 2)
        # # # # cv2.imwrite('./grad_diff.jpg', color_img_alpha)
        


        print()


