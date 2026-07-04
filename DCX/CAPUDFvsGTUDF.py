# -*- coding: utf-8 -*-

import time
import mesh2sdf.core
import torch
import math
import numpy as np
import trimesh
import open3d as o3d
import os
# from V2MUDF.VectorAdam import VectorAdam
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
    def __init__(self,query_func,resolution,threshold_ceil,dest_dir,model_name,
                 iter_max=400, bound_min=None,bound_max=None, 
                 max_batch=100000, learning_rate=0.0005, 
                 warm_up_end=25, report_freq=1):
        self.u = None
        self.mesh = None
        self.device = torch.device('cuda')

        # Evaluating parametersimport plotly as plt
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
        self.model_name = model_name
        self.optimizer = None

        self.query_func = query_func
        # a = 0

    def compute_GTUDF(vertices: np.ndarray, faces: np.ndarray, size: int = 128,
            fix: bool = False, level: float = 0.015, return_mesh: bool = False):
        r''' Converts a input mesh to signed distance field (SDF).

        Args:optimize
            vertices (np.ndarray): The vertices of the input mesh, the vertices MUST be
                in range [-1, 1].
            faces (np.ndarray): The faces of the input mesh.
            size (int): The resolution of the resulting SDF.
            fix (bool): If the input mesh is not watertight, set :attr:`fix` as True.
            level (float): The value used to extract level sets when :attr:`fix` is True,
                with a default value of 0.015 (as a reference 2/128 = 0.015625). And the
                recommended default value is 2/size.
            return_mesh (bool): If True, also return the fixed mesh.
        '''

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

    # def optimize(self):
    #     # height, width = 400, 400

    #     # # 创建棋盘图案
    #     # checkerboard = np.zeros((height, width), dtype=np.uint8)
    #     # box_size = 40
    #     # for i in range(0, height, box_size*2):
    #     #     for j in range(0, width, box_size*2):
    #     #         checkerboard[i:i+box_size, j:j+box_size] = 255
    #     #         checkerboard[i+box_size:i+box_size*2, j+box_size:j+box_size*2] = 255

    #     # # 创建圆形图案
    #     # circle_img = np.zeros((height, width), dtype=np.uint8)
    #     # cv2.circle(circle_img, (width//2, height//2), 150, 255, -1)

    #     # 显示图像

    #     # cv2.imshow('Checkerboard', checkerboard)
    #     # cv2.imshow('Circle', circle_img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     mesh_gt = trimesh.load_mesh(os.path.join("./GTUDF/",self.model_name,"{}.ply".format(self.model_name)))
        
    #     sdf = mesh2sdf.core.compute(mesh_gt.vertices, mesh_gt.faces, 256)
    #     data = np.reshape(np.abs(sdf[128]), (256, 256))
    #     # for x_im in range(count):
    #     #     for y_im in range(count):
    #     #         if data[x_im][y_im] > thres:
    #     #             data[x_im][y_im] = 0.15
    #     # 使用 'viridis' 色彩映射表绘制热图
    #     plt.figure(figsize=(18, 15))
    #     plt.imshow(data, cmap='jet')
    #     plt.colorbar()
    #     plt.axis("off")
    #     plt.savefig("./GTUDF1.jpg")
    #     # plt.title('Simple Heatmap with Viridis Colormap')
    #     plt.show()




    #     query_func = self.query_func

    #     bound = 150 / self.resolution + 0.25
    #     count = self.resolution + 301
    #     # count = 256
    #     base_0 = torch.linspace(-bound, bound, count)

    #     # base_x = base_0.repeat(count).reshape(1, -1)
    #     # base_y = base_0.reshape(-1, 1).repeat(1, count).reshape(1, -1)
    #     # base_z = torch.zeros(1, count * count)
    #     # points_init = torch.cat([base_x, base_z, base_y], 0).t()

    #     base_x = torch.zeros(1,count*count)
    #     base_y = base_0.repeat(count).reshape(1, -1)
    #     base_z = base_0.reshape(-1, 1).repeat(1, count).reshape(1, -1)
    #     points_init = torch.cat([base_x, base_z, base_y], 0).t()

    #     # points_init = torch.cat([base_x, base_z, base_y], 0).t() / 40 + torch.tensor([[-0.395833, 0, -0.0791667]])
    #     points_init.requires_grad = True

    #     dest_dir_txt = self.dest_dir + 'txt/'
    #     dest_dir_ply = self.dest_dir + 'ply/'
    #     if not os.path.exists(dest_dir_txt):
    #         os.makedirs(dest_dir_txt)
    #     if not os.path.exists(dest_dir_ply):
    #         os.makedirs(dest_dir_ply)

    #     batch_size = 50000
    #     head = 0
    #     dist_all = torch.empty(0, dtype=torch.float32)
    #     while head < points_init.shape[0]:
    #         batch = points_init[head:min((head + batch_size), points_init.shape[0])]
    #         dist = query_func(batch).reshape(-1).abs()
    #         dist_all = torch.cat([dist_all, dist])

    #         loss = dist.mean()
    #         loss.backward()

    #         head += batch_size

    #     # dist_all /= 10
    #     grad = torch.nn.functional.normalize(points_init.grad)
    #     print(points_init.grad.shape)
    #     # np.savetxt(dest_dir_txt + "/CAPUDF_dist.txt", dist_all.detach().cpu().numpy())
    #     # np.savetxt(dest_dir_txt + "/CAPUDF_grad.txt", grad.detach().cpu().numpy())



    #     points_init_gt = torch.cat([base_x, base_y, base_z], 0).t()
    #     udf, facet_indices, closest_points = igl.point_mesh_squared_distance(points_init.detach().cpu().numpy(), mesh_gt.vertices, mesh_gt.faces) #This function computes the squared distance, so we need to take the square root
    #     udf = np.sqrt(udf)
    #     # IMPORTANT: the gradients point away from the surface.
    #     udf_grads = points_init.detach().cpu().numpy() - closest_points
    #     udf_grads = torch.Tensor(udf_grads)
    #     udf_grads_normalized = torch.nn.functional.normalize(udf_grads)

    #     # Some query points are exactly on the surface and can produce NaN gradients
    #     # The UDF gradient does not exist on the surface, so here we set it to zero.
    #     udf_grads_normalized = torch.nan_to_num(udf_grads_normalized, nan=0.0)
    #     grad = torch.nan_to_num(grad, nan=0.0)

    #     dist_diff = np.abs(dist_all.detach().cpu().numpy() - udf)

    #     # diff_min = np.min(dist_diff)
    #     # diff_max = np.max(dist_diff)

    #     alpha_diff = np.zeros(0, dtype=np.float32)
    #     # a = torch.dot(grad, udf_grads_normalized)


    #     # for idx in range(count * count):
    #     #     grad_a = grad[idx]
    #     #     grad_b = udf_grads_normalized[idx]
    #     #     dot = torch.dot(grad_a, grad_b).item()
    #     #     if dot > 1.0:
    #     #         dot = 1.0
    #     #     elif dot < -1.0:
    #     #         dot = -1.0
    #     #     alpha = math.acos(dot)
    #     #     if math.isnan(alpha):
    #     #         print()
    #     #     alpha_diff = np.append(alpha_diff, alpha)

    #     # 假设 grad 和 udf_grads_normalized 都是 (N, 3) 的 Tensor 或 Array

    #     # 1. 确保都是 Tensor 且在同一设备
    #     pred_grad = grad # (N, 3)
    #     gt_grad = udf_grads_normalized

    #     # 2. 向量化点积: sum(a * b, dim=1)
    #     # dot shape: (N, )
    #     dot_product = (pred_grad * gt_grad).sum(dim=1)
    #     print(dot_product.shape)
    #     # 3. 截断数值误差 (防止出现 1.0000001 导致 nan)
    #     dot_product = torch.clamp(dot_product, -1.0, 1.0)

    #     # 4. 一次性计算 acos
    #     alpha_diff = torch.acos(dot_product)

    #     # 5. 转回 numpy 用于绘图
    #     alpha_diff = alpha_diff.cpu().numpy()

        
    #     # # # # alpha_diff_gray = (((alpha_diff - alpha_diff.min()) / (alpha_diff.max() - alpha_diff.min())) * 255).astype(np.uint8)
    #     # # # # alpha_diff_gray = np.reshape(alpha_diff_gray, (count, count))
    #     # # # # color_img_alpha = cv2.applyColorMap(alpha_diff_gray, 2)
    #     # # # # cv2.imwrite('./grad_diff.jpg', color_img_alpha)
        
    #     alpha_diff = np.reshape(alpha_diff, (count, count))
    #     plt.figure(figsize=(18, 15))
    #     plt.imshow(alpha_diff, cmap='jet')
    #     plt.colorbar()
    #     plt.axis("off")
    #     # plt.title('Simple Heatmap with Viridis Colormap')
    #     plt.savefig("./grad_diff.jpg")
    #     plt.show()


    #     # thres = 150
    #     # # for x_im in range(count * count):
    #     # #     if udf[x_im] > thres:
    #     # #         dist_diff[x_im] = 0
    #     # for x_im in range(count * count):
    #     #     if udf[x_im] > thres:
    #     #         dist_diff[x_im] = 0distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True)

    #     # data_gray = (((dist_diff - dist_diff.min()) / (dist_diff.max() - dist_diff.min())) * 255).astype(np.uint8)
    #     # data_gray = np.reshape(data_gray, (count, count))
    #     # # cv2.imshow('Uniform Noise', data_gray)
    #     # # cv2.waitKey(0)
    #     # # cv2.imshow("./cv_colormap.jpg", img_gray)
    #     # # colormap = cv2.COLORMAP_JET  # 可以选择其他COLORMAP_*
    #     # color_img = cv2.applyColorMap(data_gray, 2)
    #     # cv2.imwrite('./map.jpg', color_img)
    #     # cv2.waitKey(500)


    #     # 3. 显示结果
    #     # cv2.imshow('Grayscale', data)distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True)
    #     # cv2.imshow('Colormap Applied', color_img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     data = np.reshape(udf, (count, count))
    #     # for x_im in range(count):
    #     #     for y_im in range(count):
    #     #         if data[x_im][y_im] > thres:
    #     #             data[x_im][y_im] = 0.15
    #     # 使用 'viridis' 色彩映射表绘制热图
    #     plt.figure(figsize=(18, 15))
    #     plt.imshow(data, cmap='jet')
    #     plt.colorbar()
    #     plt.axis("off")
    #     plt.savefig("./GTUDF.jpg")
    #     # plt.title('Simple Heatmap with Viridis Colormap')
    #     plt.show()

    #     # data = np.reshape(dist_all.detach().cpu().numpy(), (count, count))
    #     # # 使用 'viridis' 色彩映射表绘制热图
    #     # plt.figure(figsize=(18, 15))
    #     # plt.imshow(data, cmap='jet')
    #     # plt.axis("off")
    #     # plt.colorbar()
    #     # plt.savefig("./CAPUDF.jpg")
    #     # plt.show()

    #     data = np.reshape(dist_diff, (count, count))
    #     # 使用 'viridis' 色彩映射表绘制热图
    #     plt.figure(figsize=(18, 15))
    #     plt.imshow(data, cmap='jet')
    #     plt.colorbar()
    #     plt.axis("off")
    #     plt.savefig("./dist_diff.jpg")
    #     plt.show()


    #     # data = np.reshape(grad_diff, (count, count))
    #     # # 使用 'viridis' 色彩映射表绘制热图
    #     # plt.imshow(data, cmap='jet')
    #     # plt.colorbar(label='Value')
    #     # # plt.title('Simple Heatmap with Viridis Colormap')
    #     # plt.show()
    #     # # plt.savefig("./GTUDF.jpg")


    #     print()

    def optimize(self):
        # height, width = 400, 400

        # # 创建棋盘图案
        # checkerboard = np.zeros((height, width), dtype=np.uint8)
        # box_size = 40
        # for i in range(0, height, box_size*2):
        #     for j in range(0, width, box_size*2):
        #         checkerboard[i:i+box_size, j:j+box_size] = 255
        #         checkerboard[i+box_size:i+box_size*2, j+box_size:j+box_size*2] = 255

        # # 创建圆形图案
        # circle_img = np.zeros((height, width), dtype=np.uint8)
        # cv2.circle(circle_img, (width//2, height//2), 150, 255, -1)

        # 显示图像

        # cv2.imshow('Checkerboard', checkerboard)
        # cv2.imshow('Circle', circle_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        mesh_gt = trimesh.load_mesh(os.path.join("./GTUDF/",self.model_name,"{}.ply".format(self.model_name)))
        
        sdf = mesh2sdf.core.compute(mesh_gt.vertices, mesh_gt.faces, 256)
        data = np.reshape(np.abs(sdf[128]), (256, 256))
        # for x_im in range(count):
        #     for y_im in range(count):
        #         if data[x_im][y_im] > thres:
        #             data[x_im][y_im] = 0.15
        # 使用 'viridis' 色彩映射表绘制热图
        plt.figure(figsize=(18, 15))
        plt.imshow(data, cmap='jet')
        plt.colorbar()
        plt.axis("off")
        plt.savefig("./GTUDF1.jpg")
        # plt.title('Simple Heatmap with Viridis Colormap')
        plt.show()




        query_func = self.query_func

        bound = 150 / self.resolution + 0.5
        count = self.resolution
        # count = 256
        base_0 = torch.linspace(-bound, bound, count)

        # base_x = base_0.repeat(count).reshape(1, -1)
        # base_y = base_0.reshape(-1, 1).repeat(1, count).reshape(1, -1)
        # base_z = torch.zeros(1, count * count)
        # points_init = torch.cat([base_x, base_z, base_y], 0).t()

        base_x = torch.zeros(1,count*count)
        base_y = base_0.repeat(count).reshape(1, -1)
        base_z = base_0.reshape(-1, 1).repeat(1, count).reshape(1, -1)
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
        print(points_init.grad.shape)
        # np.savetxt(dest_dir_txt + "/CAPUDF_dist.txt", dist_all.detach().cpu().numpy())
        # np.savetxt(dest_dir_txt + "/CAPUDF_grad.txt", grad.detach().cpu().numpy())



        points_init_gt = torch.cat([base_x, base_y, base_z], 0).t()
        udf, facet_indices, closest_points = igl.point_mesh_squared_distance(points_init.detach().cpu().numpy(), mesh_gt.vertices, mesh_gt.faces) #This function computes the squared distance, so we need to take the square root
        udf = np.sqrt(udf)
        # IMPORTANT: the gradients point away from the surface.
        udf_grads = points_init.detach().cpu().numpy() - closest_points
        udf_grads = torch.Tensor(udf_grads)
        udf_grads_normalized = torch.nn.functional.normalize(udf_grads)

        # Some query points are exactly on the surface and can produce NaN gradients
        # The UDF gradient does not exist on the surface, so here we set it to zero.
        udf_grads_normalized = torch.nan_to_num(udf_grads_normalized, nan=0.0)
        grad = torch.nan_to_num(grad, nan=0.0)

        dist_diff = np.abs(dist_all.detach().cpu().numpy() - udf)

        # diff_min = np.min(dist_diff)
        # diff_max = np.max(dist_diff)

        alpha_diff = np.zeros(0, dtype=np.float32)
        # a = torch.dot(grad, udf_grads_normalized)


        # for idx in range(count * count):
        #     grad_a = grad[idx]
        #     grad_b = udf_grads_normalized[idx]
        #     dot = torch.dot(grad_a, grad_b).item()
        #     if dot > 1.0:
        #         dot = 1.0
        #     elif dot < -1.0:
        #         dot = -1.0
        #     alpha = math.acos(dot)
        #     if math.isnan(alpha):
        #         print()
        #     alpha_diff = np.append(alpha_diff, alpha)

        # 假设 grad 和 udf_grads_normalized 都是 (N, 3) 的 Tensor 或 Array

        # 1. 确保都是 Tensor 且在同一设备
        pred_grad = grad # (N, 3)
        gt_grad = udf_grads_normalized

        # 2. 向量化点积: sum(a * b, dim=1)
        # dot shape: (N, )
        dot_product = (pred_grad * gt_grad).sum(dim=1)
        print(dot_product.shape)
        # 3. 截断数值误差 (防止出现 1.0000001 导致 nan)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # 4. 一次性计算 acos
        alpha_diff = torch.acos(dot_product)

        # 5. 转回 numpy 用于绘图
        alpha_diff = alpha_diff.cpu().numpy()

        
        # # # # alpha_diff_gray = (((alpha_diff - alpha_diff.min()) / (alpha_diff.max() - alpha_diff.min())) * 255).astype(np.uint8)
        # # # # alpha_diff_gray = np.reshape(alpha_diff_gray, (count, count))
        # # # # color_img_alpha = cv2.applyColorMap(alpha_diff_gray, 2)
        # # # # cv2.imwrite('./grad_diff.jpg', color_img_alpha)
        
        alpha_diff = np.reshape(alpha_diff, (count, count))
        plt.figure(figsize=(18, 15))
        plt.imshow(alpha_diff, cmap='jet')
        plt.colorbar()
        plt.axis("off")
        # plt.title('Simple Heatmap with Viridis Colormap')
        plt.savefig("./grad_diff.jpg")
        plt.show()


        # thres = 150
        # # for x_im in range(count * count):
        # #     if udf[x_im] > thres:
        # #         dist_diff[x_im] = 0
        # for x_im in range(count * count):
        #     if udf[x_im] > thres:
        #         dist_diff[x_im] = 0distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True)

        # data_gray = (((dist_diff - dist_diff.min()) / (dist_diff.max() - dist_diff.min())) * 255).astype(np.uint8)
        # data_gray = np.reshape(data_gray, (count, count))
        # # cv2.imshow('Uniform Noise', data_gray)
        # # cv2.waitKey(0)
        # # cv2.imshow("./cv_colormap.jpg", img_gray)
        # # colormap = cv2.COLORMAP_JET  # 可以选择其他COLORMAP_*
        # color_img = cv2.applyColorMap(data_gray, 2)
        # cv2.imwrite('./map.jpg', color_img)
        # cv2.waitKey(500)


        # 3. 显示结果
        # cv2.imshow('Grayscale', data)distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True)
        # cv2.imshow('Colormap Applied', color_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        data = np.reshape(udf, (count, count))
        # for x_im in range(count):
        #     for y_im in range(count):
        #         if data[x_im][y_im] > thres:
        #             data[x_im][y_im] = 0.15
        # 使用 'viridis' 色彩映射表绘制热图
        plt.figure(figsize=(18, 15))
        plt.imshow(data, cmap='jet')
        plt.colorbar()
        plt.axis("off")
        plt.savefig("./GTUDF.jpg")
        # plt.title('Simple Heatmap with Viridis Colormap')
        plt.show()

        # data = np.reshape(dist_all.detach().cpu().numpy(), (count, count))
        # # 使用 'viridis' 色彩映射表绘制热图
        # plt.figure(figsize=(18, 15))
        # plt.imshow(data, cmap='jet')
        # plt.axis("off")
        # plt.colorbar()
        # plt.savefig("./CAPUDF.jpg")
        # plt.show()

        data = np.reshape(dist_diff, (count, count))
        # 使用 'viridis' 色彩映射表绘制热图
        plt.figure(figsize=(18, 15))
        plt.imshow(data, cmap='jet')
        plt.colorbar()
        plt.axis("off")
        plt.savefig("./dist_diff.jpg")
        plt.show()


        # data = np.reshape(grad_diff, (count, count))
        # # 使用 'viridis' 色彩映射表绘制热图
        # plt.imshow(data, cmap='jet')
        # plt.colorbar(label='Value')
        # # plt.title('Simple Heatmap with Viridis Colormap')
        # plt.show()
        # # plt.savefig("./GTUDF.jpg")


        print()

