# -*- coding: utf-8 -*-

import time
import torch
import math
import numpy as np
import trimesh
import open3d as o3d
import os
from DCX.VectorAdam import VectorAdam
import warnings
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 32
    X = torch.linspace(bound_min, bound_max, resolution).split(N)
    Y = torch.linspace(bound_min, bound_max, resolution).split(N)
    Z = torch.linspace(bound_min, bound_max, resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()

                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

    return u


def laplacian_calculation(mesh, equal_weight=True):

    neighbors = mesh.vertex_neighbors
    vertices = mesh.vertices.view(np.ndarray)
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])

    if equal_weight:
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        ones = np.ones(3)
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        data = np.concatenate([i / i.sum() for i in norms])

    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def laplacian_step(laplacian_op,samples):
    laplacian_v = torch.sparse.mm(laplacian_op, samples[:, 0:3]) - samples[:, 0:3]
    return laplacian_v


def get_abc(vertices, faces):
    fvs = vertices[faces]
    sub_a = fvs[:, 0, :] - fvs[:, 1, :]
    sub_b = fvs[:, 1, :] - fvs[:, 2, :]
    sub_c = fvs[:, 0, :] - fvs[:, 2, :]
    sub_a = torch.linalg.norm(sub_a, dim=1)
    sub_b = torch.linalg.norm(sub_b, dim=1)
    sub_c = torch.linalg.norm(sub_c, dim=1)
    return sub_a, sub_b, sub_c


def calculate_s(vertices, faces):
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    p = (sub_a + sub_b + sub_c)/2

    s = p*(p-sub_a)*(p-sub_b)*(p-sub_c)
    s[s<1e-30]=1e-30

    sqrts = torch.sqrt(s)
    return sqrts


def get_mid(vertices, faces):
    fvs = vertices[faces]
    re = torch.mean(fvs,dim=1)
    return re


def get_aver(distances, face):
    return (distances[face[0]] + distances[face[1]] + distances[face[2]]) / 3.0


def remove_far(gt_pts, mesh, dis_trunc=0.0125, is_use_prj=False):
    
    gt_kd_tree = cKDTree(gt_pts)
    distances, vertex_ids = gt_kd_tree.query(mesh.vertices, p=2, distance_upper_bound=dis_trunc)
    faces_remaining = []
    faces = mesh.faces

    if is_use_prj:
        normals = gt_pts.vertex_normals
        closest_points = gt_pts.vertices[vertex_ids]
        closest_normals = normals[vertex_ids]
        direction_from_surface = mesh.vertices - closest_points
        distances = direction_from_surface * closest_normals
        distances = np.sum(distances, axis=1)

    for i in range(faces.shape[0]):
        if get_aver(distances, faces[i]) < dis_trunc:
            faces_remaining.append(faces[i])
    mesh_cleaned = mesh.copy()
    mesh_cleaned.faces = faces_remaining
    mesh_cleaned.remove_unreferenced_vertices()

    return mesh_cleaned


class DCX:
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

        # Evaluating parameters
        # self.iter_1st = iter_1st
        # self.iter_2nd = iter_2nd
        self.iter_max = iter_max
        self.iter_max_vice = iter_max - 150
        self.max_batch = max_batch
        self.report_freq = report_freq
        self.report_freq_vice = 20
        self.warm_up_end = warm_up_end
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.threshold_ceil = threshold_ceil
        self.dest_dir = dest_dir
        self.model_name = model_name

        self.optimizer = None

        self.query_func = query_func

    def filter_query_points_for_filling_holes(self, query_points_numpy):
        query_func = self.query_func
        query_points_numpy = np.asarray(query_points_numpy, dtype=np.float32).reshape(-1, 3)
        if query_points_numpy.shape[0] == 0:
            print("Supply-sampling finished, 0 points were supplied!")
            return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float32)
        query_points = torch.from_numpy(query_points_numpy).float().cuda()
        threshold = 0.707 / self.resolution

        cnt = 7 ** 3
        batch_size = 500 * cnt
        head = 0
        query_points.requires_grad_(False)
        dist_min_all = torch.zeros(0, dtype=torch.float32).cuda()
        dist_min_points_indexes_all = torch.zeros(0, dtype=torch.int).cuda()
        # dist_min_points_all = torch.zeros(0, 3, dtype=torch.float32).cuda()
        index_dim_x_offset = torch.linspace(0, 499, 500, dtype=torch.int).cuda() * cnt
        i = 0
        while head < query_points.shape[0]:
            end = min((head + batch_size), query_points.shape[0])
            batch = query_points[head:end]
            dist = query_func(batch).reshape(-1).abs().detach().reshape(-1, cnt)
            dist_min, dist_min_indexes = torch.min(dist, 1)
            dist_min_points_indexes = i * batch_size + index_dim_x_offset[0:math.floor((end-head)/cnt)] + dist_min_indexes
            i += 1
            dist_min_all = torch.cat([dist_min_all, dist_min.detach()])
            dist_min_points_indexes_all = torch.cat([dist_min_points_indexes_all, dist_min_points_indexes])
            head += batch_size

        need_indexes = torch.nonzero(dist_min_all < threshold).reshape(-1)
        dist_min_points = torch.index_select(query_points, 0, dist_min_points_indexes_all)
        need_points = torch.index_select(dist_min_points, 0, need_indexes)
        print("Supply-sampling finished, {} points were supplied!".format(need_indexes.shape[0]))
        return need_indexes.detach().cpu().numpy().astype(np.int64), need_points.detach().cpu().numpy().astype(np.float32)

    def optimize(self):
        query_func = self.query_func

        # ######test 
        # u = extract_fields(-0.5, 0.5, 256, query_func)
        # model = self.dest_dir.split('/')[-2]
        # np.save("./{}_dist.npy".format(model), u)
        # ######test

        points_final_all = np.zeros([0, 3], dtype=np.float32)

        bound = 10 / self.resolution + 0.5
        count = self.resolution + 21
        # bound = 10 / self.resolution + 1
        # count = 1024 + 20
        base_0 = torch.linspace(-bound + 0.5 / self.resolution, bound - 0.5 / self.resolution, count)
        base_x = base_0.repeat(count * count).reshape(1, -1)
        base_y = base_0.reshape(-1, 1).repeat(count, count).reshape(1, -1)
        base_z = base_0.reshape(-1, 1).repeat(1, count*count).reshape(1, -1)
        points_init = torch.cat([base_x, base_y, base_z], 0).t()
        dest_dir_txt = self.dest_dir + 'txt/'
        dest_dir_ply = self.dest_dir + 'ply/'
        if not os.path.exists(dest_dir_txt):
            os.makedirs(dest_dir_txt)
        if not os.path.exists(dest_dir_ply):
            os.makedirs(dest_dir_ply)

        # # 加速一下，先用大刻度大阈值点云查询，阈值之内的点生成周围的刻度更小的点换小阈值再判断 todo!!!
        # point_init_selected = points_init[(points_init[:, 2] > 0.5 - 1 / 128) & ((points_init[:, 0] < 1 / 128) & (points_init[:, 0] > -1 / 128))]
        # points_init_mesh = trimesh.Trimesh(vertices=point_init_selected.detach().cpu().numpy())
        # points_init_mesh.export(dest_dir_ply + "/points_cubes.ply")

        # del points_init_mesh, base_0, base_x, base_y, base_z

        # gt_pts_mesh = trimesh.load("/home/kylin-h20/Desktop/Others/experiment/CAPUDF/SIGA_batch_1/" + self.model_name + "/pcd_" + self.model_name + ".ply")
        gt_pts_mesh = trimesh.load("./ckpt/" + self.model_name[3:] + "/pcd_" + self.model_name[3:] + ".ply")
        gt_kdtree = cKDTree(gt_pts_mesh.vertices)

        distance, vertex_ids = gt_kdtree.query(points_init.detach().cpu().numpy(), p=2, distance_upper_bound=1.5/128)

        points_raw_np = points_init.detach().cpu().numpy()[distance<1.5/128]
        out_mesh = trimesh.Trimesh(vertices=points_raw_np)
        out_mesh.export("./points_raw.ply")
        points_init = torch.from_numpy(points_raw_np).cuda()

        batch_size = 50000
        # threshold_ceil = 1.5 / 256
        threshold_floor = 1.5 / 1024
        # gap = 1 / 512points_raw
        gap = 1 / 256
        head = 0
        indexes = torch.empty(0, dtype=torch.int32)
        indexes_0 = torch.empty(0, dtype=torch.int32)
        while head < points_init.shape[0]:
            batch = points_init[head:min((head + batch_size), points_init.shape[0])]
            dist = query_func(batch).reshape(-1).abs()
            # index = torch.nonzero((dist < (gap + (1 / self.resolution) / 2)) & (dist > (gap - (1 / self.resolution) / 2))).reshape(-1)
            # index = torch.nonzero(dist <= (gap + (1 / self.resolution) / 2)).reshape(-1)
            index = torch.nonzero((dist <= (2 * gap)) & (dist != 0)).reshape(-1)
            indexes = torch.cat([indexes, head + index])
            index = torch.nonzero(dist == 0).reshape(-1)
            indexes_0 = torch.cat([indexes_0, head + index])
            
            # a = dist[dist < 0].detach().cpu().numpy()
            # if a.shape[0] > 0:
            #     dist_temp = np.append(dist_temp, values=a, axis=0)
            head += batch_size


        # # Get gradients of points_raw
        points_raw = torch.index_select(points_init, 0, indexes).detach()
        points_dist_zero = torch.index_select(points_init, 0, indexes_0).detach()
        points_dist_0_mesh = trimesh.Trimesh(vertices=points_dist_zero.detach().cpu().numpy())
        # points_final_snap_to_half_grid = points_dist_zero.clone()
        points_dist_0_mesh.export(dest_dir_ply + "/points_raw_dist_0.ply")
        # np.savetxt(dest_dir_txt + "/vertices_dist_0.txt", points_dist_0_mesh.vertices)
        points_final_all = np.concatenate((points_final_all, points_dist_0_mesh.vertices), axis=0)
        del points_init, points_dist_0_mesh, indexes_0
        points_raw.requires_grad = True
        head = 0
        while head < points_raw.shape[0]:
            batch = points_raw[head:min((head + batch_size), points_raw.shape[0])]
            dist = query_func(batch).reshape(-1)

            loss = dist.mean()
            loss.backward()

            head += batch_size

        grad = torch.nn.functional.normalize(points_raw.grad)

        
        dist_last = torch.empty(0, dtype=torch.float32)

        # # ## ablation, direct optimizing
        # # points_optimized = points_raw
        # # points_optimized.requires_grad = True
        # # self.optimizer = VectorAdam([points_optimized])
        # # for iter in range(self.iter_max):
        # #     self.update_learning_rate(iter, 0.025)
        # #     epoch_loss = 0
        # #     self.optimizer.zero_grad()
        # #     head = 0
            
        # #     while head < points_raw.shape[0]:
        # #         end = min((head + batch_size), points_raw.shape[0])
        # #         batch = points_optimized[head:end]
        # #         # batch = torchquery_func.cat([loca, points_raw[head:end, 1], points_raw[head:end, 2]]).reshape(3, -1).t()
        # #         dist = query_func(batch)

        # #         loss = dist.abs().mean()
        # #         epoch_loss += loss.data
        # #         loss.backward()

        # #         head += batch_squery_funcize

        # #     self.optimizer.step()
        # #     if iter == 0 or iter % 50 == 49:
        # #         print("    {} epoch in 1st stage finished, loss = {}!".format(iter, epoch_loss/math.ceil(points_raw.shape[0]/batch_size)))

        print("Starting evaluating!")
        alpha = torch.zeros(points_raw.shape[0], dtype=torch.float32)
        alpha.requires_grad = True
        self.optimizer = VectorAdam([alpha])
        for iter in range(self.iter_max):
            # # lr for NUDF
            # self.update_learning_rate(iter, 1.5)

            # lr for GTUDF
            self.update_learning_rate(iter, 3)
            epoch_loss = 0
            self.optimizer.zero_grad()
            head = 0
            
            while head < points_raw.shape[0]:
                end = min((head + batch_size), points_raw.shape[0])
                batch = alpha[head:end].unsqueeze(1).expand(-1, 3) * grad[head:end] + points_raw[head:end]
                # batch = torch.cat([loca, points_raw[head:end, 1], points_raw[head:end, 2]]).reshape(3, -1).t()
                dist = query_func(batch)
                if iter == (self.iter_max - 1):
                    dist_last = torch.cat([dist_last, dist])

                loss = dist.mean()
                epoch_loss += loss.data
                loss.backward()

                head += batch_size

            self.optimizer.step()
            if iter == 0 or iter % 50 == 49:
                print("    {} epoch in 1st stage finished, loss = {}!".format(iter, epoch_loss/math.ceil(points_raw.shape[0]/batch_size)))
        
        points_optimized = points_raw + alpha.unsqueeze(1).expand(-1, 3) * grad
        points_optimized_mesh = trimesh.Trimesh(vertices=points_optimized.detach().cpu().numpy())
        points_optimized_mesh.export(dest_dir_ply + "/new_scheme_points_optimized.ply")

        temp_all = np.concatenate((points_optimized_mesh.vertices, points_final_all), axis=0)
        np.savetxt(dest_dir_txt + "/new_scheme_points_final_all_wo_selecting.txt", temp_all)
        
        # # # # final selection
        dist_last = dist_last.reshape(-1)
        # indexes = torch.nonzero(dist_last < (1.25/1024)).reshape(-1)
        # indexes = torch.nonzero(dist_last < (1.5/1024)).reshape(-1)
        indexes = torch.nonzero(dist_last < (1/256)).reshape(-1)
        points_optimized_mesh = trimesh.Trimesh(vertices=torch.index_select(points_optimized, 0, indexes).detach().cpu().numpy())


        # # Selecting according distances between result after optimizing and input pointclouds.
        # distance, vertex_ids = gt_kdtree.query(points_optimized_mesh.vertices, p=2, distance_upper_bound=1.5/128)

        # points_raw_np = points_optimized_mesh.vertices[distance<1.5/128]
        # out_mesh = trimesh.Trimesh(vertices=points_raw_np)
        # out_mesh.export("/home/kylin-h20/optimized_points_with_ckdtree.ply")


        points_optimized_x_select = torch.from_numpy(points_optimized_mesh.vertices).cuda().float()
        points_optimized_x_select.requires_grad = True
        head = 0
        dist_last = torch.zeros(0).cuda()
        while head < points_optimized_x_select.shape[0]:
            batch = points_optimized_x_select[head:min((head + batch_size), points_optimized_x_select.shape[0])]
            dist = query_func(batch).reshape(-1)
            dist_last = torch.cat([dist_last, dist])

            loss = dist.mean()
            loss.backward()

            head += batch_size

        grad = torch.nn.functional.normalize(points_optimized_x_select.grad)
        points_optimized_mesh.vertex_normals = grad.detach().cpu().numpy()
        points_final_all = np.concatenate((points_final_all, points_optimized_mesh.vertices), axis=0)

        global_bounds = np.zeros([2, 3], dtype=np.float32)
        # global_bounds[0][0] = points_final_all[:, 0].min()
        # global_bounds[0][1] = points_final_all[:, 1].min()
        # global_bounds[0][2] = points_final_all[:, 2].min()
        # global_bounds[1][0] = points_final_all[:, 0].max()
        # global_bounds[1][1] = points_final_all[:, 1].max()
        # global_bounds[1][2] = poin.abs()ts_final_all[:, 2].max()
        global_bounds[0][0] = temp_all[:, 0].min()
        global_bounds[0][1] = temp_all[:, 1].min()
        global_bounds[0][2] = temp_all[:, 2].min()
        global_bounds[1][0] = temp_all[:, 0].max()
        global_bounds[1][1] = temp_all[:, 1].max()
        global_bounds[1][2] = temp_all[:, 2].max()
        return points_final_all, global_bounds.reshape(-1)



    def optimize1(self):
        query_func = self.query_func
        points_final_all = np.zeros([0, 3], dtype=np.float32)
        bound = 10 / self.resolution + 0.5
        count = self.resolution + 21

        base_0 = torch.linspace(-bound + 0.5 / self.resolution, 
                                bound - 0.5 / self.resolution, 
                                count, device='cpu')
        points_candidate_list = []
        total_points = count ** 3
        query_batch_size = self.max_batch
        dist_upper_bound = 3.0 / self.resolution

        s1 = time.time()
        for head in range(0, total_points, query_batch_size):
            tail = min(head + query_batch_size, total_points)
            
            indices = torch.arange(head, tail, device='cpu', dtype=torch.long)

            idx_x = indices % count
            idx_y = (indices // count) % count
            idx_z = indices // (count * count)

            batch_pts = torch.stack([base_0[idx_x], base_0[idx_y], base_0[idx_z]], dim=1)
            with torch.no_grad():
                distance = query_func(batch_pts.cuda().float()).reshape(-1).abs()
            mask = distance < dist_upper_bound
            
            if torch.any(mask):
                points_candidate_list.append(batch_pts[mask.cpu()].numpy())

        if len(points_candidate_list) == 0:
            print("No points found within query_func distance threshold!")
            empty_points = np.zeros((0, 3), dtype=np.float32)
            empty_bounds = np.zeros((2, 3), dtype=np.float32)
            return empty_points, empty_bounds
        points_raw_np = np.concatenate(points_candidate_list, axis=0)
        del points_candidate_list
        
        print(f"query_func filtered: {points_raw_np.shape[0]} points remain.")
        e1 = time.time()
        print(f"query_func time: {e1 - s1:.5f} seconds")
        points_init = torch.from_numpy(points_raw_np).cuda().float()

        batch_size = 50000
        gap = 1 / self.resolution
        head = 0
        indexes = torch.empty(0, dtype=torch.int32)
        indexes_0 = torch.empty(0, dtype=torch.int32)
        while head < points_init.shape[0]:
            batch = points_init[head:min((head + batch_size), points_init.shape[0])]
            dist = query_func(batch).reshape(-1).abs()
            index = torch.nonzero((dist <= (2.0 * gap)) & (dist != 0)).reshape(-1)
            indexes = torch.cat([indexes, head + index])
            index = torch.nonzero(dist == 0).reshape(-1)
            indexes_0 = torch.cat([indexes_0, head + index])
            head += batch_size

        points_raw = torch.index_select(points_init, 0, indexes).detach()
        points_dist_zero = torch.index_select(points_init, 0, indexes_0).detach()
        points_dist_0_mesh = trimesh.Trimesh(vertices=points_dist_zero.detach().cpu().numpy())
        points_final_all = np.concatenate((points_final_all, points_dist_0_mesh.vertices), axis=0)
        del points_init, points_dist_0_mesh, indexes_0

        points_raw.requires_grad = True
        head = 0
        while head < points_raw.shape[0]:
            batch = points_raw[head:min((head + batch_size), points_raw.shape[0])]
            dist = query_func(batch).reshape(-1)
            loss = dist.mean()
            loss.backward()
            head += batch_size

        grad = torch.nn.functional.normalize(points_raw.grad)
        dist_last = torch.empty(0, dtype=torch.float32)

        print("Starting evaluating!")
        s_project = time.time()
        alpha = torch.zeros(points_raw.shape[0], dtype=torch.float32)
        alpha.requires_grad = True
        self.optimizer = VectorAdam([alpha])
        for iter in range(self.iter_max):
            # # lr for NUDF
            self.update_learning_rate(iter, 1.5)

            # lr for GTUDF
            # self.update_learning_rate(iter, 3)
            epoch_loss = 0
            self.optimizer.zero_grad()
            head = 0
            while head < points_raw.shape[0]:
                end = min((head + batch_size), points_raw.shape[0])
                batch = alpha[head:end].unsqueeze(1).expand(-1, 3) * grad[head:end] + points_raw[head:end]
                dist = query_func(batch)
                if iter == (self.iter_max - 1):
                    dist_last = torch.cat([dist_last, dist])

                loss = dist.mean()
                epoch_loss += loss.data
                loss.backward()

                head += batch_size
            self.optimizer.step()
            if iter == 0 or iter % 50 == 49:
                print("    {} epoch in 1st stage finished, loss = {}!".format(iter, epoch_loss/math.ceil(points_raw.shape[0]/batch_size)))
        e_project = time.time()
        print(f"Projection time: {e_project - s_project:.5f} seconds")
        points_optimized = points_raw + alpha.unsqueeze(1).expand(-1, 3) * grad

        dist_last = dist_last.reshape(-1)
        indexes = torch.nonzero(dist_last < (1.0/self.resolution)).reshape(-1)
        points_optimized_mesh = trimesh.Trimesh(vertices=torch.index_select(points_optimized, 0, indexes).detach().cpu().numpy())
        # points_optimized_mesh = trimesh.Trimesh(vertices=points_optimized.cpu().detach().numpy())
        points_optimized_x_select = torch.from_numpy(points_optimized_mesh.vertices).cuda().float()
        points_optimized_x_select.requires_grad = True
        head = 0
        dist_last = torch.zeros(0).cuda()
        while head < points_optimized_x_select.shape[0]:
            batch = points_optimized_x_select[head:min((head + batch_size), points_optimized_x_select.shape[0])]
            dist = query_func(batch).reshape(-1)
            dist_last = torch.cat([dist_last, dist])
            loss = dist.mean()
            loss.backward()
            head += batch_size

        grad = torch.nn.functional.normalize(points_optimized_x_select.grad)
        points_optimized_mesh.vertex_normals = grad.detach().cpu().numpy()
        points_final_all = np.concatenate((points_final_all, points_optimized_mesh.vertices), axis=0)

        if points_final_all.shape[0] == 0:
            global_bounds = np.zeros((2, 3), dtype=np.float32)
        else:
            global_bounds = np.stack(
                [points_final_all.min(axis=0), points_final_all.max(axis=0)],
                axis=0,
            ).astype(np.float32)

        print("Finished evaluating!")
        return points_final_all, global_bounds

    def optimize2_true(self,dataset_dir):
        query_func = self.query_func
        points_final_all = np.zeros([0, 3], dtype=np.float32)

        # # old method which uses too much memory
        # bound = 10 / self.resolution + 0.5
        # count = self.resolution + 21
        # # bound = 10 / self.resolution + 1
        # # count = 1024 + 20
        # base_0 = torch.linspace(-bound + 0.5 / self.resolution, bound - 0.5 / self.resolution, count)
        # base_x = base_0.repeat(count * count).reshape(1, -1)
        # base_y = base_0.reshape(-1, 1).repeat(count, count).reshape(1, -1)
        # base_z = base_0.reshape(-1, 1).repeat(1, count*count).reshape(1, -1)
        # points_init = torch.cat([base_x, base_y, base_z], 0).t()


        # # gt_pts_mesh = trimesh.load("/home/kylin-h20/Desktop/Others/experiment/CAPUDF/SIGA_batch_1/" + self.model_name + "/pcd_" + self.model_name + ".ply")
        # gt_pts_mesh = trimesh.load("/home/kylin-h20/Desktop/Others/experiment/GTUDF/" + self.model_name + "/pcd_" + self.model_name + ".ply")
        # gt_kdtree = cKDTree(gt_pts_mesh.vertices)

        # distance, vertex_ids = gt_kdtree.query(points_init.detach().cpu().numpy(), p=2, distance_upper_bound=1.5/512)
        # del distance, vertex_ids, gt_kdtree

        # points_raw_np = points_init.detach().cpu().numpy()[distance<1.5/512]
        # out_mesh = trimesh.Trimesh(vertices=points_raw_np)
        # out_mesh.export("/home/kylin-h20/init_points_with_ckdtree.ply")

        # points_init = torch.from_numpy(points_raw_np).cuda()


        # New method
        # gt_pts_mesh = trimesh.load("/home/kylin-h20/Desktop/Others/experiment/CAPUDF/SIGA_batch_1/" + self.model_name + "/pcd_" + self.model_name + ".ply")
        if self.model_name[:3] == "pcd":
            gt_pts_mesh = trimesh.load("/home/kylin-h20/Files/Others/Model/nonmaniford_dataset_DIY/pcd/" + self.model_name + ".ply")
            # gt_pts_mesh = trimesh.load("/home/kylin-h20/Desktop/Others/experiment/GTUDF/" + self.model_name + "/" + self.model_name + ".ply")
        else:
            gt_pts_mesh_path = os.path.join("./ckpt/", dataset_dir, self.model_name, "pcd_{}.ply".format(self.model_name))
            print(gt_pts_mesh_path)
            gt_pts_mesh = trimesh.load(gt_pts_mesh_path)
            # gt_pts_mesh = trimesh.load("/home/kylin-h20/Desktop/Others/experiment/GTUDF/" + self.model_name + "/pcd_" + self.model_name + ".ply")
        bbox = torch.from_numpy(gt_pts_mesh.bounds).float().cuda()
        pcd_raw = torch.from_numpy(gt_pts_mesh.vertices).float().cuda()
        del gt_pts_mesh

        expand_scale = 2
        temp = []
        for k in range(-expand_scale, expand_scale + 1):
            for j in range(-expand_scale, expand_scale + 1):
                for i in range(-expand_scale, expand_scale + 1):
                    temp.append([i, j, k])
        temp = torch.tensor(temp).cuda()

        bbox = (bbox * self.resolution).int()
        width = bbox[1] - bbox[0] + 1
        orders = torch.tensor([width[0], width[0] * width[1]]).cuda()

        neighbors = temp[:, 0] + temp[:, 1] * orders[0] + temp[:, 2] * orders[1]
        neighbors_count = neighbors.shape[0]
        pcd_raw = (pcd_raw * self.resolution).int() - bbox[0]
        pcd_indexes = pcd_raw[:, 0] + pcd_raw[:, 1] * orders[0] + pcd_raw[:, 2] * orders[1]
        expanded_indexes = (pcd_indexes.unsqueeze(1).repeat(1, neighbors_count) + neighbors).reshape(1, -1).squeeze()
        expanded_indexes = expanded_indexes.unique()

        points_init = torch.zeros(expanded_indexes.shape[0], 3, dtype=torch.float32).cuda()
        points_init[:, 2] = (expanded_indexes / orders[1]).floor()
        points_init[:, 1] = ((expanded_indexes % orders[1]) / orders[0]).floor()
        points_init[:, 0] = ((expanded_indexes % orders[1]) % orders[0]).floor()

        points_init = ((points_init + bbox[0]) / self.resolution) + 0.5 / self.resolution

        batch_size = 50000
        # threshold_ceil = 1.5 / 256
        threshold_floor = 1.5 / 1024
        # gap = 1 / 512points_raw
        gap = 1 / 256
        head = 0
        indexes = torch.empty(0, dtype=torch.int32)
        indexes_0 = torch.empty(0, dtype=torch.int32)
        while head < points_init.shape[0]:
            batch = points_init[head:min((head + batch_size), points_init.shape[0])]
            dist = query_func(batch).reshape(-1).abs()
            # index = torch.nonzero((dist < (gap + (1 / self.resolution) / 2)) & (dist > (gap - (1 / self.resolution) / 2))).reshape(-1)
            # index = torch.nonzero(dist <= (gap + (1 / self.resolution) / 2)).reshape(-1)
            index = torch.nonzero((dist <= (2 * gap)) & (dist != 0)).reshape(-1)
            indexes = torch.cat([indexes, head + index])
            index = torch.nonzero(dist == 0).reshape(-1)
            indexes_0 = torch.cat([indexes_0, head + index])
            
            # a = dist[dist < 0].detach().cpu().numpy()
            # if a.shape[0] > 0:
            #     dist_temp = np.append(dist_temp, values=a, axis=0)
            head += batch_size


        # # Get gradients of points_raw
        points_raw = torch.index_select(points_init, 0, indexes).detach()
        points_dist_zero = torch.index_select(points_init, 0, indexes_0).detach()
        points_dist_0_mesh = trimesh.Trimesh(vertices=points_dist_zero.detach().cpu().numpy())
        # points_final_snap_to_half_grid = points_dist_zero.clone()
        # points_dist_0_mesh.export(dest_dir_ply + "/points_raw_dist_0.ply")
        # np.savetxt(dest_dir_txt + "/vertices_dist_0.txt", points_dist_0_mesh.vertices)
        points_final_all = np.concatenate((points_final_all, points_dist_0_mesh.vertices), axis=0)
        del points_init, points_dist_0_mesh, indexes_0
        points_raw.requires_grad = True
        head = 0
        while head < points_raw.shape[0]:
            batch = points_raw[head:min((head + batch_size), points_raw.shape[0])]
            dist = query_func(batch).reshape(-1)

            loss = dist.mean()
            loss.backward()

            head += batch_size

        grad = torch.nn.functional.normalize(points_raw.grad)

        
        dist_last = torch.empty(0, dtype=torch.float32)

        # # ## ablation, direct optimizing
        # # points_optimized = points_raw
        # # points_optimized.requires_grad = True
        # # self.optimizer = VectorAdam([points_optimized])
        # # for iter in range(self.iter_max):
        # #     self.update_learning_rate(iter, 0.025)
        # #     epoch_loss = 0
        # #     self.optimizer.zero_grad()
        # #     head = 0
            
        # #     while head < points_raw.shape[0]:
        # #         end = min((head + batch_size), points_raw.shape[0])
        # #         batch = points_optimized[head:end]
        # #         # batch = torchquery_func.cat([loca, points_raw[head:end, 1], points_raw[head:end, 2]]).reshape(3, -1).t()
        # #         dist = query_func(batch)

        # #         loss = dist.abs().mean()
        # #         epoch_loss += loss.data
        # #         loss.backward()

        # #         head += batch_squery_funcize

        # #     self.optimizer.step()
        # #     if iter == 0 or iter % 50 == 49:
        # #         print("    {} epoch in 1st stage finished, loss = {}!".format(iter, epoch_loss/math.ceil(points_raw.shape[0]/batch_size)))

        print("Starting evaluating!")
        alpha = torch.zeros(points_raw.shape[0], dtype=torch.float32)
        alpha.requires_grad = True
        self.optimizer = VectorAdam([alpha])
        for iter in range(self.iter_max):
            # # lr for NUDF
            # self.update_learning_rate(iter, 1.5)

            # lr for GTUDF
            self.update_learning_rate(iter, 3)
            epoch_loss = 0
            self.optimizer.zero_grad()
            head = 0
            
            while head < points_raw.shape[0]:
                end = min((head + batch_size), points_raw.shape[0])
                batch = alpha[head:end].unsqueeze(1).expand(-1, 3) * grad[head:end] + points_raw[head:end]
                # batch = torch.cat([loca, points_raw[head:end, 1], points_raw[head:end, 2]]).reshape(3, -1).t()
                dist = query_func(batch)
                if iter == (self.iter_max - 1):
                    dist_last = torch.cat([dist_last, dist])

                loss = dist.mean()
                epoch_loss += loss.data
                loss.backward()

                head += batch_size

            self.optimizer.step()
            if iter == 0 or iter % 50 == 49:
                print("    {} epoch in 1st stage finished, loss = {}!".format(iter, epoch_loss/math.ceil(points_raw.shape[0]/batch_size)))
        
        points_optimized = points_raw + alpha.unsqueeze(1).expand(-1, 3) * grad
        points_optimized_mesh = trimesh.Trimesh(vertices=points_optimized.detach().cpu().numpy())

        temp_all = np.concatenate((points_optimized_mesh.vertices, points_final_all), axis=0)
        # np.savetxt(dest_dir_txt + "/new_scheme_points_final_all_wo_selecting.txt", temp_all)
        
        # # # # final selection
        dist_last = dist_last.reshape(-1)
        # indexes = torch.nonzero(dist_last < (1.25/1024)).reshape(-1)
        # indexes = torch.nonzero(dist_last < (1.5/1024)).reshape(-1)
        indexes = torch.nonzero(dist_last < (1.0/256)).reshape(-1)
        points_optimized_mesh = trimesh.Trimesh(vertices=torch.index_select(points_optimized, 0, indexes).detach().cpu().numpy())


        # # Selecting according distances between result after optimizing and input pointclouds.
        # distance, vertex_ids = gt_kdtree.query(points_optimized_mesh.vertices, p=2, distance_upper_bound=1.5/128)

        # points_raw_np = points_optimized_mesh.vertices[distance<1.5/128]
        # out_mesh = trimesh.Trimesh(vertices=points_raw_np)
        # out_mesh.export("/home/kylin-h20/optimized_points_with_ckdtree.ply")


        points_optimized_x_select = torch.from_numpy(points_optimized_mesh.vertices).cuda().float()
        points_optimized_x_select.requires_grad = True
        head = 0
        dist_last = torch.zeros(0).cuda()
        while head < points_optimized_x_select.shape[0]:
            batch = points_optimized_x_select[head:min((head + batch_size), points_optimized_x_select.shape[0])]
            dist = query_func(batch).reshape(-1)
            dist_last = torch.cat([dist_last, dist])

            loss = dist.mean()
            loss.backward()

            head += batch_size

        grad = torch.nn.functional.normalize(points_optimized_x_select.grad)
        points_optimized_mesh.vertex_normals = grad.detach().cpu().numpy()
        # points_optimized_mesh.export(dest_dir_ply + "/new_scheme_points_optimized_1024*150.ply")
        points_final_all = np.concatenate((points_final_all, points_optimized_mesh.vertices), axis=0)

        global_bounds = np.zeros([2, 3], dtype=np.float32)
        # global_bounds[0][0] = points_final_all[:, 0].min()
        # global_bounds[0][1] = points_final_all[:, 1].min()
        # global_bounds[0][2] = points_final_all[:, 2].min()
        # global_bounds[1][0] = points_final_all[:, 0].max()
        # global_bounds[1][1] = points_final_all[:, 1].max()
        # global_bounds[1][2] = poin.abs()ts_final_all[:, 2].max()
        global_bounds[0][0] = temp_all[:, 0].min()
        global_bounds[0][1] = temp_all[:, 1].min()
        global_bounds[0][2] = temp_all[:, 2].min()
        global_bounds[1][0] = temp_all[:, 0].max()
        global_bounds[1][1] = temp_all[:, 1].max()
        global_bounds[1][2] = temp_all[:, 2].max()
        # np.savetxt(dest_dir_txt + "/new_scheme_global_bounds.txt", global_bounds)
        # np.savetxt(dest_dir_txt + "/new_scheme_points_final_all.txt", points_final_all)
        # print("finished!")
        return points_final_all, global_bounds.reshape(-1)



    def finetune(self, vertices=None, faces=None):
        query_func = self.query_func

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        xyz = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
        xyz.requires_grad = True
        self.optimizer = VectorAdam([xyz])
        laplacian_op = laplacian_calculation(mesh).cuda()

        vertex_faces = np.asarray(mesh.vertex_faces)
        face_mask = np.ones_like(vertex_faces).astype(bool)
        face_mask[vertex_faces==-1] = False
        for it in range(self.iter_max_vice):
            # display = xyz[:100]
            if it == self.iter_max_vice:
                points = xyz.detach().cpu().numpy()

                normal_mesh = trimesh.Trimesh(vertices=points, faces=mesh.faces, process=False)
                normals = torch.FloatTensor(normal_mesh.face_normals).cuda()
                origin_points = get_mid(xyz,mesh.faces).detach().clone()

            self.update_learning_rate(it, 0.005)

            epoch_loss = 0
            self.optimizer.zero_grad()
            num_samples = xyz.shape[0]
            if it % 50 == 0:
                a = 0
            head = 0
            while head < num_samples:
                sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
                df = query_func(sample_subset)
                # del sample_subset
                df_loss = df.abs().mean()
                loss = df_loss

                if it <= self.iter_max_vice:
                    s_value = calculate_s(xyz, mesh.faces)
                    face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]

                    face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                    face_weight = torch.sum(face_weight, dim=1)

                    face_weight = torch.sqrt(face_weight.detach())
                    face_weight = face_weight.max() / face_weight

                    lap_v = laplacian_step(laplacian_op, xyz)
                    lap_v = torch.mul(lap_v, lap_v)
                    lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
                    laplacian_loss = face_weight * torch.sum(lap_v, dim=1)
                    # laplacian_loss = 2000 * laplacian_loss.mean()
                    laplacian_loss = 500 * laplacian_loss.mean()
                    loss = loss + laplacian_loss

                epoch_loss += loss.data
                loss.backward()
                head += self.max_batch

            mid_num_samples = len(mesh.faces)
            mid_head = 0
            while mid_head < mid_num_samples:
                mid_points = get_mid(xyz, mesh.faces)
                sub_mid_points = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]
                mid_df = query_func(sub_mid_points)
                mid_df_loss = mid_df.mean()
                loss = mid_df_loss
                if it > self.iter_max_vice:
                    offset = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])] - origin_points[
                                                                                                         mid_head: min(
                                                                                                             mid_head + self.max_batch,
                                                                                                             mid_points.shape[
                                                                                                                 0])]
                    normal_loss = torch.norm(
                        torch.cross(offset, normals[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])], dim=-1),
                        dim=-1)
                    normal_loss = 0.5* normal_loss.mean()
                    loss +=  normal_loss
                epoch_loss += loss.data
                loss.backward()
                mid_head += self.max_batch


            self.optimizer.step()
            # if (it+1) % self.report_freq_vice == 0:
            #     print(" {} iteration, loss={}".format(it, epoch_loss))

        
        print("System Info: Mesh after optimizing has been exported, please check it!")
        return xyz.detach().cpu().numpy(), mesh.faces


    # def finetune(self, vertices,faces):
    #     query_func = self.query_func

    #     # mesh = trimesh.Trimesh(vertices=vertices,faces=faces)
    #     output_dir = "./result"
    #     print(os.path.join(output_dir,self.model_name,"mesh_of_" + self.model_name + "_after_postprocessing.ply"))
    #     mesh = trimesh.load_mesh(os.path.join(output_dir,self.model_name,"mesh_of_" + self.model_name + "_after_postprocessing.ply"))


    #     xyz = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
    #     xyz.requires_grad = True
    #     self.optimizer = VectorAdam([xyz])
    #     laplacian_op = laplacian_calculation(mesh).cuda()
    #     vertex_faces = np.asarray(mesh.vertex_faces)
    #     face_mask = np.ones_like(vertex_faces).astype(bool)
    #     face_mask[vertex_faces==-1] = False
    #     for it in range(self.iter_max_vice):
    #         if it == self.iter_max_vice:
    #             points = xyz.detach().cpu().numpy()
    #             normal_mesh = trimesh.Trimesh(vertices=points, faces=mesh.faces, process=False)
    #             normals = torch.FloatTensor(normal_mesh.face_normals).cuda()
    #             origin_points = get_mid(xyz,mesh.faces).detach().clone()
    #         self.update_learning_rate(it, 0.005)
    #         epoch_loss = 0
    #         self.optimizer.zero_grad()
    #         num_samples = xyz.shape[0]
    #         if it % 50 == 0:
    #             a = 0
    #         head = 0
    #         while head < num_samples:
    #             sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
    #             df = query_func(sample_subset)
    #             df_loss = df.abs().mean()
    #             loss = df_loss

    #             if it <= self.iter_max_vice:
    #                 s_value = calculate_s(xyz, mesh.faces)
    #                 face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]

    #                 face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
    #                 face_weight = torch.sum(face_weight, dim=1)

    #                 face_weight = torch.sqrt(face_weight.detach())
    #                 face_weight = face_weight.max() / face_weight

    #                 lap_v = laplacian_step(laplacian_op, xyz)
    #                 lap_v = torch.mul(lap_v, lap_v)
    #                 lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
    #                 laplacian_loss = face_weight * torch.sum(lap_v, dim=1)
    #                 laplacian_loss = 500 * laplacian_loss.mean()
    #                 loss = loss + laplacian_loss

    #             epoch_loss += loss.data
    #             loss.backward()
    #             head += self.max_batch

    #         mid_num_samples = len(mesh.faces)
    #         mid_head = 0
    #         while mid_head < mid_num_samples:
    #             mid_points = get_mid(xyz, mesh.faces)
    #             sub_mid_points = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]
    #             mid_df = query_func(sub_mid_points)
    #             mid_df_loss = mid_df.mean()
    #             loss = mid_df_loss
    #             if it > self.iter_max_vice:
    #                 offset = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])] - origin_points[
    #                                                                                                      mid_head: min(
    #                                                                                                          mid_head + self.max_batch,
    #                                                                                                          mid_points.shape[
    #                                                                                                              0])]
    #                 normal_loss = torch.norm(
    #                     torch.cross(offset, normals[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])], dim=-1),
    #                     dim=-1)
    #                 normal_loss = 0.5* normal_loss.mean()
    #                 loss +=  normal_loss
    #             epoch_loss += loss.data
    #             loss.backward()
    #             mid_head += self.max_batch


    #         self.optimizer.step()
    #         if (it+1) % self.report_freq_vice == 0:
    #             print(" {} iteration, loss={}".format(it, epoch_loss))

    #     return xyz.detach().cpu().numpy(), mesh.faces


    def update_learning_rate(self, iter_step, scale):
        warn_up = self.warm_up_end
        max_iter = self.iter_max + 200
        init_lr = self.learning_rate
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        lr = scale * lr * init_lr
        if iter_step >= 400:
            lr *= 0.1

        for g in self.optimizer.param_groups:
            g['lr'] = lr
