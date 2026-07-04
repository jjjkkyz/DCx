import time
import os
import trimesh
import numpy as np
import argparse
import cubvh
from pyhocon import ConfigFactory
from CAPUDF.models.fields import CAPUDFNetwork
from DCX.upsampling_speedup_according_to_gradients_of_initial_points_directly_removefar import DCX
from DCX.cxh_GTUDF import DiffUdfCubvh
import dcx_pkg
import torch

from trimesh.grouping import group_rows
from CAPUDF.run import Runner

def normalized_mesh(mesh):
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extents = mesh.extents
    max_extent = extents.max()
    scale_factor = 1.0 / max_extent
    
    mesh.vertices -= center
    mesh.vertices *= scale_factor

def sample_mesh_trimesh(mesh,num_points=100000000,batch_size=10000000,device='cpu'):

    min_bbox = np.min(mesh.vertices, axis=0)
    max_bbox = np.max(mesh.vertices, axis=0)
    bbox = np.hstack((min_bbox, max_bbox))


    verts_np = mesh.vertices
    faces_np = mesh.faces
    verts = torch.tensor(verts_np, dtype=torch.float32)
    faces = torch.tensor(faces_np, dtype=torch.int64)
    samples_list = []
    num_batch = int(np.ceil(num_points / batch_size))
    if device != 'cpu':
        from pytorch3d.structures import Meshes
        from pytorch3d.ops import sample_points_from_meshes
        verts_cuda = verts.unsqueeze(0).to(device)
        faces_cuda = faces.unsqueeze(0).to(device)
        cu_mesh = Meshes(verts=verts_cuda,faces = faces_cuda)
        print(f"Sampling points from GTmesh by GPU")
        for i in range(num_batch):
            current_batch_size = min(batch_size, num_points - i * batch_size)
            sampled_points = sample_points_from_meshes(cu_mesh, num_samples=current_batch_size)[0]
            samples_list.append(sampled_points.cpu().numpy())
        
    else :
        print(f"Sampling points from GTmesh by CPU")
        for i in range(num_batch):
            current_batch_size = min(batch_size, num_points - i * batch_size)
            samples_points = trimesh.sample.sample_surface(mesh, current_batch_size)[0]
            samples_list.append(samples_points)
    samples = np.concatenate(samples_list, axis=0)
    samples = samples[:num_points]

   
    print("Sampling completed.")

    return samples, bbox


def mesh_extraction(points, bbox, dataname, resolution, enable_thinning,
                    enable_supplementary_sampling, enable_postprocessing,
                    hole_query_fn=None):
    points = np.asarray(points, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.shape[0] != 6:
        raise ValueError(f"bbox must contain 6 values, got shape {bbox.shape}")
    print(f"Extracting mesh with resolution {resolution} and bounding box {bbox}...")

    voxel_ids, voxel_points, bbox, orders = dcx_pkg.points_to_voxels(
        points=points,
        bbox=bbox,
        res=resolution,
    )
    cube_ids, cube_types = dcx_pkg.get_cube_types(
        voxel_ids=voxel_ids,
        voxel_points=voxel_points,
        orders=orders,
        res=resolution,
    )
    
    if enable_supplementary_sampling:
        if enable_thinning:
            voxel_ids, voxel_points, cube_ids, cube_types = dcx_pkg.thinning(
                voxel_ids=voxel_ids,
                voxel_points=voxel_points,
                cube_ids=cube_ids,
                cube_types=cube_types,
                orders=orders,
                res=resolution,
            )
        margin_voxel_ids, _, _ = dcx_pkg.reconstruction(
            voxel_ids=voxel_ids,
            voxel_points=voxel_points,
            cube_ids=cube_ids,
            cube_types=cube_types,
            orders=orders,
            res=resolution,
            pattern=1,
            enable_postprocessing=False,
            dataname=dataname,
        )
        if hole_query_fn is None:
            raise ValueError("hole_query_fn is required when supplementary_sampling is enabled")
        candidate_voxel_ids, query_points = dcx_pkg.filling_holes_prepare(
            margin_voxel_ids=margin_voxel_ids,
            voxel_ids=voxel_ids,
            voxel_points=voxel_points,
            bbox=bbox,
            orders=orders,
            res=resolution,
        )
        selected_candidate_indexes, selected_points = hole_query_fn(query_points)
        selected_candidate_indexes = np.asarray(selected_candidate_indexes, dtype=np.int64).reshape(-1)
        selected_points = np.asarray(selected_points, dtype=np.float32).reshape(-1, 3)
        selected_voxel_ids = np.asarray(candidate_voxel_ids, dtype=np.int64)[selected_candidate_indexes]
        voxel_ids, voxel_points, add_voxel_ids, add_voxel_points = dcx_pkg.filling_holes_apply(
            selected_voxel_ids=selected_voxel_ids,
            selected_points=selected_points,
            voxel_ids=voxel_ids,
            voxel_points=voxel_points,
        )
    
        cube_ids, cube_types = dcx_pkg.get_cube_types(
            voxel_ids=voxel_ids,
            voxel_points=voxel_points,
            orders=orders,
            res=resolution,
        )

    if enable_thinning:
        voxel_ids, voxel_points, cube_ids, cube_types = dcx_pkg.thinning(
            voxel_ids=voxel_ids,
            voxel_points=voxel_points,
            cube_ids=cube_ids,
            cube_types=cube_types,
            orders=orders,
            res=resolution,
        )
    
    _, vertices, faces = dcx_pkg.reconstruction(
        voxel_ids=voxel_ids,
        voxel_points=voxel_points,
        cube_ids=cube_ids,
        cube_types=cube_types,
        orders=orders,
        res=resolution,
        pattern=0,
        enable_postprocessing=enable_postprocessing,
        dataname=dataname,
    )
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def test(args):
    args.dir_name = args.dataname
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda')

    object_bbox_min = np.array([0.0, 0.0, 0.0])-0.5
    object_bbox_max = np.array([0.0, 0.0, 0.0])+0.5

    conf_path = args.conf
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    
    dataset_dir = conf.get("dir_path.dataset_dir",default="")
    capudf_result_dir = os.path.join(conf.get_string("dir_path.result_dir"), dataset_dir,args.datadir,"CAPUDF_output")
    result_dir = os.path.join(conf.get_string("dir_path.result_dir"), dataset_dir, args.datadir)

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(capudf_result_dir, exist_ok=True)
    resolution = conf.get_int('evaluate.resolution')
    

    if args.udf_type == 0:
        udf_network = CAPUDFNetwork(**conf['model.udf_network']).to(device)
        checkpoint_name = conf.get_string('evaluate.load_ckpt')
        checkpoint_path = os.path.join("./ckpt",dataset_dir, args.dataname,checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print("!"*50)
            print(f"Checkpoint not found at {checkpoint_path}. Starting CAPUDF training to generate the checkpoint...")
            print("!"*50)
            # samples, bbox = sample_mesh_trimesh(input_mesh, num_points=100000, batch_size=conf.get_int("evaluate.sample_batch_size"),sharp = False)
            # pc = trimesh.points.PointCloud(samples)
            # pc.export(os.path.join(capudf_result_dir, "{}.ply".format(args.dataname)))
            # runner = Runner(args.datadir,args.dataname, checkpoint_path, "./CAPUDF/confs/base.conf",capudf_result_dir)
            # runner.train()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        udf_network.load_state_dict(checkpoint['udf_network_fine'])

        evaluator = DCX(lambda pts: torch.abs(udf_network.udf(pts)), conf.get_int('evaluate.resolution'), conf.get_float('evaluate.threshold'),
                        iter_max=conf.get_int("evaluate.iter_max"), bound_min=object_bbox_min, bound_max=object_bbox_max, 
                        max_batch=conf.get_int("evaluate.max_batch"), learning_rate=conf.get_float("evaluate.learning_rate"), 
                        warm_up_end=conf.get_int("evaluate.warm_up_end"), report_freq=conf.get_int("evaluate.report_freq"), dest_dir=os.path.join(result_dir, ""), model_name=args.dataname)
        samples, bbox = evaluator.optimize1()


    elif args.udf_type == 1:
        data_dir = os.path.join(conf.get_string("dir_path.data_dir"), dataset_dir, args.datadir)
        data = os.path.join(data_dir, "{}.ply".format(args.dataname))
        input_mesh = trimesh.load_mesh(data)
        normalized_mesh(input_mesh)
        
        BVH = cubvh.cuBVH(input_mesh.vertices, input_mesh.faces)
        UDF = DiffUdfCubvh(BVH, torch.FloatTensor(input_mesh.vertices).to(device), torch.LongTensor(input_mesh.faces).to(device))
        evaluator = DCX(lambda pts: torch.abs(UDF(pts)), conf.get_int('evaluate.resolution'), conf.get_float('evaluate.threshold'),
                        iter_max=conf.get_int("evaluate.iter_max"), bound_min=object_bbox_min, bound_max=object_bbox_max, 
                        max_batch=conf.get_int("evaluate.max_batch"), learning_rate=conf.get_float("evaluate.learning_rate"), 
                        warm_up_end=conf.get_int("evaluate.warm_up_end"), report_freq=conf.get_int("evaluate.report_freq"), dest_dir=result_dir, model_name=args.dataname)

        samples, bbox = sample_mesh_trimesh(input_mesh, num_points=args.num_points, batch_size=conf.get_int("evaluate.sample_batch_size"), device=device)

    vertices, faces = mesh_extraction(
        points=samples,
        bbox=bbox,
        dataname=args.dataname,
        resolution=resolution,
        enable_thinning=args.thinning,
        enable_supplementary_sampling=args.supplementary_sampling,
        enable_postprocessing=args.postprocessing,
        hole_query_fn=evaluator.filter_query_points_for_filling_holes,
    )
    
    if args.finetune:
        ft_vertices, ft_faces = evaluator.finetune(vertices, faces)
        mesh = trimesh.Trimesh(vertices=ft_vertices, faces=ft_faces,process = False)
        mesh.export(os.path.join(result_dir,"{}.ply".format(args.dataname)))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(os.path.join(result_dir,"{}.ply".format(args.dataname)))




if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/test.conf", help="Path to the configuration file.")
    parser.add_argument('--datadir', type=str, required=True, help="Directory path containing the input data.")
    parser.add_argument('--dataname', type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument('--gpu', type=int, default=3, help="GPU device ID to use.")
    parser.add_argument('--udf_type', '--udf', type=int, default=1, choices=[0, 1], help="Select UDF type: 0 for NUDF, 1 for GTUDF.")
    parser.add_argument('--num_points', '--num', type=int, default=200000000, help="Number of points to sample.")
    parser.add_argument('--thinning', '--thin', action='store_true', help="Enable the thinning process.")
    parser.add_argument('--supplementary_sampling', '--supsamp', action='store_true', help="Enable the supplementary sampling.")
    parser.add_argument('--postprocessing', '--postp', action='store_true', help="Enable the post-processing stage.")
    parser.add_argument('--finetune', '--ft', action='store_true', help="Enable the fine-tuning process.")
    args = parser.parse_args() 
    test(args)
