#include "thinning_and_reconstruction_with_3_special_classes.cpp"
#include "points_to_voxels.cpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <set>
#include <chrono>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iterator>
#include <chrono>

using namespace std;
namespace py = pybind11;
using LL = long long;

void initialize_thinning_context(const vector<int> &orders, int res)
{
    if (orders.size() != 2)
    {
        throw std::runtime_error("orders must have shape (2,)");
    }

    order_1 = orders[0];
    order_2 = orders[1];
    resolution = res;

    voxel_offsets = {0, 1, order_1, order_1 + 1, order_2, order_2 + 1, order_2 + order_1, order_2 + order_1 + 1};
    cube_offsets = {{0, 1, 0}, {-1, 2, 1}, {-order_1, 4, 2}, {-order_1 - 1, 8, 3},
        {-order_2, 16, 4}, {-order_2 - 1, 32, 5}, {-order_2 - order_1, 64, 6}, {-order_2 - order_1 - 1, 128, 7}};
    neighbors_offset_6 = {-1, 1, -order_1, order_1, -order_2, order_2};
    neighbors_offset_26 = {-order_2 - order_1 - 1, -order_2 - order_1, -order_2 - order_1 + 1, -order_2 - 1, -order_2, -order_2 + 1,
        -order_2 + order_1 - 1, -order_2 + order_1, -order_2 + order_1 + 1, -order_1 - 1, -order_1, -order_1 + 1,
        -1, 1, order_1 - 1, order_1, order_1 + 1, order_2 - order_1 - 1, order_2 - order_1, order_2 - order_1 + 1,
        order_2 - 1, order_2, order_2 + 1, order_2 + order_1 - 1, order_2 + order_1, order_2 + order_1 + 1};
    added_check_offsets = {0, -1, -order_1, -order_2};
    neighbors_offset_by_cube_face = {0, -1, 1, -order_1, order_1, -order_2, order_2};

    initialize_tables();
}

vector<int> parse_orders(py::array_t<int, py::array::c_style | py::array::forcecast> orders_arr)
{
    auto orders_buf = orders_arr.request();
    if (orders_buf.ndim != 1 || orders_buf.shape[0] != 2)
    {
        throw std::runtime_error("orders must have shape (2,)");
    }

    int* orders_ptr = static_cast<int*>(orders_buf.ptr);
    return {orders_ptr[0], orders_ptr[1]};
}

unordered_map<LL, vector<float>> parse_voxels(
    py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> voxel_points_arr)
{
    auto voxel_ids_buf = voxel_ids_arr.request();
    auto voxel_points_buf = voxel_points_arr.request();
    if (voxel_ids_buf.ndim != 1)
    {
        throw std::runtime_error("voxel_ids must have shape (N,)");
    }
    if (voxel_points_buf.ndim != 2 || voxel_points_buf.shape[1] != 3 || voxel_points_buf.shape[0] != voxel_ids_buf.shape[0])
    {
        throw std::runtime_error("voxel_points must have shape (N, 3) and match voxel_ids");
    }

    auto voxel_ids = voxel_ids_arr.unchecked<1>();
    auto voxel_points = voxel_points_arr.unchecked<2>();
    unordered_map<LL, vector<float>> voxels;
    for (py::ssize_t i = 0; i < voxel_ids.shape(0); ++i)
    {
        voxels[static_cast<LL>(voxel_ids(i))] = {
            voxel_points(i, 0),
            voxel_points(i, 1),
            voxel_points(i, 2)
        };
    }
    return voxels;
}

unordered_map<LL, int> parse_cube_types(
    py::array_t<long long, py::array::c_style | py::array::forcecast> cube_ids_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> cube_types_arr)
{
    auto cube_ids_buf = cube_ids_arr.request();
    auto cube_types_buf = cube_types_arr.request();
    if (cube_ids_buf.ndim != 1)
    {
        throw std::runtime_error("cube_ids must have shape (N,)");
    }
    if (cube_types_buf.ndim != 1 || cube_types_buf.shape[0] != cube_ids_buf.shape[0])
    {
        throw std::runtime_error("cube_types must have shape (N,) and match cube_ids");
    }

    auto cube_ids = cube_ids_arr.unchecked<1>();
    auto cube_types = cube_types_arr.unchecked<1>();
    unordered_map<LL, int> cube_type_map;
    for (py::ssize_t i = 0; i < cube_ids.shape(0); ++i)
    {
        cube_type_map[static_cast<LL>(cube_ids(i))] = cube_types(i);
    }
    return cube_type_map;
}

unordered_set<LL> parse_voxel_set(py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr)
{
    auto voxel_ids_buf = voxel_ids_arr.request();
    if (voxel_ids_buf.ndim != 1)
    {
        throw std::runtime_error("voxel_ids must have shape (N,)");
    }

    auto voxel_ids = voxel_ids_arr.unchecked<1>();
    unordered_set<LL> voxel_set;
    for (py::ssize_t i = 0; i < voxel_ids.shape(0); ++i)
    {
        voxel_set.insert(static_cast<LL>(voxel_ids(i)));
    }
    return voxel_set;
}

py::tuple serialize_voxels(const unordered_map<LL, vector<float>> &voxels)
{
    vector<pair<LL, vector<float>>> voxel_items(voxels.begin(), voxels.end());
    sort(voxel_items.begin(), voxel_items.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    py::array_t<long long> voxel_ids_arr({static_cast<py::ssize_t>(voxel_items.size())});
    py::array_t<float> voxel_points_arr({static_cast<py::ssize_t>(voxel_items.size()), static_cast<py::ssize_t>(3)});
    auto voxel_ids_buf = voxel_ids_arr.mutable_unchecked<1>();
    auto voxel_points_buf = voxel_points_arr.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(voxel_items.size()); ++i)
    {
        voxel_ids_buf(i) = voxel_items[i].first;
        for (py::ssize_t j = 0; j < 3; ++j)
        {
            voxel_points_buf(i, j) = voxel_items[i].second[j];
        }
    }

    return py::make_tuple(voxel_ids_arr, voxel_points_arr);
}

py::tuple serialize_cube_types(const unordered_map<LL, int> &cube_types)
{
    vector<pair<LL, int>> cube_items(cube_types.begin(), cube_types.end());
    sort(cube_items.begin(), cube_items.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    py::array_t<long long> cube_ids_arr({static_cast<py::ssize_t>(cube_items.size())});
    py::array_t<int> cube_types_arr({static_cast<py::ssize_t>(cube_items.size())});
    auto cube_ids_buf = cube_ids_arr.mutable_unchecked<1>();
    auto cube_types_buf = cube_types_arr.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(cube_items.size()); ++i)
    {
        cube_ids_buf(i) = cube_items[i].first;
        cube_types_buf(i) = cube_items[i].second;
    }

    return py::make_tuple(cube_ids_arr, cube_types_arr);
}

py::array_t<long long> serialize_voxel_set(const unordered_set<LL> &voxel_set)
{
    vector<LL> voxel_items(voxel_set.begin(), voxel_set.end());
    sort(voxel_items.begin(), voxel_items.end());

    py::array_t<long long> voxel_ids_arr({static_cast<py::ssize_t>(voxel_items.size())});
    auto voxel_ids_buf = voxel_ids_arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(voxel_items.size()); ++i)
    {
        voxel_ids_buf(i) = voxel_items[i];
    }
    return voxel_ids_arr;
}

py::array_t<float> serialize_vertices(const vector<vector<float>> &vertices)
{
    py::array_t<float> vertices_arr({static_cast<py::ssize_t>(vertices.size()), static_cast<py::ssize_t>(3)});
    auto vertices_buf = vertices_arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < vertices_buf.shape(0); ++i)
    {
        for (py::ssize_t j = 0; j < 3; ++j)
        {
            vertices_buf(i, j) = vertices[i][j];
        }
    }
    return vertices_arr;
}

py::array_t<long long> serialize_faces(const vector<vector<LL>> &faces)
{
    py::array_t<long long> faces_arr({static_cast<py::ssize_t>(faces.size()), static_cast<py::ssize_t>(3)});
    auto faces_buf = faces_arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < faces_buf.shape(0); ++i)
    {
        for (py::ssize_t j = 0; j < 3; ++j)
        {
            faces_buf(i, j) = faces[i][j];
        }
    }
    return faces_arr;
}

py::tuple points_to_voxels_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> bbox_arr,
    int resolution)
{
    auto points_buf = points_arr.request();
    auto bbox_buf = bbox_arr.request();
    if (points_buf.ndim != 2 || points_buf.shape[1] != 3)
    {
        throw std::runtime_error("points must have shape (N, 3)");
    }
    if (bbox_buf.ndim != 1 || bbox_buf.shape[0] != 6)
    {
        throw std::runtime_error("bbox must have shape (6,)");
    }

    float* points = static_cast<float*>(points_buf.ptr);
    auto bbox_out = py::array_t<float>({static_cast<py::ssize_t>(6)});
    auto bbox_out_buf = bbox_out.mutable_unchecked<1>();
    auto bbox_in_buf = bbox_arr.unchecked<1>();
    for (py::ssize_t i = 0; i < 6; ++i)
    {
        bbox_out_buf(i) = bbox_in_buf(i);
    }

    float* bbox = static_cast<float*>(bbox_out.request().ptr);
    int num_points = static_cast<int>(points_buf.shape[0]);
    unordered_map<LL, vector<float>> voxels;
    vector<int> orders(2);
    points_to_voxels<LL>(voxels, points, num_points, bbox, orders, resolution);
    vector<pair<LL, vector<float>>> voxel_items(voxels.begin(), voxels.end());
    sort(voxel_items.begin(), voxel_items.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    py::array_t<long long> voxel_ids_arr({static_cast<py::ssize_t>(voxel_items.size())});
    auto voxel_ids_buf = voxel_ids_arr.mutable_unchecked<1>();
    py::array_t<float> voxel_points_arr({static_cast<py::ssize_t>(voxel_items.size()), static_cast<py::ssize_t>(3)});
    auto voxel_points_buf = voxel_points_arr.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(voxel_items.size()); ++i)
    {
        voxel_ids_buf(i) = voxel_items[i].first;
        for (py::ssize_t j = 0; j < 3; ++j)
        {
            voxel_points_buf(i, j) = voxel_items[i].second[j];
        }
    }

    py::array_t<int> orders_arr({static_cast<py::ssize_t>(2)});
    auto orders_buf = orders_arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < 2; ++i)
    {
        orders_buf(i) = orders[i];
    }

    return py::make_tuple(voxel_ids_arr, voxel_points_arr, bbox_out, orders_arr);
}

py::tuple get_cube_types_py(
    py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> voxel_points_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> orders_arr,
    int resolution)
{
    auto voxels = parse_voxels(voxel_ids_arr, voxel_points_arr);
    auto orders = parse_orders(orders_arr);
    initialize_thinning_context(orders, resolution);

    unordered_map<LL, int> cube_types;
    get_cube_types<LL>(voxels, cube_types);
    return serialize_cube_types(cube_types);
}

py::tuple thinning_py(
    py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> voxel_points_arr,
    py::array_t<long long, py::array::c_style | py::array::forcecast> cube_ids_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> cube_types_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> orders_arr,
    int resolution)
{
    auto voxels = parse_voxels(voxel_ids_arr, voxel_points_arr);
    auto cube_types = parse_cube_types(cube_ids_arr, cube_types_arr);
    auto orders = parse_orders(orders_arr);
    initialize_thinning_context(orders, resolution);
    thinning<LL>(voxels, cube_types);
    auto voxels_out = serialize_voxels(voxels);
    auto cube_types_out = serialize_cube_types(cube_types);
    return py::make_tuple(voxels_out[0], voxels_out[1], cube_types_out[0], cube_types_out[1]);
}

py::tuple reconstruction_py(
    py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> voxel_points_arr,
    py::array_t<long long, py::array::c_style | py::array::forcecast> cube_ids_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> cube_types_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> orders_arr,
    int resolution,
    int pattern,
    bool enable_postprocessing,
    string dataname)
{
    auto voxels = parse_voxels(voxel_ids_arr, voxel_points_arr);
    auto cube_types = parse_cube_types(cube_ids_arr, cube_types_arr);
    auto orders = parse_orders(orders_arr);
    initialize_thinning_context(orders, resolution);
    current_model_name = dataname;

    unordered_set<LL> margin_voxels;
    vector<vector<float>> out_vertices;
    vector<vector<LL>> out_faces;
    
    reconstruction<LL>(voxels, cube_types, pattern, margin_voxels, enable_postprocessing, out_vertices, out_faces);

    return py::make_tuple(
        serialize_voxel_set(margin_voxels),
        serialize_vertices(out_vertices),
        serialize_faces(out_faces));
}

py::tuple filling_holes_py(
    string dataname,
    py::array_t<long long, py::array::c_style | py::array::forcecast> margin_voxel_ids_arr,
    py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> voxel_points_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> bbox_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> orders_arr,
    int resolution)
{
    throw std::runtime_error(
        "filling_holes() legacy file-based path is disabled. "
        "Use filling_holes_prepare() in Python, run UDF filtering there, then call filling_holes_apply().");
}

py::tuple filling_holes_prepare_py(
    py::array_t<long long, py::array::c_style | py::array::forcecast> margin_voxel_ids_arr,
    py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> voxel_points_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> bbox_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> orders_arr,
    int resolution)
{
    auto voxels = parse_voxels(voxel_ids_arr, voxel_points_arr);
    auto margin_voxels = parse_voxel_set(margin_voxel_ids_arr);
    auto orders = parse_orders(orders_arr);
    initialize_thinning_context(orders, resolution);

    auto bbox_buf = bbox_arr.request();
    if (bbox_buf.ndim != 1 || bbox_buf.shape[0] != 6)
    {
        throw std::runtime_error("bbox must have shape (6,)");
    }
    auto bbox_out = py::array_t<float>({static_cast<py::ssize_t>(6)});
    auto bbox_out_buf = bbox_out.mutable_unchecked<1>();
    auto bbox_in_buf = bbox_arr.unchecked<1>();
    for (py::ssize_t i = 0; i < 6; ++i)
    {
        bbox_out_buf(i) = bbox_in_buf(i);
    }

    vector<LL> candidate_voxels;
    vector<vector<float>> query_points;
    filling_holes_prepare<LL>(
        margin_voxels,
        voxels,
        static_cast<float*>(bbox_out.request().ptr),
        resolution,
        candidate_voxels,
        query_points);
    py::array_t<long long> candidate_voxel_ids_arr({static_cast<py::ssize_t>(candidate_voxels.size())});
    auto candidate_voxel_ids_buf = candidate_voxel_ids_arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(candidate_voxels.size()); ++i)
    {
        candidate_voxel_ids_buf(i) = candidate_voxels[i];
    }

    py::array_t<float> query_points_arr({static_cast<py::ssize_t>(query_points.size()), static_cast<py::ssize_t>(3)});
    auto query_points_buf = query_points_arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(query_points.size()); ++i)
    {
        for (py::ssize_t j = 0; j < 3; ++j)
        {
            query_points_buf(i, j) = query_points[i][j];
        }
    }

    return py::make_tuple(candidate_voxel_ids_arr, query_points_arr);
}

py::tuple filling_holes_apply_py(
    py::array_t<long long, py::array::c_style | py::array::forcecast> selected_voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> selected_points_arr,
    py::array_t<long long, py::array::c_style | py::array::forcecast> voxel_ids_arr,
    py::array_t<float, py::array::c_style | py::array::forcecast> voxel_points_arr)
{
    auto voxels = parse_voxels(voxel_ids_arr, voxel_points_arr);
    auto added_voxels = parse_voxels(selected_voxel_ids_arr, selected_points_arr);

    for (const auto &item : added_voxels)
    {
        voxels[item.first] = item.second;
    }
    
    auto voxels_out = serialize_voxels(voxels);
    auto added_voxels_out = serialize_voxels(added_voxels);
    return py::make_tuple(voxels_out[0], voxels_out[1], added_voxels_out[0], added_voxels_out[1]);
}

PYBIND11_MODULE(DCX_main, m) 
{
    m.def("points_to_voxels", &points_to_voxels_py, "Convert points to voxels",
        py::arg("points"),
        py::arg("bbox"),
        py::arg("res"));
    m.def("get_cube_types", &get_cube_types_py, "Compute cube types from voxels",
        py::arg("voxel_ids"),
        py::arg("voxel_points"),
        py::arg("orders"),
        py::arg("res"));
    m.def("thinning", &thinning_py, "Run thinning on voxels and cube types",
        py::arg("voxel_ids"),
        py::arg("voxel_points"),
        py::arg("cube_ids"),
        py::arg("cube_types"),
        py::arg("orders"),
        py::arg("res"));
    m.def("reconstruction", &reconstruction_py, "Run reconstruction from voxels and cube types",
        py::arg("voxel_ids"),
        py::arg("voxel_points"),
        py::arg("cube_ids"),
        py::arg("cube_types"),
        py::arg("orders"),
        py::arg("res"),
        py::arg("pattern"),
        py::arg("enable_postprocessing"),
        py::arg("dataname"));
    m.def("filling_holes", &filling_holes_py, "Fill holes using margin voxels",
        py::arg("dataname"),
        py::arg("margin_voxel_ids"),
        py::arg("voxel_ids"),
        py::arg("voxel_points"),
        py::arg("bbox"),
        py::arg("orders"),
        py::arg("res"));
    m.def("filling_holes_prepare", &filling_holes_prepare_py, "Prepare hole-filling query points",
        py::arg("margin_voxel_ids"),
        py::arg("voxel_ids"),
        py::arg("voxel_points"),
        py::arg("bbox"),
        py::arg("orders"),
        py::arg("res"));
    m.def("filling_holes_apply", &filling_holes_apply_py, "Apply selected supplementary points",
        py::arg("selected_voxel_ids"),
        py::arg("selected_points"),
        py::arg("voxel_ids"),
        py::arg("voxel_points"));

}
