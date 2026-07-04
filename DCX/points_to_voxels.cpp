#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include "happly.h"
#include<chrono>
using namespace std;
using namespace std::chrono;


void get_bound(float* bbox, 
    vector<int> &orders, 
    int resolution)
{
    float temp_0, temp_1;
    vector<int> width(3);
    for(int i = 0; i < 3; i++)
    {
        temp_0 = int(floor(bbox[i] * resolution)) - 3;
        temp_1 = int(floor(bbox[i+3] * resolution)) + 3;
        width[i] = int(temp_1 - temp_0);
        bbox[i] = temp_0 / resolution;
        bbox[i+3] = temp_1 / resolution;
    }
    orders[0] = width[0];
    orders[1] = width[0] * width[1];
}

template<typename CubeID>
int voxelization(unordered_map<CubeID, vector<float>> &voxels, 
    float* points, 
    int num_points, 
    float* bbox, 
    vector<int> &orders, 
    int resolution)
{
    unordered_map<CubeID, int> aux;
    typename unordered_map<CubeID, int>::iterator iter_0;
    typename unordered_map<CubeID, vector<float>>::iterator iter_1;
    vector<float> point(3), point_merge(3);
    vector<int> point_scaling(3);
    float value, value_scaling;
    CubeID voxel;
    int count;
    for (int i = 0;i<num_points;i++) 
    {
        
        point[0] = points[i*3];
        point[1] = points[i*3+1];
        point[2] = points[i*3+2];

        point_scaling[0] = int(floor(((point[0] - bbox[0]) * resolution)));
        point_scaling[1] = int(floor(((point[1] - bbox[1]) * resolution)));
        point_scaling[2] = int(floor(((point[2] - bbox[2]) * resolution)));

        voxel = static_cast<CubeID>(point_scaling[0]) +
            static_cast<CubeID>(point_scaling[1]) * static_cast<CubeID>(orders[0]) +
            static_cast<CubeID>(point_scaling[2]) * static_cast<CubeID>(orders[1]);
        iter_0 = aux.find(voxel);
        if(iter_0 == aux.end())
        {
            aux.insert(make_pair(voxel, 1));
            voxels.insert(make_pair(voxel, point));
        }
        else
        {
            iter_1 = voxels.find(voxel);
            count = iter_0->second + 1;
            for(int j = 0; j < 3; j++)
            {
                point_merge[j] = (iter_0->second * iter_1->second[j] + point[j]) / count;
            }
            iter_1->second = point_merge;

            iter_0->second = count;
        }

    }
    
    return 0;
}



template<typename CubeID>
void points_to_voxels(unordered_map<CubeID, vector<float>> &voxels, 
    float* points, 
    int num_points, 
    float* bbox, 
    vector<int> &orders, 
    int resolution)
{
    get_bound(bbox, orders, resolution);

    voxelization<CubeID>(voxels, points, num_points, bbox, orders, resolution);
    // voxelization_by_readbin<CubeID>(model_name, voxels, bbox, orders, resolution);
}
