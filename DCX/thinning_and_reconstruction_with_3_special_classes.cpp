#include "happly.h"
#include "V2M_tables_with_3_special_classes.hpp"
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
#include <array>
#include <chrono>
using namespace std;
using namespace std::chrono;


#define BORDER_0 0
#define BORDER_1 5
string current_model_name;


template <typename T>
using Generate_Edges_Func = void(*)(T, int, int, std::unordered_map<T, std::pair<int, int>>&, std::unordered_map<T, std::pair<int, int>>&);
template <typename T>
using Generate_Faces_Func = void(*)(pair<T, int>, unordered_map<T, int>&, vector<T>&, vector<vector<T>>&, vector<vector<float>>&, set<vector<float>>&, unordered_map<T, vector<float>> &, T &); 
unordered_map<int, int> added_checking_for_class_3_map;
vector<vector<vector<int>>> added_check_for_class_11_thinning(8, vector<vector<int>>(3, vector<int>(3, 0)));

template <typename CubeID>
void get_cube_types(unordered_map<CubeID, vector<float>> &voxels, 
    unordered_map<CubeID, int> &cube_types)
{
    CubeID cube, neighbor;
    typename unordered_map<CubeID, int>::iterator iter;
    unordered_map<CubeID, int> temp;
    for(auto voxel : voxels)
    {
        for(vector<int> offset : cube_offsets)
        {
            cube = voxel.first + offset[0];
            iter = temp.find(cube);
            if(iter == temp.end())
            {
                temp.insert(make_pair(cube, offset[1]));
            }
            else 
            {
                iter->second += offset[1];
            }
        }
    }

    int cnt = 0;
    for(auto pair : temp)
    {
        if(cube_class[pair.second][0][0] != -1) cube_types.insert(pair);
    }
    temp.clear();
}

template <typename CubeID>
void erase_voxel(CubeID voxel, 
    unordered_map<CubeID, vector<float>> &voxels, 
    unordered_map<CubeID, int> &cube_types)
{
    CubeID cube;
    typename unordered_map<CubeID, int>::iterator iter;

    voxels.erase(voxel);
    for(auto offset : cube_offsets)
    {
        cube = voxel + offset[0];
        iter = cube_types.find(cube);
        if(iter != cube_types.end())
        {
            iter->second -= offset[1];
        }
    }
}

template <typename CubeID>
void generate_edges_for_class_0(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;

    for(int i = 1; i < edge_table[type][idx][0].size(); i++)
    {
        iter = required_edges.find(cube + edge_table[type][idx][0][i]);
        if(iter != required_edges.end()) iter->second.second++;
    }
}

template <typename CubeID>
void generate_edges_for_class_1(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;
    CubeID temp;
    temp = cube + neighbors_offset_by_cube_face[edge_table[type][idx][0][0]];
    if(local_cubes.find(temp) != local_cubes.end()) return;

    for(int i = 1; i < edge_table[type][idx][0].size(); i++)
    {
        iter = required_edges.find(cube + edge_table[type][idx][0][i]);
        if(iter != required_edges.end()) iter->second.second++;
    }
}

template <typename CubeID>
void generate_edges_for_class_2(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;
    CubeID temp;
    temp = cube + neighbors_offset_by_cube_face[edge_table[type][idx][0][0]];
    iter = local_cubes.find(temp);
    if(iter != local_cubes.end() && cube_class[iter->second.first][0][0] == 9) return;

    for(int i = 1; i < edge_table[type][idx][0].size(); i++)
    {
        iter = required_edges.find(cube + edge_table[type][idx][0][i]);
        if(iter != required_edges.end()) iter->second.second++;
    }
}

template <typename CubeID>
void generate_edges_for_class_3(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;
    CubeID temp;
    for(auto list : edge_table[type][idx])
    {
        if(list[0] != 0)
        {
            temp = cube + neighbors_offset_by_cube_face[list[0]];
            iter = local_cubes.find(temp);
            if(iter != local_cubes.end())
            {
                auto iter_temp = added_checking_for_class_3_map.find(type);
                if(iter_temp != added_checking_for_class_3_map.end())
                {
                    if(cube_class[iter->second.first][0][0] == 3) continue;
                    if(cube_class[iter->second.first][0][0] == 7 && iter_temp->second == iter->second.first) continue; 
                }
            }
                
        }

        for(int i = 1; i < list.size(); i++)
        {
            iter = required_edges.find(cube + list[i]);
            if(iter != required_edges.end()) iter->second.second++;
        }
    }
}

template <typename CubeID>
void generate_edges_for_class_4(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;
    CubeID temp;
    for(auto list : edge_table[type][idx])
    {
        temp = cube + neighbors_offset_by_cube_face[list[0]];
        if(local_cubes.find(temp) == local_cubes.end()) continue;
        else
        {
            iter = required_edges.find(cube + list[1]);
            if(iter != required_edges.end()) iter->second.second++;
        }
    }
}

template <typename CubeID>
void generate_edges_for_class_6(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;
    CubeID temp;
    for(auto list : edge_table[type][idx])
    {
        temp = cube + neighbors_offset_by_cube_face[list[0]];
        iter = local_cubes.find(temp);
        if(iter != local_cubes.end() && cube_class[iter->second.first][0][0] == 9) continue;
        
        for(int i = 1; i < list.size(); i++)
        {
            iter = required_edges.find(cube + list[i]);
            if(iter != required_edges.end()) iter->second.second++;
        }
    }
}

template <typename CubeID>
void generate_edges_for_class_8(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;
    CubeID temp;
    for(auto list : edge_table[type][idx])
    {
        temp = cube + neighbors_offset_by_cube_face[list[0]];
        iter = local_cubes.find(temp);
        if(list[0] % 2 == 1)
        {
            if(cube_class[iter->second.first][0][0] != 1) continue;
        }
        else
        {
            if(iter != local_cubes.end()) continue;
        }
        
        for(int i = 1; i < list.size(); i++)
        {
            iter = required_edges.find(cube + list[i]);
            if(iter != required_edges.end()) iter->second.second++;
        }
    }
}

template <typename CubeID>
void generate_edges_for_class_9(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    typename unordered_map<CubeID, pair<int, int>>::iterator iter;
    CubeID temp;
    for(auto list : edge_table[type][idx])
    {
        if(list[0] != 0)
        {
            temp = cube + neighbors_offset_by_cube_face[list[0]];
            iter = local_cubes.find(temp);
            if(iter != local_cubes.end() && cube_class[iter->second.first][0][0] == 9) continue;
        }
        
        for(int i = 1; i < list.size(); i++)
        {
            iter = required_edges.find(cube + list[i]);
            if(iter != required_edges.end()) iter->second.second++;
        }
    }
}

template <typename CubeID>
void generate_edges_for_class_10(CubeID cube, 
    int type, 
    int idx, 
    unordered_map<CubeID, pair<int, int>> &required_edges, 
    unordered_map<CubeID, pair<int, int>> &local_cubes)
{
    return;
}

template <typename CubeID>
bool checking_if_voxel_valid_for_class_11(CubeID cube, 
    int idx, 
    unordered_map<CubeID, vector<float>> &voxels)
{
    for(auto neighbors : added_check_for_class_11_thinning[idx])
    {
        if(voxels.find(cube + neighbors[0]) == voxels.end()) continue;
        if(voxels.find(cube + neighbors[1]) == voxels.end() && voxels.find(cube + neighbors[2]) == voxels.end()) 
            return true;
    }
    return false;
}
template<typename CubeID>
vector<Generate_Edges_Func<CubeID>> generate_edges = 
{
    generate_edges_for_class_0<CubeID>, 
    generate_edges_for_class_1<CubeID>, 
    generate_edges_for_class_2<CubeID>,
    generate_edges_for_class_3<CubeID>, 
    generate_edges_for_class_4<CubeID>, 
    generate_edges_for_class_0<CubeID>,
    generate_edges_for_class_6<CubeID>, 
    generate_edges_for_class_3<CubeID>, 
    generate_edges_for_class_8<CubeID>,
    generate_edges_for_class_9<CubeID>, 
    generate_edges_for_class_10<CubeID>
};

template <typename CubeID>
bool checking_if_voxel_valid(CubeID voxel, 
    unordered_map<CubeID, int> &cube_types,
     unordered_map<CubeID, vector<float>> &voxels)
{
    CubeID cube, partner;
    typename unordered_map<CubeID, int>::iterator iter;
    unordered_map<CubeID, pair<int, int>> required_edges;
    typename unordered_map<CubeID, pair<int, int>>::iterator iter_l;
    unordered_map<CubeID, pair<int, int>> local_cubes;
    vector<pair<CubeID, int>> special_cubes;

    for(auto offset : cube_offsets)
    {
        cube = voxel + offset[0];
        iter = cube_types.find(cube);
        if(iter != cube_types.end())
        {
            if(cube_class[iter->second][0][0] == -1) continue;
            if(voxel_type[iter->second][offset[2]][0] == 2) return true;
            if(cube_class[iter->second][0][0] == 11)
            {
                special_cubes.push_back(make_pair(cube, offset[2]));
                continue;
            }
            else if(cube_class[iter->second][0][0] == 10) return true;
            
            local_cubes.insert(make_pair(cube, make_pair(iter->second, offset[2])));
            for(int i = 1; i < voxel_type[iter->second][offset[2]].size(); i++)
            {
                partner = cube + voxel_type[iter->second][offset[2]][i];
                iter_l = required_edges.find(partner);

                if(!special_cubes.size() && voxel_type[iter->second][offset[2]][0]) return true;
                else
                {
                    partner = cube + voxel_type[iter->second][offset[2]][i];
                    required_edges.insert(make_pair(partner, make_pair(0, 0)));
                }
            }
        }
    }

    if(special_cubes.size())
    {
        for(auto cube_temp : special_cubes)
        {
            if(checking_if_voxel_valid_for_class_11<CubeID>(cube_temp.first, cube_temp.second, voxels)) return true;
        }

        return false;
    }

    if(!required_edges.size()) return true;

    for(auto info : local_cubes)
    {
        auto &generate_edges_func = generate_edges<CubeID>;
        generate_edges_func[cube_class[info.second.first][0][0]](info.first, info.second.first, info.second.second, required_edges, local_cubes);
    }

    for(auto edge : required_edges)
    {
        if(edge.second.first)
        {
            if(edge.second.second >= 3) return true;
        }
        else
        {
            if(edge.second.second >= 2) return true;
        }
    }

    return false;
}

template <typename CubeID>
void thinning(unordered_map<CubeID, vector<float>> &voxels, 
    unordered_map<CubeID, int> &cube_types)
{
    CubeID cube, neighbor;
    int count_0, count_1, count_2, round = 0, deleted_count = 0;
    typename unordered_map<CubeID, int>::iterator iter;
    unordered_map<CubeID, int> valid_cubes, invalid_cubes;
    unordered_set<CubeID> valid_voxels, invalid_voxels, invalid_cand, checking_cand;
    typename unordered_set<CubeID>::iterator iter_set;
    vector<CubeID> deleted_voxels, checking_list;
    vector<CubeID> *ptr_cur, *ptr_next;

    for(auto voxel : voxels)
    {
        count_0 = 0;
        count_1 = 0;

        for(vector<int> offset : cube_offsets)
        {
            cube = voxel.first + offset[0];
            iter = cube_types.find(cube);
            if(iter == cube_types.end()) continue;
            if(cube_class[iter->second][0][0] >= BORDER_0)
            {
                if(cube_class[iter->second][0][0] >= BORDER_1) count_1++;
                else count_0++;
            }

        }

        if(count_1 != 0)
        {
            invalid_cand.insert(voxel.first);
            count_2 = 0;
            for(auto offset : neighbors_offset_6)
            {
                if(voxels.find(voxel.first + offset) != voxels.end()) count_2++;
            }
            if(count_2 != 6) checking_list.push_back(voxel.first);
        }
        else
        {
            if(count_0 == 0) deleted_voxels.push_back(voxel.first);
        }
    }

    for(auto voxel : deleted_voxels) erase_voxel<CubeID>(voxel, voxels, cube_types);

    unordered_set<CubeID> erased_voxles;

    int num = 0;
    while(checking_list.size())
    {
        sort(checking_list.begin(), checking_list.end());
        num = 0;
        for(auto voxel : checking_list)
        {
            iter_set = invalid_cand.find(voxel);
            if(invalid_cand.find(voxel) == invalid_cand.end()) continue;

            if(!checking_if_voxel_valid<CubeID>(voxel, cube_types, voxels))
            {
                erase_voxel<CubeID>(voxel, voxels, cube_types);
                erased_voxles.insert(voxel);
                invalid_cand.erase(iter_set);
                for(auto offset : neighbors_offset_26)
                {
                    neighbor = voxel + offset;
                    if(invalid_cand.find(neighbor) != invalid_cand.end()) checking_cand.insert(neighbor);
                }
            }
        }

        checking_list.assign(checking_cand.begin(), checking_cand.end());
        checking_cand.clear();
    }
}



void initialize_tables()
{
    static const auto pure_voxel_type = voxel_type;
    static const auto pure_edge_table = edge_table;
    static bool is_first_run = true;

    if (!is_first_run)
    {
        voxel_type = pure_voxel_type;
        edge_table = pure_edge_table;
    }
    else
    {
        is_first_run = false;
    }

    for(int i = 0; i <= 255; i++)
    {
        if(voxel_type[i].size() > 0)
        {
            for(auto &voxel_partner : voxel_type[i])
            {
                for(int j = 1; j < voxel_partner.size(); j++)
                {
                    voxel_partner[j] = voxel_offsets[voxel_partner[j]];
                }
            }
        }
    }

    for(int i = 0; i <= 255; i++)
    {
        if(edge_table[i].size() > 0)
        {
            for(int j = 0; j < 8; j++)
            {
                if(edge_table[i][j].size() > 0)
                {
                    for(auto &voxel_partner : edge_table[i][j])
                    {
                        for(int k = 1; k < voxel_partner.size(); k++)
                        {
                            voxel_partner[k] = voxel_offsets[voxel_partner[k]];
                        }
                    }
                }
            }
        }
    }

    added_checking_for_class_3_map.clear();
    for(auto pair : added_checking_for_class_3)
    {
        added_checking_for_class_3_map.insert(make_pair(pair[0], pair[1]));
    }

    added_check_for_class_11_thinning = 
    {
        {{-order_2 - order_1, -order_2, -order_1}, {-order_2 - 1, -order_2, -1}, {-order_1 - 1, -order_1, -1}},
        {{-order_2 - order_1 + 1, -order_2 + 1, -order_1 + 1}, {-order_2 + 2, -order_2 + 1, 2}, {-order_1 + 2, -order_1 + 1, 2}},
        {{-order_2 + 2*order_1, -order_2 + order_1, 2*order_1}, {-order_2 + order_1 - 1, -order_2 + order_1, order_1 - 1}, {2*order_1 - 1, order_1-1, 2*order_1}},
        {{-order_2 + order_1 + 2, -order_2 + order_1 + 1, order_1 + 2}, {-order_2 + 2*order_1 + 1, -order_2 + order_1 + 1, 2*order_1 + 1}, {2*order_1 + 2, order_1 + 2, 2*order_1 + 1}},
        {{order_2 - order_1 - 1, order_2 - order_1, order_2 - 1}, {2*order_2 - order_1, order_2 - order_1, 2*order_2}, {2*order_2 - 1, order_2 - 1, 2*order_2}},
        {{order_2 - order_1 + 2, order_2 - order_1 + 1, order_2 + 2}, {2*order_2 - order_1 + 1, order_2 - order_1 + 1, 2*order_2 + 1}, {2*order_2 + 2, order_2 + 2, 2*order_2 + 1}},
        {{order_2 + 2*order_1 - 1, order_2 + order_1 - 1, order_2 + 2*order_1}, {2*order_2 + order_1 - 1, order_2 + order_1 - 1, 2*order_2 + order_1}, {2*order_2 + 2*order_1, order_2 + 2*order_1, 2*order_2 + order_1}},
        {{order_2 + 2*order_1 + 2, order_2 + 2*order_1 + 1, order_2 + order_1 + 2}, {2*order_2 + order_1 + 2, order_2 + order_1 + 2, 2*order_2 + order_1 + 1}, {2*order_2 + 2*order_1 + 1, order_2 + 2*order_1 + 1, 2*order_2 + order_1 + 1}}
    };

    edges_check = 
    {
        {-order_2 - order_1, -order_2 - order_1 + 1}, {-order_2 + 2*order_1, -order_2 + 2*order_1 + 1}, {2*order_2 - order_1, 2*order_2 - order_1 + 1}, {2*order_2 + 2*order_1, 2*order_2 + 2*order_1 + 1},
        {-order_2 - 1, -order_2 + order_1 - 1}, {-order_2 + 2, -order_2 + order_1 + 2}, {2*order_2 - 1, 2*order_2 + order_1 - 1}, {2*order_2 + 2, 2*order_2 + order_1 + 2},
        {-order_1 - 1, order_2 - order_1 - 1}, {-order_1 + 2, order_2 - order_1 + 2}, {2*order_1 - 1, order_2 + 2*order_1 - 1}, {2*order_1 + 2, order_2 + 2*order_1 + 2}
    };
}


vector<vector<int>> central_face_bound_with_edge = 
{
    {0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}
};


vector<vector<int>> cube_face_bound_with_edges = 
{
    {2, 4}, {3, 4}, {2, 5}, {3, 5}, {0, 4}, {1, 4}, {0, 5}, {1, 5}, {0, 2}, {1, 2}, {0, 3}, {1, 3}
};


vector<vector<int>> surrounding_face = 
{
    {0, 2, 4, 6},
    {1, 3, 5, 7},
    {0, 1, 4, 5},
    {2, 3, 6, 7},
    {0, 1, 2, 3},
    {4, 5, 6, 7}
};

template <typename CubeID>
void generate_faces_for_class_0(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    vector<CubeID> face(3);
    for(auto temp : face_table[cube.second][0])
    {
        for(int idx = 0; idx < 3; idx++) face[idx] = voxels_order[temp[idx]];
        faces.push_back(face);
    }
}

template <typename CubeID>
void generate_faces_for_class_1(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    int neighbor_idx = cube_class[cube.second][1][0];
    vector<CubeID> face(3);
    auto temp = face_table[cube.second][neighbor_idx][0];

    if(cube_types.find(cube.first + neighbors_offset_by_cube_face[neighbor_idx]) != cube_types.end()) return;

    for(int idx = 0; idx < 3; idx++) face[idx] = voxels_order[temp[idx]];
    faces.push_back(face);
}

template <typename CubeID>
void generate_faces_for_class_2(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    int neighbor_idx = cube_class[cube.second][1][0];
    vector<CubeID> face(3);
    typename unordered_map<CubeID, int>::iterator iter = cube_types.find(cube.first + neighbors_offset_by_cube_face[neighbor_idx]);

    if(iter != cube_types.end() && cube_class[iter->second][0][0] == 9) return;

    for(auto temp : face_table[cube.second][neighbor_idx])
    {
        for(int idx = 0; idx < 3; idx++) face[idx] = voxels_order[temp[idx]];
        faces.push_back(face);
    }
}

template <typename CubeID>
void generate_faces_for_class_3(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    CubeID neighbor;
    vector<CubeID> face(3);
    typename unordered_map<CubeID, int>::iterator iter;

    for(auto temp : face_table[cube.second][0])
    {
        for(int idx = 0; idx < 3; idx++) face[idx] = voxels_order[temp[idx]];
        faces.push_back(face);
    }

    if(cube_class[cube.second][1].size() == 2)
    {
        neighbor = cube.first + neighbors_offset_by_cube_face[cube_class[cube.second][1][1]];
        iter = cube_types.find(neighbor);
        if(iter != cube_types.end())
        {
            auto iter_temp = added_checking_for_class_3_map.find(cube.second);
            if(iter_temp != added_checking_for_class_3_map.end())
            {
                if(cube_class[iter->second][0][0] == 3) return;
                if(cube_class[iter->second][0][0] == 7 && iter_temp->second == iter->second) return;
            }
            
        }
        
        auto temp = face_table[cube.second][cube_class[cube.second][1][1]][0];
        for(int idx = 0; idx < 3; idx++) face[idx] = voxels_order[temp[idx]];
        faces.push_back(face);
    }
}

template <typename CubeID>
void generate_faces_for_class_4(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    typename unordered_map<CubeID, vector<float>>::iterator iter;
    vector<float> new_vertice(3, 0);
    vector<CubeID> face(3);
    for(int i = 0; i < 8; i++)
    {
        if(voxels_order[i] == -1) continue;
        iter = voxels.find(cube.first + voxel_offsets[i]);
        for(int j = 0; j < 3; j++)
        {
            new_vertice[j] += iter->second[j];
        }
    }

    for(int j = 0; j < 3; j++)
    {
        new_vertice[j] /= 4;
    }
    new_vertices.push_back(new_vertice);
    
    for(int i = 0; i < 6; i++)
    {
        if(cube_types.find(cube.first + neighbors_offset_6[i]) != cube_types.end())
        {
            face = {voxels_order[face_table[cube.second][i+1][0][0]], voxels_order[face_table[cube.second][i+1][0][1]], count};
            faces.push_back(face);
        }
    }

    count++;
}

template <typename CubeID>
void generate_faces_for_class_6(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    CubeID neighbor;
    vector<CubeID> face(3);
    typename unordered_map<CubeID, int>::iterator iter;

    for(auto i : cube_class[cube.second][1])
    {
        neighbor = cube.first + neighbors_offset_by_cube_face[i];
        iter = cube_types.find(neighbor);
        if(iter != cube_types.end() && cube_class[iter->second][0][0] == 9) continue;

        for(auto temp : face_table[cube.second][i])
        {
            for(int idx = 0; idx < 3; idx++) face[idx] = voxels_order[temp[idx]];
            faces.push_back(face);
        }
    }
}

template <typename CubeID>
void generate_faces_for_class_8(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    CubeID neighbor;
    vector<CubeID> face(3);
    typename unordered_map<CubeID, int>::iterator iter;

    for(auto i : cube_class[cube.second][1])
    {
        neighbor = cube.first + neighbors_offset_by_cube_face[i];
        iter = cube_types.find(neighbor);
        if(i % 2)
        {
            if(iter != cube_types.end() && cube_class[iter->second][0][0] != 1) continue;
        }
        else
        {
            if(iter != cube_types.end()) continue;
        }

        auto temp = face_table[cube.second][i][0];
        for(int idx = 0; idx < 3; idx++) face[idx] = voxels_order[temp[idx]];
        faces.push_back(face);
    }
}

template <typename CubeID>
void generate_faces_for_class_9(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    CubeID neighbor, idx_e, new_idx_surrounding;
    vector<CubeID> face(3);
    typename unordered_map<CubeID, int>::iterator iter;
    typename unordered_map<CubeID, vector<float>>::iterator iter_v;
    vector<float> new_vertice(3, 0);

    neighbor = cube.first + neighbors_offset_by_cube_face[cube_class[cube.second][1][1]];
    iter = cube_types.find(neighbor);
    if(iter != cube_types.end() && cube_class[iter->second][0][0] == 9)
    {
        for(int i : surrounding_face[cube_class[cube.second][1][1] - 1])
        {
            iter_v = voxels.find(cube.first + voxel_offsets[i]);
            for(int j = 0; j < 3; j++)
            {
                new_vertice[j] += iter_v->second[j];
            }
        }

        for(int j = 0; j < 3; j++)
        {
            new_vertice[j] /= 4;
        }

        if(new_vertices_aux.find(new_vertice) == new_vertices_aux.end())
        {
            new_vertices.push_back(new_vertice);
            new_idx_surrounding = count;
            count++;
            new_vertices_aux.insert(new_vertice);
        }
        else
        {
           for(CubeID k = 0; k < new_vertices.size(); k++)
           {
                if(new_vertices[k][0] == new_vertice[0] && new_vertices[k][1] == new_vertice[1] && new_vertices[k][2] == new_vertice[2])
                {
                    new_idx_surrounding = voxels.size() + k;
                }
           }
        }

        face[0] = new_idx_surrounding;
        if(cube_class[cube.second][1][1] % 2 == 0)
        {
            idx_e = 4;
        }
        else
        {
            idx_e = 2;
        }

        for(CubeID idx = idx_e; idx < face_table[cube.second][0].size(); idx++)
        {
            auto temp = face_table[cube.second][0][idx];
            for(int i = 0; i < 2; i++) face[i + 1] = voxels_order[temp[i]];
            faces.push_back(face);
        }
    }
    else
    {
        if(cube_class[cube.second][1][1] % 2 == 0)
        {
            idx_e = 4;
        }
        else
        {
            idx_e = 2;
            for(int idx = 0; idx < 2; idx++)
            {
                auto temp = face_table[cube.second][cube_class[cube.second][1][1]][idx];
                for(int i = 0; i < 3; i++) face[i] = voxels_order[temp[i]];
                faces.push_back(face);
            }
        }

        for(CubeID idx = 0; idx < idx_e; idx++)
        {
            auto temp = face_table[cube.second][0][idx];
            for(int i = 0; i < 3; i++) face[i] = voxels_order[temp[i]];
            faces.push_back(face);
        }
    }
}

template <typename CubeID>
void generate_faces_for_class_10(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    vector<int> edges_flag(12, 0);
    typename unordered_map<CubeID,  vector<float>>::iterator iter;
    typename unordered_map<CubeID, int>::iterator iter_0;
    vector<float> new_vertice(3, 0);
    vector<CubeID> face(3);
    bool flag, flag_same_class;

    for(int i = 0; i < 8; i++)
    {
        if(voxels_order[i] == -1) continue;
        iter = voxels.find(cube.first + voxel_offsets[i]);
        for(int j = 0; j < 3; j++)
        {
            new_vertice[j] += iter->second[j];
        }
    }

    for(int j = 0; j < 3; j++)
    {
        new_vertice[j] /= 6;
    }
    new_vertices.push_back(new_vertice);

    for(int i = 0; i < 6; i++)
    {
        iter_0 = cube_types.find(cube.first + neighbors_offset_6[i]);
        if(iter_0 == cube_types.end()) continue;
        flag = false;
        flag_same_class = false;
        for(int j = 0; j < 3; j++)
        {
            if(added_checking_for_class_10[cube_class[cube.second][1][0]][i][j] == iter_0->second)
            {
                flag = true;
                if(j == 2) flag_same_class = true;
            }
        }
        if(flag)
        {
            face[0] = voxels_order[face_table_for_class_10[cube_class[cube.second][1][0]][i][1][0]];
            face[1] = voxels_order[face_table_for_class_10[cube_class[cube.second][1][0]][i][1][1]];
            face[2] = count;
            faces.push_back(face);
            if(flag_same_class && i % 2 == 1)
            {
                face[0] = voxels_order[face_table_for_class_10[cube_class[cube.second][1][0]][i][0][0]];
                face[1] = voxels_order[face_table_for_class_10[cube_class[cube.second][1][0]][i][0][1]];
                face[2] = voxels_order[face_table_for_class_10[cube_class[cube.second][1][0]][i][0][2]];
                faces.push_back(face);
            }
            for(auto k : edges_bounded_with_face[i]) edges_flag[k]++;
        }
    }
    for(auto k : present_cube_edges_for_class_10[cube_class[cube.second][1][0]])
    {
        if(edges_flag[k] == 1) continue;
        else
        {
            face[0] = voxels_order[cube_edges[k][0]];
            face[1] = voxels_order[cube_edges[k][1]];
            face[2] = count;
            faces.push_back(face);
        }
    }

    count++;
}


template <typename CubeID>
void generate_faces_for_class_11(pair<CubeID, int> cube, 
    unordered_map<CubeID, int> &cube_types, 
    vector<CubeID> &voxels_order, 
    vector<vector<CubeID>> &faces, 
    vector<vector<float>> &new_vertices, 
    set<vector<float>> &new_vertices_aux, 
    unordered_map<CubeID, vector<float>> &voxels, 
    CubeID &count)
{
    vector<bool> cube_face(6, false);
    typename unordered_map<CubeID, vector<float>>::iterator iter;
    vector<float> new_vertice(3, 0);
    vector<CubeID> face(3);
    CubeID new_idx_surrounding, new_idx_central = count;
    vector<vector<CubeID>> faces_added;

    for(int i = 0; i < 8; i++)
    {
        iter = voxels.find(cube.first + voxel_offsets[i]);
        for(int j = 0; j < 3; j++)
        {
            new_vertice[j] += iter->second[j];
        }
    }

    for(int j = 0; j < 3; j++)
    {
        new_vertice[j] /= 8;
    }
    new_vertices.push_back(new_vertice);

    count++;
    
    for(int i = 0; i < 12; i++)
    {
        if(voxels.find(cube.first + edges_check[i][0]) != voxels.end() || voxels.find(cube.first + edges_check[i][1]) != voxels.end())
        {
            face = {voxels_order[central_face_bound_with_edge[i][0]], voxels_order[central_face_bound_with_edge[i][1]], new_idx_central};
            faces.push_back(face);
            faces_added.push_back({central_face_bound_with_edge[i][0], central_face_bound_with_edge[i][1], -1});

            for(int j : cube_face_bound_with_edges[i]) cube_face[j] = true;
        }
    }

    for(int i = 0; i < 6; i++)
    {
        if(!cube_face[i])
        {
            new_vertice = {0, 0, 0};
            for(int j : surrounding_face[i])
            {
                iter = voxels.find(cube.first + voxel_offsets[j]);
                for(int k = 0; k < 3; k++)
                {
                    new_vertice[k] += iter->second[k];
                }
            }

            for(int j = 0; j < 3; j++)
            {
                new_vertice[j] /= 4;
            }

            if(new_vertices_aux.find(new_vertice) == new_vertices_aux.end())
            {
                new_vertices.push_back(new_vertice);
                new_idx_surrounding = count;
                count++;
                new_vertices_aux.insert(new_vertice);
            }
            else
            {
               for(CubeID k = 0; k < new_vertices.size(); k++)
               {
                    if(new_vertices[k][0] == new_vertice[0] && new_vertices[k][1] == new_vertice[1] && new_vertices[k][2] == new_vertice[2])
                    {
                        new_idx_surrounding = voxels.size() + k;
                    }
               }
            }
            
            for(int k : surrounding_face[i])
            {
                faces.push_back({voxels_order[k], new_idx_central, new_idx_surrounding});
                faces_added.push_back({k, -1, i+10});
            }
        }
    }
    return;
}

template<typename CubeID>
vector<Generate_Faces_Func<CubeID>> generate_faces = 
{
    generate_faces_for_class_0<CubeID>, 
    generate_faces_for_class_1<CubeID>, 
    generate_faces_for_class_2<CubeID>, 
    generate_faces_for_class_3<CubeID>, 
    generate_faces_for_class_4<CubeID>, 
    generate_faces_for_class_0<CubeID>, 
    generate_faces_for_class_6<CubeID>, 
    generate_faces_for_class_3<CubeID>, 
    generate_faces_for_class_8<CubeID>, 
    generate_faces_for_class_9<CubeID>, 
    generate_faces_for_class_10<CubeID>, 
    generate_faces_for_class_11<CubeID>
};


struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1);
    }
};

template <typename CubeID>
inline array<pair<CubeID, CubeID>, 3> get_sorted_face_edges(const vector<CubeID> &face)
{
    array<CubeID, 3> sorted_face = {face[0], face[1], face[2]};
    sort(sorted_face.begin(), sorted_face.end());
    return {
        make_pair(sorted_face[0], sorted_face[1]),
        make_pair(sorted_face[0], sorted_face[2]),
        make_pair(sorted_face[1], sorted_face[2])
    };
}

template <typename CubeID>
inline void increase_edge_count(unordered_map<pair<CubeID, CubeID>, int, pair_hash> &edges_stat,
    const pair<CubeID, CubeID> &edge)
{
    auto iter = edges_stat.emplace(edge, 0);
    ++iter.first->second;
}

template <typename CubeID>
inline void decrease_edge_count(unordered_map<pair<CubeID, CubeID>, int, pair_hash> &edges_stat,
    const pair<CubeID, CubeID> &edge)
{
    --edges_stat.find(edge)->second;
}


template <typename CubeID>
void selecting_special_vertices(vector<vector<CubeID>> &faces, unordered_set<CubeID> &vertices_margin)
{
    unordered_map<pair<CubeID, CubeID>, int, pair_hash> edges_stat;
    typename unordered_map<pair<CubeID, CubeID>, int, pair_hash>::iterator iter;
    pair<CubeID, CubeID> edge;
    vector<pair<int, int>> point_pair = {make_pair(0, 1), make_pair(0, 2), make_pair(1, 2)};

    for(auto face : faces)
    {
        auto temp = face;
        sort(temp.begin(), temp.end());
        for(int i = 0; i < 3; i++)
        {
            edge = make_pair(temp[point_pair[i].first], temp[point_pair[i].second]);
            iter = edges_stat.find(edge);
            if(iter == edges_stat.end())
            {
                edges_stat.insert(make_pair(edge, 1));
            }
            else
            {
                iter->second++;
            }
        }
    }

    for(auto edge_temp : edges_stat)
    {
        if(edge_temp.second == 1)
        {
            vertices_margin.insert(edge_temp.first.first);
            vertices_margin.insert(edge_temp.first.second);
        }
    }
}

template <typename CubeID>
unordered_set<CubeID> delete_redundant_faces(vector<vector<CubeID>> &faces, int points_num, string model_name, vector<bool> &faces_flag)
{
    (void)model_name;

    vector<unordered_set<CubeID>> edges(points_num);
    vector<vector<CubeID>> faces_attached(points_num);
    unordered_map<pair<CubeID, CubeID>, int, pair_hash> edges_stat;
    edges_stat.reserve(faces.size() * 3);
    pair<CubeID, CubeID> edge;
    unordered_set<CubeID> vertices_margin, vertices_margin_temp;
    unordered_map<CubeID, int> vertices_nmf;
    unordered_set<CubeID> cand_0, cand_1, vertices_attaching_nmv_temp;
    unordered_map<CubeID, vector<typename unordered_map<pair<CubeID, CubeID>, int, pair_hash>::iterator>> vertices_attaching_nmv;
    unordered_set<CubeID> *p_cur = &cand_0, *p_next = &cand_1; 
    bool flag, flag_inside_loop = true, flag_outside_loop = true;
    unordered_set<CubeID> vertices_deleted, vertices_valid;
    vector<CubeID> nmv_attached;
    vector<typename unordered_map<pair<CubeID, CubeID>, int, pair_hash>::iterator> nme_attached;
    int count_deleted_pre_round = 0, count_deleted_record = 0, face_count = 0;
    unordered_set<pair<CubeID, CubeID>, pair_hash> edges_margin, edges_nmf;
    

    for(const auto &face : faces)
    {
        auto face_edges = get_sorted_face_edges(face);
        for(const auto &face_edge : face_edges)
        {
            increase_edge_count(edges_stat, face_edge);
        }
        
        edges[face[0]].insert(face[1]);
        edges[face[0]].insert(face[2]);
        edges[face[1]].insert(face[0]);
        edges[face[1]].insert(face[2]);
        edges[face[2]].insert(face[0]);
        edges[face[2]].insert(face[1]);
        faces_attached[face[0]].push_back(face_count);
        faces_attached[face[1]].push_back(face_count);
        faces_attached[face[2]].push_back(face_count);
        face_count++;
    }

    while(flag_outside_loop)
    {
        count_deleted_record = vertices_deleted.size();
        while(flag_inside_loop)
        {

            count_deleted_pre_round = vertices_deleted.size();
            vertices_nmf.clear();
            vertices_margin_temp.clear();
            vertices_margin.clear();
            for(const auto &edge_temp : edges_stat)
            {
                if(edge_temp.second >= 3)
                {
                    vertices_nmf.emplace(edge_temp.first.first, false);
                    vertices_nmf.emplace(edge_temp.first.second, false);
                }
                else if(edge_temp.second == 1)
                {
                    vertices_margin_temp.insert(edge_temp.first.first);
                    vertices_margin_temp.insert(edge_temp.first.second);
                }
            }
            for(const auto &v : vertices_margin_temp)
            {
                if(vertices_nmf.find(v) == vertices_nmf.end())
                {
                    vertices_margin.insert(v);
                }
            }
            int cnt = 0;
            for(auto &vertice : vertices_nmf)
            {
                
                if(vertice.second == false)
                {
                    vertices_attaching_nmv.clear();
                    vertices_attaching_nmv_temp.clear();
                    vertice.second = true;
                    (*p_cur).clear();
                    (*p_next).clear();
                    (*p_cur).insert(vertice.first);
                    while(!p_cur->empty())
                    {
                        for(auto v : (*p_cur))
                        {
                            vertices_nmf.find(v)->second = true;
                            for(auto partner : edges[v])
                            {

                                auto iter = vertices_nmf.find(partner);
                                if(iter != vertices_nmf.end())
                                {
                                    if(iter->second == false)
                                    {
                                        (*p_next).insert(partner);
                                    }
                                }
                                else
                                {
                                    vertices_attaching_nmv_temp.insert(partner);
                                }
                            }
                        }

                        std::swap(p_cur, p_next);
                        (*p_next).clear();
                    }
                    for(auto v : vertices_attaching_nmv_temp)
                    {
                        flag = false;
                        nmv_attached.clear();

                        if(vertices_margin.find(v) != vertices_margin.end()) flag = true;

                        for(auto partner: edges[v])
                        {
                            if(vertices_deleted.find(partner) != vertices_deleted.end()) continue;
                            if(vertices_margin.find(partner) != vertices_margin.end())
                            {
                                flag = true;
                            }
                            else if(vertices_nmf.find(partner) != vertices_nmf.end())
                            {
                                nmv_attached.push_back(partner);
                            }
                            else if(vertices_attaching_nmv_temp.find(partner) == vertices_attaching_nmv_temp.end())
                            {
                                flag = false;
                                break;
                            }
                        }
                        if(!flag || nmv_attached.size() < 1) continue;
                        else if(nmv_attached.size() == 1)
                        {
                            vertices_deleted.insert(v);
                            cnt++;
                            for(auto face_attached : faces_attached[v])
                            {
                                if(faces_flag[face_attached] == false) continue;

                                faces_flag[face_attached] = false;
                                auto face_edges = get_sorted_face_edges(faces[face_attached]);
                                for(const auto &face_edge : face_edges)
                                {
                                    decrease_edge_count(edges_stat, face_edge);
                                }

                            }
                        }
                        else
                        {
                            nme_attached.clear();
                            std::sort(nmv_attached.begin(), nmv_attached.end());
                            for(int i = 0; i < nmv_attached.size() - 1; i++)
                            {
                                for(int j = i + 1; j < nmv_attached.size(); j++)
                                {
                                    auto pair_temp = make_pair(nmv_attached[i], nmv_attached[j]);
                                    auto iter_temp = edges_stat.find(pair_temp);
                                    if(iter_temp != edges_stat.end() && iter_temp->second >= 3)
                                    {
                                        nme_attached.push_back(iter_temp);
                                    }
                                }
                            }
                            
                            if(!nme_attached.empty()) vertices_attaching_nmv.insert(make_pair(v, nme_attached));
                            else
                            {
                                vertices_deleted.insert(v);
                                cnt++;
                                for(auto face_attached : faces_attached[v])
                                {
                                    if(faces_flag[face_attached] == false) continue;

                                    faces_flag[face_attached] = false;
                                    auto face_edges = get_sorted_face_edges(faces[face_attached]);
                                    for(const auto &face_edge : face_edges)
                                    {
                                        decrease_edge_count(edges_stat, face_edge);
                                    }

                                }
                            }

                        }
                    }
                    vertices_valid.clear();
                    for(const auto &v_attaching_nmv : vertices_attaching_nmv)
                    {
                        if(vertices_deleted.find(v_attaching_nmv.first) != vertices_deleted.end() 
                            || vertices_valid.find(v_attaching_nmv.first) != vertices_valid.end()) continue;

                        (*p_cur).clear();
                        (*p_cur).insert(v_attaching_nmv.first);
                        (*p_next).clear();
                        while(!p_cur->empty())
                        {
                            for(auto v : (*p_cur))
                            {
                                flag = true;
                                for(auto iter_temp : vertices_attaching_nmv.find(v)->second)
                                {
                                    if(iter_temp->second <= 2)
                                    {
                                        flag = false;
                                        break;
                                    }
                                }

                                if(flag)
                                {
                                    vertices_deleted.insert(v);
                                    cnt++;
                                    for(auto partner : edges[v])
                                    {
                                        if(vertices_deleted.find(partner) != vertices_deleted.end())
                                        {
                                            continue;
                                        }

                                        if(vertices_attaching_nmv.find(partner) != vertices_attaching_nmv.end() && vertices_valid.find(partner) == vertices_valid.end())
                                        {
                                            (*p_next).insert(partner);
                                        }
                                    }

                                    for(auto face_attached : faces_attached[v])
                                    {
                                        if(faces_flag[face_attached] == false) continue;

                                        faces_flag[face_attached] = false;
                                        auto face_edges = get_sorted_face_edges(faces[face_attached]);
                                        for(const auto &face_edge : face_edges)
                                        {
                                            decrease_edge_count(edges_stat, face_edge);
                                        }
                                    }
                                }
                                else vertices_valid.insert(v);
                            }

                            std::swap(p_cur, p_next);
                            (*p_next).clear();
                        }
                    }
                    
                }
            }

            if(vertices_deleted.size() == count_deleted_pre_round) flag_inside_loop = false;

        }
        flag_inside_loop = true;
        count_deleted_pre_round = 0;
        count_deleted_record = 0;
        edges_margin.clear();
        edges_nmf.clear();
        vertices_nmf.clear();
        for(const auto &edge_temp : edges_stat)
        {
            if(edge_temp.second >= 3)
            {
                edges_nmf.insert(edge_temp.first);
                // if(edges)
            }
            else if(edge_temp.second == 1)
            {
                edges_margin.insert(edge_temp.first);
            }
        }

        while(flag_inside_loop)
        {
            count_deleted_pre_round = count_deleted_record;
            vector<pair<CubeID, CubeID>> edges_margin_t(edges_margin.begin(), edges_margin.end());
            for(const auto &edge_temp : edges_margin_t)
            {
                for(auto face_attached : faces_attached[edge_temp.first])
                {
                    if(faces_flag[face_attached] == false) continue;

                    if(faces[face_attached][0] == edge_temp.second || faces[face_attached][1] == edge_temp.second || faces[face_attached][2] == edge_temp.second)
                    {
                        auto face_edges = get_sorted_face_edges(faces[face_attached]);
                        flag = false;
                        for(const auto &face_edge : face_edges)
                        {
                            if(face_edge.first == edge_temp.first && face_edge.second == edge_temp.second) continue;

                            if(edges_nmf.find(face_edge) != edges_nmf.end())
                            {
                                flag = true;
                            }
                        }

                        if(!flag)
                        {
                            auto flag_temp = false;
                            for(auto v : faces[face_attached])
                            {
                                flag_temp = false;
                                for(auto partner : edges[v])
                                {
                                    if(v > partner) edge = make_pair(partner, v);
                                    else edge = make_pair(v, partner);

                                    if(edges_nmf.find(edge) != edges_nmf.end())
                                    {
                                        flag_temp = true;
                                        break;
                                    }
                                }

                                if(flag_temp == false) break;
                            }

                            if(flag_temp) flag = true;
                        }

                        if(flag)
                        {
                            faces_flag[face_attached] = false;
                            count_deleted_record++;
                            for(const auto &face_edge : face_edges)
                            {
                                if(edges_margin.find(face_edge) != edges_margin.end())
                                {
                                    edges_margin.erase(face_edge);
                                    decrease_edge_count(edges_stat, face_edge);
                                }
                                else if(edges_nmf.find(face_edge) != edges_nmf.end())
                                {
                                    auto iter_edge = edges_stat.find(face_edge);
                                    iter_edge->second--;
                                    if(iter_edge->second <= 2)
                                    {
                                        edges_margin.erase(face_edge);
                                    }
                                }
                                else
                                {
                                    edges_margin.insert(face_edge);
                                    decrease_edge_count(edges_stat, face_edge);
                                }
                            }
                        }
                    }
                }
            }

            if(count_deleted_record == count_deleted_pre_round) flag_inside_loop = false;
        }
        if(count_deleted_record > 0) flag_outside_loop = true;
        else flag_outside_loop = false;
    }

    return vertices_deleted;
}

template <typename CubeID>
void reconstruction(unordered_map<CubeID, vector<float>> &voxels, 
    unordered_map<CubeID, int> &cube_types, 
    int pattern, 
    unordered_set<CubeID> &margin_voxels,
    bool enable_postprocessing,
    vector<vector<float>> &out_vertices,
    vector<vector<CubeID>> &out_faces)
{
    unordered_map<CubeID, int> voxel_with_order;
    vector<CubeID> voxels_order(8);
    vector<vector<CubeID>> faces;
    typename unordered_map<CubeID, int>::iterator iter;
    vector<vector<float>> new_vertices;
    set<vector<float>> new_vertices_aux;
    CubeID voxel_temp, count = voxels.size(), idx = 0;
    out_vertices.clear();
    out_faces.clear();

    for(auto voxel : voxels)
    {
        if(pattern)voxel_with_order.insert(make_pair(voxel.first, voxel.first));
        else voxel_with_order.insert(make_pair(voxel.first, idx++));
    }

    for(auto cube : cube_types)
    {
        for(int i = 0; i < 8; i++)
        {
            if(voxel_type[cube.second][i].size())
            {
                voxel_temp = cube.first + voxel_offsets[i];
                iter = voxel_with_order.find(voxel_temp);
                if(iter == voxel_with_order.end()) cout << "Error happened in generate face stage while processing " << cube.first << "th cube!" << endl;
                else voxels_order[i] = iter->second;
            }
            else voxels_order[i] = -1;
        }
        if(cube_class[cube.second][0][0] != -1)
        {
            auto &generate_faces_func = generate_faces<CubeID>;
            generate_faces_func[cube_class[cube.second][0][0]](cube, cube_types, voxels_order, faces, new_vertices, new_vertices_aux, voxels, count);
        }
    }
    
    if(pattern)
    {
        selecting_special_vertices<CubeID>(faces, margin_voxels);
        return;
    }

    for(auto const& voxel : voxels)
    {
        out_vertices.push_back(voxel.second);
    }
    for(auto const& v : new_vertices)
    {
        out_vertices.push_back(v);
    }

    if(enable_postprocessing == false)
    {
        out_faces = faces;
        return;
    }
    vector<bool> faces_flag(faces.size(), true);
    if(enable_postprocessing)delete_redundant_faces<CubeID>(faces, voxels.size() + new_vertices.size(), current_model_name, faces_flag);
    for(int idx_f = 0; idx_f < faces.size(); idx_f++)
    {
        if(faces_flag[idx_f]) {
            out_faces.push_back(faces[idx_f]);
        }
    }
    
}

template <typename CubeID>
void filling_holes_prepare(unordered_set<CubeID> margin_voxels,
    unordered_map<CubeID, vector<float>> &voxels,
    float* bbox,
    int res,
    vector<CubeID> &added_voxels_cand_v,
    vector<vector<float>> &query_points)
{
    CubeID neighbor;
    unordered_set<CubeID> added_voxels_cand;
    vector<float> query_point(3), query_point_raw(3);
    float query_offset = 1.0 / res / 8;

    added_voxels_cand_v.clear();
    query_points.clear();

    for(auto voxel : margin_voxels)
    {
        for(auto offset : neighbors_offset_26)
        {
            neighbor = voxel + offset;
            if(voxels.find(neighbor) == voxels.end()) added_voxels_cand.insert(neighbor);
        }
    }

    added_voxels_cand_v.assign(added_voxels_cand.begin(), added_voxels_cand.end());
    sort(added_voxels_cand_v.begin(), added_voxels_cand_v.end());

    for(auto voxel : added_voxels_cand_v)
    {
        query_point_raw[2] = float(voxel / order_2) / resolution + bbox[2] - 3 * query_offset;
        query_point_raw[1] = float(voxel % order_2 / order_1) / resolution + bbox[1] - 3 * query_offset;
        query_point_raw[0] = float(voxel % order_2 % order_1) / resolution + bbox[0] - 3 * query_offset;

        query_point[2] = query_point_raw[2];
        for(int i = 0; i < 7; i++)
        {
            query_point[1] = query_point_raw[1];
            for(int j = 0; j < 7; j++)
            {
                query_point[0] = query_point_raw[0];
                for(int k = 0; k < 7; k++)
                {
                    query_points.push_back(query_point);
                    query_point[0] += query_offset;
                }

                query_point[1] += query_offset;
            }

            query_point[2] += query_offset;
        }
    }
}

template <typename CubeID>
void filling_holes_apply(
    const vector<CubeID> &selected_voxels,
    const vector<vector<float>> &selected_points,
    unordered_map<CubeID, vector<float>> &voxels,
    unordered_map<CubeID, vector<float>> &added_voxels)
{
    for(size_t i = 0; i < selected_voxels.size(); ++i)
    {
        added_voxels.insert(make_pair(selected_voxels[i], selected_points[i]));
        voxels.insert(make_pair(selected_voxels[i], selected_points[i]));
    }
}

template <typename CubeID>
void filling_holes(string model_name, 
    unordered_set<CubeID> margin_voxels, 
    unordered_map<CubeID, vector<float>> &voxels, 
    unordered_map<CubeID, vector<float>> &added_voxels, 
    float* bbox, 
    int res)

{
    vector<CubeID> added_voxels_cand_v;
    vector<vector<float>> query_points;
    filling_holes_prepare<CubeID>(margin_voxels, voxels, bbox, res, added_voxels_cand_v, query_points);
}
