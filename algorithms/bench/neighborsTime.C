// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parse_command_line.h"
#include "time_loop.h"
#include "../utils/NSGDist.h"
#include "../utils/euclidian_point.h"
#include "../utils/point_range.h"
#include "../utils/mips_point.h"
#include "../utils/graph.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>

#include "../HNSW/HNSW.hpp"
#include "../HNSW/dist.hpp"
#include "../utils/euclidian_point.h"
#include <spdlog/spdlog.h>

using namespace parlayANN;

using uint = unsigned int;

std::vector<descr_l2<float>::type_point> generate_random_points(size_t num_points, size_t dimensions)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    static std::vector<std::vector<float>> all_coords(num_points, std::vector<float>(dimensions));

    std::vector<descr_l2<float>::type_point> points;
    points.reserve(num_points);

    for (size_t i = 0; i < num_points; ++i)
    {
        std::vector<float> coords(dimensions);
        for (size_t j = 0; j < dimensions; ++j)
        {   all_coords[i][j] = dist(gen);
            coords[j] = dist(gen);
        }

        point<float> p(i, all_coords[i].data());
        points.push_back(p);
    }

    return points;
}

void test() {
    auto points = generate_random_points(100, 3);


    time_loop(1, 0,
    [&] () {},
    [&] () {
        // ANN::HNSW<descr_l2<float>>(G, k, BP, Query_Points, GT, res_file, graph_built, Points);
        ANN::HNSW<descr_l2<float>>(points.begin(), points.end(), 3);
        // ANN::HNSW<Point>(G, k, BP, Query_Points, GT, res_file, graph_built, Points);
    },
    [&] () {});

}

int main(int argc, char* argv[]) {
    auto points = generate_random_points(200, 3);
    for (auto v : points) {
        auto *a = v.coord;
        spdlog::info("start !!! {} {} {}", a[0], a[1], a[2]);
    }
    auto hnsw = new ANN::HNSW<descr_l2<float>>(points.begin(), points.end(), 3);
    std::cout << "Finished insertion" << std::endl;
    float coords1[] = {1.0f, 2.0f, 3.0f};
    point<float> point1{1, coords1};
    search_control ctrl{};
    auto res = hnsw->search(point1, 10, 50, ctrl);
    std::cout << "-----------" << std::endl;
    std::cout << res.size() << std::endl;
    for (auto r : res) {
        spdlog::info("id {} dist {}", r.first, r.second);
    }
    std::cout << "********" << std::endl;
    auto res1 = hnsw->search_exact(point1, 10);
    for (auto r : res1) {
        spdlog::info("id {} dist {}", r.first, r.second);
    }
    return 0;
}