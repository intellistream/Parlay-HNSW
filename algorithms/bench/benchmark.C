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
#include <fstream>
#include <memory>

#include "../HNSW/HNSW.hpp"
#include "../HNSW/dist.hpp"
#include "../utils/euclidian_point.h"
#include <spdlog/spdlog.h>

using namespace parlayANN;

using uint = unsigned int;

template <typename T>
void read_ifvecs(const std::string& filename, 
                 std::vector<point<T>>& all_points, 
                 std::vector<point<T>>& random_points, 
                 uint32_t& dim) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cout << "error !" << std::endl;
    throw std::runtime_error("Failed to open file: " + filename);
  }

  uint32_t id = 0;
  int32_t vector_dim = 0;

  static std::vector<std::vector<T>> all_coords;

  while (file.peek() != EOF) {
    if (vector_dim == 0) {
      file.read(reinterpret_cast<char*>(&vector_dim), sizeof(int32_t));
      if (vector_dim <= 0) {
        throw std::runtime_error("Invalid vector dimension in file.");
      }
      dim = static_cast<uint32_t>(vector_dim);
      all_coords.reserve(1000000); 
    } else {
      int32_t temp_dim;
      file.read(reinterpret_cast<char*>(&temp_dim), sizeof(int32_t));
      if (temp_dim != vector_dim) {
        throw std::runtime_error("Inconsistent vector dimensions in file.");
      }
    }

    std::vector<T> coords(vector_dim);
    file.read(reinterpret_cast<char*>(coords.data()), vector_dim * sizeof(T));

    all_coords.emplace_back(std::move(coords));

    all_points.push_back(point<T>(id++, all_coords.back().data()));
  }

  if (all_points.size() > 500) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::sample(all_points.begin(), all_points.end(), 
                std::back_inserter(random_points), 500, gen);
  } else {
    random_points = all_points;
  }

  spdlog::info("{} loaded, dimension : {}, total points: {}, random points: {}", 
               filename, dim, all_points.size(), random_points.size());
}

template <typename T>
std::vector<point<T>> get_random_points(const std::vector<point<T>>& points, size_t sample_size) {
  if (sample_size > points.size()) {
      throw std::invalid_argument("Sample size exceeds the number of available points.");
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<point<T>> sample;
  std::sample(points.begin(), points.end(), std::back_inserter(sample), sample_size, gen);

  return sample;
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
                "[-data_type <data_type>] [-dist_func <dist_func>]"
                "[-k <k> ] [-file_path <base_path>]");

  char* data_type = P.getOptionValue("-data_type");
  char* dist_func = P.getOptionValue("-dist_func");
  char* file_path = P.getOptionValue("-file_path");
  int k = P.getOptionIntValue("-k", 5);

      std::cout << "Data type: " << data_type << "\n"
              << "Distance function: " << dist_func << "\n"
              << "Base file: " << file_path << std::endl;

  std::string df = std::string(dist_func);
  std::string dt = std::string(data_type);

  if (dt == "float") {
    std::vector<descr_l2<float>::type_point> points;
    std::vector<descr_l2<float>::type_point> qpoints;
    uint32_t dim = 0;
    read_ifvecs(file_path, points, qpoints, dim);

    time_loop(1, 0,
    [&] () {},
    [&] () {
      auto hnsw = new ANN::HNSW<descr_l2<float>>(points.begin(), points.end(), dim);

      float total_recall = 0.0;
      search_control ctrl{};
      for (auto qp : qpoints) {
        search_control ctrl{};
        auto res = hnsw->search(qp, k, 50, ctrl);
        auto exact_res = hnsw->search_exact(qp, k);
       
        std::set<int> exact_ids;
        for (const auto& pair : exact_res) {
          exact_ids.insert(pair.first); 
        }

        size_t hits = 0;
        for (const auto& pair : res) {
          if (exact_ids.find(pair.first) != exact_ids.end()) {
            ++hits;  
          }
        }

        float recall = static_cast<float>(hits) / exact_ids.size();
        total_recall += recall;
      }
      
      spdlog::info("Overall recall {}", total_recall / qpoints.size());
    },
    [&] () {});

  } else if (dt == "uint8") {

  } else if (dt == "int8") {

  }

  return 0;
}

