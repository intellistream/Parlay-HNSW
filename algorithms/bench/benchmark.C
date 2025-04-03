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
#include <cstdlib>
#include <thread>
#include <chrono>

#include "../HNSW/HNSW.hpp"
#include "../HNSW/dist.hpp"
#include "../utils/euclidian_point.h"
#include <spdlog/spdlog.h>

using namespace parlayANN;

using uint = unsigned int;

template <typename T>
void bench(commandLine& P) {
  char* dist_func = P.getOptionValue("-dist_func");
  char* data_path = P.getOptionValue("-data_path");
  char* query_path = P.getOptionValue("-query_path");
  int k = P.getOptionIntValue("-k", 10);
  int n = P.getOptionIntValue("-n", 1000000);
  int t = P.getOptionIntValue("-t", std::thread::hardware_concurrency());
  int query_n = P.getOptionIntValue("-query_n", 100);

  setenv("PARLAY_NUM_THREADS", std::to_string(t).c_str(), 1);

  std::cout << "Data type: " << data_type << "\n"
          << "Threads: " << t << "\n"
          << "Data file: " << data_path << "\n"
          << "Query file: " << query_path << std::endl;
  
  PointRange<Euclidian_Point<int8_t>> Points(data_path);
  PointRange<Euclidian_Point<int8_t>> Query_Points(query_path);

  auto start = std::chrono::high_resolution_clock::now();
  auto hnsw = new ANN::HNSW<descr_l2<float>>(points.begin(), points.end(), dim);

  float total_recall = 0.0;
  search_control ctrl{};
  for (auto qp : qpoints) {
    auto res = hnsw->search(qp, k, 50, ctrl);
  }
  
  spdlog::info("Overall recall {}", total_recall / qpoints.size());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start; 

  double qps = static_cast<double>(points.size()) / duration.count();
  std::cout << "Total queries: " << points.size() << "\n"
            << "Total time: " << duration.count() << " seconds\n"
            << "QPS: " << qps << " queries per second" << std::endl;
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
                "[-data_type <data_type>] [-k <k> ] [-n <n>] [-t <t>]" 
                "[-data_path <data_path>] [-query_path <query_path>]");

  char* data_type = P.getOptionValue("-data_type");
  if (strcmp(data_type, "float") == 0) {
    bench<float>(P);
  } else if (strcmp(data_type, "uint8_t") == 0) {
    bench<uint8_t>(P);
  } else if (strcmp(data_type, "int8_t") == 0) {
    bench<int8_t>(P);
  }

  return 0;
}

