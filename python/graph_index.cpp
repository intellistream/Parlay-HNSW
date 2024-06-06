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


#include "../algorithms/vamana/index.h"
#include "../algorithms/utils/types.h"
#include "../algorithms/utils/point_range.h"
#include "../algorithms/utils/graph.h"
#include "../algorithms/utils/euclidian_point.h"
#include "../algorithms/utils/mips_point.h"
#include "../algorithms/utils/stats.h"
#include "../algorithms/utils/beamSearch.h"
#include "../algorithms/HNSW/HNSW.hpp"
#include "pybind11/numpy.h"

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include <cstdio>
#include <utility>
#include <optional>

namespace py = pybind11;
using NeighborsAndDistances = std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template<typename T, typename Point>
struct GraphIndex{
  Graph<unsigned int> G;
  PointRange<T, Point> Points;
  using QuantT = uint8_t;
  using QuantPoint = Euclidian_Point<QuantT>;
  using QuantRange = PointRange<QuantT, QuantPoint>;
  QuantRange Quant_Points;
  bool use_quantization;

  std::optional<ANN::HNSW<Desc_HNSW<T, Point>>> HNSW_index;

  GraphIndex(std::string &data_path, std::string &index_path, size_t num_points, size_t dimensions, bool is_hnsw=false)
    : use_quantization(false) {
    Points = PointRange<T, Point>(data_path.data());
    assert(num_points == Points.size());
    assert(dimensions == Points.dimension());
    
    if (sizeof(T) > 1) {
      use_quantization = true;
      Quant_Points = QuantRange(Points);
    }

    if(is_hnsw) {
      HNSW_index = ANN::HNSW<Desc_HNSW<T, Point>>(
                                                  index_path,
                                                  [&](unsigned int i/*indexType*/){
                                                    return Points[i];
                                                  }
                                                  );
    }
    else {
      G = Graph<unsigned int>(index_path.data());
    }
  }

  auto search_dispatch(const Point &q, QueryParams &QP, bool quant)
  {
    // if(HNSW_index) {
    //   using indexType = unsigned int; // be consistent with the type of G
    //   using std::pair;
    //   using seq_t = parlay::sequence<pair<indexType, typename Point::distanceType>>;

    //   indexType dist_cmps = 0;
    //   search_control ctrl{};
    //   if(QP.limit>0) {
    //     ctrl.limit_eval = QP.limit;
    //   }
    //   ctrl.count_cmps = &dist_cmps;

    //   seq_t frontier = HNSW_index->search(q, QP.k, QP.beamSize, ctrl);
    //   return pair(pair(std::move(frontier), seq_t{}), dist_cmps);
    // }
    //    else {
    using indexType = unsigned int;
    parlay::sequence<indexType> starts(1, 0);
    stats<indexType> Qstats(1);
    if (quant && use_quantization) {
      typename QuantPoint::T buffer[128];
      if ( Quant_Points.params.slope == 1) {
        for (int i=0; i < Quant_Points.params.dims; i++)
          buffer[i] = q[i];
        QuantPoint quant_q(buffer, 0, Quant_Points.params);
        return beam_search(quant_q, G, Quant_Points, starts, QP).first.first;
      } else {
        QuantPoint::translate_point(buffer, q, Quant_Points.params);
        QuantPoint quant_q(buffer, 0, Quant_Points.params);
        return beam_search_rerank(q, quant_q, G,
                                  Points, Quant_Points,
                                  Qstats, starts, QP);
      }
    } else {
      return beam_search(q, G, Points, starts, QP).first.first;
    }
  }

  NeighborsAndDistances batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
                                     uint64_t num_queries, uint64_t knn,
                                     uint64_t beam_width, bool quant = false, int64_t visit_limit = -1){
    if(visit_limit == -1) visit_limit = HNSW_index? 0: G.size();
    QueryParams QP(knn, beam_width, 1.35, visit_limit, HNSW_index?0:G.max_degree());

    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&] (size_t i){
      std::vector<T> v(Points.dimension());
      for (int j=0; j < v.size(); j++)
        v[j] = queries.data(i)[j];
      Point q = Point(v.data(), 0, Points.params);
      auto frontier = search_dispatch(q, QP, quant);
      for(int j=0; j<knn; j++){
        ids.mutable_data(i)[j] = frontier[j].first;
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });
    return std::make_pair(std::move(ids), std::move(dists));
  }

  NeighborsAndDistances single_search(py::array_t<T> &q, uint64_t knn,
                                      uint64_t beam_width, bool quant, int64_t visit_limit) {
    if(visit_limit == -1) visit_limit = HNSW_index? 0: G.size();
    QueryParams QP(knn, beam_width, 1.35, visit_limit, HNSW_index?0:G.max_degree());
    int dims = Points.dimension();

    py::array_t<unsigned int> ids({knn});
    py::array_t<float> dists({knn});
    //auto p = q.unchecked<3>();
    T v[dims];
    for (int j=0; j < dims; j++)
      v[j] = q.data()[j];
    Point p = Point(v, 0, Points.params);
    auto frontier = search_dispatch(p, QP, quant);
    for(int j=0; j<knn; j++) {
      ids.mutable_data()[j] = frontier[j].first;
      dists.mutable_data()[j] = frontier[j].second;
    }
    return std::make_pair(std::move(ids), std::move(dists));
  }

  NeighborsAndDistances batch_search_from_string(std::string &queries, uint64_t num_queries, uint64_t knn,
                                                 uint64_t beam_width, bool quant = false){
    QueryParams QP(knn, beam_width, 1.35, HNSW_index?0:G.size(), HNSW_index?0:G.max_degree());
    PointRange<T, Point> QueryPoints = PointRange<T, Point>(queries.data());
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});
    parlay::parallel_for(0, num_queries, [&] (size_t i){
      auto frontier = search_dispatch(QueryPoints[i], QP, quant);
      for(int j=0; j<knn; j++){
        ids.mutable_data(i)[j] = frontier[j].first;
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });
    return std::make_pair(std::move(ids), std::move(dists));
  }

  void check_recall(std::string &gFile, py::array_t<unsigned int, py::array::c_style | py::array::forcecast> &neighbors, int k){
    groundTruth<unsigned int> GT = groundTruth<unsigned int>(gFile.data());

    size_t n = GT.size();
    
    int numCorrect = 0;
    for (unsigned int i = 0; i < n; i++) {
      parlay::sequence<int> results_with_ties;
      for (unsigned int l = 0; l < k; l++)
        results_with_ties.push_back(GT.coordinates(i,l));
      // float last_dist = GT.distances(i, k-1);
      // for (unsigned int l = k; l < GT.dimension(); l++) {
      //   if (i == 4) std::cout << l << std::endl;
      //   if (GT.distances(i,l) == last_dist) {
      //     std::cout << "unlikely" << std::endl;
      //     results_with_ties.push_back(GT.coordinates(i,l));
      //   }
      // }
      std::set<int> reported_nbhs;
      for (unsigned int l = 0; l < k; l++) reported_nbhs.insert(neighbors.mutable_data(i)[l]);
      for (unsigned int l = 0; l < results_with_ties.size(); l++) {
        if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    float recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
    std::cout << "Recall: " << recall << std::endl;
  }

  // void check_recall(std::string &gFile, py::array_t<unsigned int, py::array::c_style | py::array::forcecast> &neighbors, int k){
  //   groundTruth<unsigned int> GT = groundTruth<unsigned int>(gFile.data());

  //   size_t n = GT.size();

  //   int numCorrect = 0;
  //   for (unsigned int i = 0; i < n; i++) {
  //     parlay::sequence<int> results_with_ties;
  //     for (unsigned int l = 0; l < k; l++)
  //       results_with_ties.push_back(GT.coordinates(i,l));
  //     std::cout << i << std::endl;
  //     float last_dist = GT.distances(i, k-1);
  //     for (unsigned int l = k; l < GT.dimension(); l++) {
  //       if (GT.distances(i,l) == last_dist) {
  //         results_with_ties.push_back(GT.coordinates(i,l));
  //       }
  //     }
  //     std::cout << "aa" << std::endl;
  //     std::set<int> reported_nbhs;
  //     for (unsigned int l = 0; l < k; l++) reported_nbhs.insert(neighbors.mutable_data(i)[l]);
  //     for (unsigned int l = 0; l < results_with_ties.size(); l++) {
  //       if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()) {
  //         numCorrect += 1;
  //       }
  //     }
  //   }
  //   float recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
  //   std::cout << "Recall: " << recall << std::endl;
  // }

};
