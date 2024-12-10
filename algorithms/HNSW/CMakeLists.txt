#TODO: not sure how to build this (no makefile?).
add_executable(bench-hnsw ../bench/benchmark.C)
  target_link_libraries(bench-hnsw PRIVATE parlay spdlog)
  target_precompile_headers(bench-hnsw PRIVATE HNSW.hpp)

