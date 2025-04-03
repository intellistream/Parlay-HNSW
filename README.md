# Parlay-HNSW

Refer to the ParlayANN (https://github.com/cmuparlay/ParlayANN) implementation to create a HNSW.

## Init third-party
```bash
git submodule init
git submodule update
```

## Prepare data
```bash
mkdir -p data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
```

## Build
```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Run HNSW
```bash
cd algorithm/HNSW
/bench-hnsw -data_type float -dist_func l2 -k 5 -file_path  ../../data/sift/sift_base.fvecs
```
