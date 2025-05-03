# HW3: pairwise alignment with the Striped Smither Waterman algorithm using CUDA

## Repository Link
[2025-Parallel-Program-Optimization/hw3](https://github.com/jasminechennnnn/2025-Parallel-Program-Optimization/tree/main/hw3)

todo:
- [x] install xsimd on server3
GotohAligner是一種序列比對實作，基於Gotoh演算法，這是Smith-Waterman演算法的一個改進版本。
Gotoh演算法的主要特點與差異：


## 1 Usage

## 2 Enviroment


## 3 Note
### basic CUDA 
thread, wrap
block
grid


host device
alloc memcpy

### NVBIO library
- ref: https://developer.nvidia.com/nvbio
- NVBIO is a GPU-accelerated C++ framework for High-Throughput Sequence Analysis for both short and long read alignment.

**aln::batch_alignment_score**
- source:
    - https://nvlabs.github.io/nvbio/group___batch_alignment.html#ga96aa53d0150bba09d393f544c11a865f
    - https://nvlabs.github.io/nvbio/batched__inl_8h_source.html#l00995
- 目的
    - 執行GPU上的批次比對
- 參數
    - **aln::make_gotoh_aligner**
        - https://nvlabs.github.io/nvbio/structnvbio_1_1aln_1_1_gotoh_aligner.html
        - https://nvlabs.github.io/nvbio/alignment__base_8h_source.html#l00284


### Gotoh Aligner
與 smith-waterman差異：
1. Affine Gap Penalties  
傳統Smith-Waterman使用線性間隙懲罰，每個間隙懲罰相同
Gotoh演算法引入了不同的間隙開啟(gap open)和間隙延伸(gap extension)懲罰
這更符合生物學現實，因為蛋白質序列中連續的間隙(insertion/deletion)通常是單一進化事件
2. 三個動態規劃矩陣  
傳統Smith-Waterman使用一個DP矩陣
Gotoh使用三個矩陣: H(主矩陣), E(水平間隙), F(垂直間隙)
這允許算法區分新開的間隙和延伸的間隙
