# HW2: pairwise alignment with the Striped Smither Waterman algorithm using XSIMD


## Repository Link
[2025-Parallel-Program-Optimization/hw2](https://github.com/jasminechennnnn/2025-Parallel-Program-Optimization/tree/main/hw2)

## 1 Usage

There are two ways to run the program:

### Method 1: Run with specific FASTA files

```bash
./ssw_xsimd <target.fasta> <query.fasta>
```

### Method 2: Use he default test files

```bash
make run
```

This automatically executes `./ssw_xsimd target.fa query.fa`

## 2 Example Output
running on M3 chip for example:
```
./ssw_xsimd target.fa query.fa
XSIMD info: 
NEON available
Register size: 128 bits (16 bytes)

--------XSIMD result for target: chr1--------
target_name: chr1
query_name: 6:163296599:F:198;None;None/1
Baseline time: 1.2 ms
SIMD time: 0.1 ms
Speedup: 10.8X
optimal_alignment_score: 37    suboptimal_alignment_score: 28    strand: +    target_begin: 453    target_end: 492    query_begin: 17    query_end: 51

Seq1(Target):      453    CCAATGCCACAAAACATCTGTCTCTAACTGGTG--TGTGTGT        492
                          |||  ||| ||||  |||||| |*||| |||||  |*|||||
Seq2(Query):        17    CCA--GCC-CAAA--ATCTGT-TTTAA-TGGTGGATTTGTGT         51

--------XSIMD result for target: chr1--------
target_name: chr1
query_name: 3:153409880:F:224;None;3,153410143,G,A/1
Baseline time: 1.2 ms
SIMD time: 0.1 ms
Speedup: 12.2X
optimal_alignment_score: 42    suboptimal_alignment_score: 41    strand: +    target_begin: 523    target_end: 577    query_begin: 3    query_end: 53

Seq1(Target):      523    GAGAGAGAAAATTTCACTCCCTCCATAAATCTCACAGTATTCTTTTCTTTTTCCT        577
                          || ||||**|||||*|*||*||*||*|*|**|*|| ||| |||||| ||||*|||
Seq2(Query):         3    GA-AGAGTTAATTTAAGTCACTTCAAACAGATTAC-GTA-TCTTTT-TTTTCCCT         53

--------XSIMD result for target: chr1--------
target_name: chr1
query_name: Y:26750420:R:-132;None;None/1
Baseline time: 1.3 ms
SIMD time: 0.1 ms
Speedup: 17.2X
optimal_alignment_score: 32    suboptimal_alignment_score: 29    strand: +    target_begin: 120    target_end: 163    query_begin: 1    query_end: 44

Seq1(Target):      120    AA-AACATAGGAA-AAAATTA--TTTAATAATAAAATTTA-ATTGGCAA        163
                          || ||||   ||| **|||||  ||*||*|||  |*|||| |||*||||
Seq2(Query):         1    AACAACA---GAAGTTAATTAGCTTCAAAAAT--ACTTTATATTTGCAA         44

```

## 3 References
- Smither Waterman algorithm: https://youtu.be/f1VYKHGKQsw?si=5nNK1AluUadAx6Ws
- Stripped Smither Waterman algorithm (and other cpu acceleration methods): https://zhuanlan.zhihu.com/p/676835444
- SSW Library: https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library
- XSIMD installation: https://formulae.brew.sh/formula/xsimd
- XSIMD source code: https://github.com/xtensor-stack/xsimd?tab=readme-ov-file
- SSE2 instructions: https://blog.csdn.net/tercel_zhang/article/details/80050120