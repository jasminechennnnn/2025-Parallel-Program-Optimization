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
DEBUG C++: Final values:
DEBUG C++: results[0]: score=37, ref=491, read=50
DEBUG C++: results[1]: score=28, ref=426, read=0
DEBUG C++: Final values:
DEBUG C++: results[0]: score=37, ref=452, read=34
DEBUG C++: results[1]: score=16, ref=480, read=0

--------XSIMD result--------
target_name: chr1
query_name: 6:163296599:F:198;None;None/1
Baseline time: 1.1 ms
SIMD time: 0.1 ms
Speedup: 8.0X
optimal_alignment_score: 37    suboptimal_alignment_score: 28    strand: +    target_begin: 453    target_end: 492    query_begin: 17    query_end: 51

Seq1(Target):      453    CCAATGCCACAAAACATCTGTCTCTAACTGGTG--TGTGTGT        492
                          |||  ||| ||||  |||||| |*||| |||||  |*|||||
Seq2(Query):        17    CCA--GCC-CAAA--ATCTGT-TTTAA-TGGTGGATTTGTGT         51
```

## 3 References
- Smither Waterman algorithm: https://youtu.be/f1VYKHGKQsw?si=5nNK1AluUadAx6Ws
- Stripped Smither Waterman algorithm (and other cpu acceleration methods): https://zhuanlan.zhihu.com/p/676835444
- XSIMD installation: https://formulae.brew.sh/formula/xsimd
- XSIMD source code: https://github.com/xtensor-stack/xsimd?tab=readme-ov-file