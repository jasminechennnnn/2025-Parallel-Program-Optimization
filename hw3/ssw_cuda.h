/**
 * @file ssw_cuda.h
 * @brief CUDA-accelerated Smith-Waterman algorithm header
 * 
 * This header defines the structures and function declarations for the
 * CUDA-accelerated Smith-Waterman algorithm.
 */

#ifndef SSW_CUDA_H
#define SSW_CUDA_H

#include <vector>
#include <memory>
#include <cstdint>
#include <utility>
#include <string>

// Alignment result structure
struct alignment_end {
    uint16_t score;
    int32_t ref;   // reference ending position
    int32_t read;  // read ending position
};

// Full alignment result structure
struct s_align {
    uint16_t score1;     // best alignment score
    uint16_t score2;     // sub-optimal alignment score
    int32_t ref_begin1;  // 0-based best alignment beginning position on reference
    int32_t ref_end1;    // 0-based best alignment ending position on reference
    int32_t read_begin1; // 0-based best alignment beginning position on read
    int32_t read_end1;   // 0-based best alignment ending position on read
    int32_t ref_end2;    // 0-based sub-optimal alignment ending position on reference
    int8_t flag;         // alignment status (1: the best alignment is a reverse complement alignment; 2: suboptimal score is a valid score; 0: otherwise)
    std::vector<std::pair<char, int>> cigar; // cigar string in std::pair<operation, length> format
};

// From ssw_cuda.cu
template <typename T>
std::vector<alignment_end> ssw_cuda_template(
    const std::vector<int8_t>& ref_num,
    int8_t ref_dir,
    const std::vector<int8_t>& read_num,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const std::vector<T>& profile,
    T terminate,
    T bias,
    int32_t maskLen);

// Byte precision implementation
std::vector<alignment_end> ssw_byte_cuda(
    const std::vector<int8_t>& ref_num,
    int8_t ref_dir,
    const std::vector<int8_t>& read_num,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const std::vector<int8_t>& profile,
    uint8_t terminate,
    uint8_t bias,
    int32_t maskLen = 0);

// Word precision implementation
std::vector<alignment_end> ssw_word_cuda(
    const std::vector<int8_t>& ref_num,
    int8_t ref_dir,
    const std::vector<int8_t>& read_num,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const std::vector<int16_t>& profile,
    uint16_t terminate = 0,
    int32_t maskLen = 0);

// Main function for SW alignment
std::unique_ptr<s_align> ssw_main_cuda(
    const std::vector<int8_t>& profile_byte,
    const std::vector<int16_t>& profile_word,
    const std::vector<int8_t>& ref_num,
    const std::vector<int8_t>& read_num,
    const std::vector<int8_t>& scoring_matrix,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const int32_t maskLen,
    const uint8_t bias);

// From ssw_common.cpp
std::vector<int8_t> seq_reverse(
    const std::vector<int8_t>& seq_num, 
    int32_t seq_end);

std::vector<std::pair<char, int>> banded_sw(
    const std::vector<int8_t>& ref_num,
    const std::vector<int8_t>& read_num,
    int32_t score,
    const uint32_t weight_gapO,
    const uint32_t weight_gapE,
    int32_t band_width,
    const std::vector<int8_t>& scoring_matrix);

// template <typename T>
// std::vector<T> qP_cuda(
//     const std::vector<int8_t>& read_num, 
//     const std::vector<int8_t>& scoring_matrix, 
//     const std::string& precision);

// // Explicit template declarations to make available
// extern template std::vector<int8_t> qP_cuda<int8_t>(
//     const std::vector<int8_t>&, const std::vector<int8_t>&, const std::string&);
// extern template std::vector<int16_t> qP_cuda<int16_t>(
//     const std::vector<int8_t>&, const std::vector<int8_t>&, const std::string&);

#endif // SSW_CUDA_H