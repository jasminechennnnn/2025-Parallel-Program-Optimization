/**
 * @file ssw_common.h
 * @brief Common functions for Smith-Waterman algorithm
 * 
 * This header defines common utility functions used by both SIMD and CUDA
 * implementations of the Smith-Waterman algorithm.
 */

#ifndef SSW_COMMON_H
#define SSW_COMMON_H

#include <vector>
#include <cstdint>
#include <utility>
#include <string>
#include <vector>
#include <cmath>       // sqrt()
#include <iostream>    // std::cerr
#include <stdexcept>   // std::runtime_error
#include <algorithm>   // min_element()
#include <limits>      // numeric_limits

// Helper macro for rounding up to next power of 2
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

// Helper macros for banded Smith-Waterman
#define set_u(u, w, i, j) { int x=(i)-(w); x=x>0?x:0; (u)=(j)-x+1; }
#define set_d(u, w, i, j, p) { int x=(i)-(w); x=x>0?x:0; x=(j)-x; (u)=((x)*(p)); }

// struct decaration from ssw_xsimd.h
struct ScoringMatrix;
struct AlignmentResult;

// Function declarations
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

/**
 * @brief CUDA-accelerated Striped Smith-Waterman algorithm
 * 
 * @param ref Reference sequence as string
 * @param read Query sequence as string
 * @param scoring Scoring parameters for the alignment
 * @return Alignment result containing scores, positions and CIGAR string
 */
AlignmentResult smith_waterman_striped_cuda(
    const std::string& ref, 
    const std::string& read, 
    const ScoringMatrix& scoring);


// template <typename T>
// std::vector<T> qP_cuda(
//     const std::vector<int8_t>& read_num, 
//     const std::vector<int8_t>& scoring_matrix, 
//     const std::string& precision);

// // Explicit template declarations
// extern template std::vector<int8_t> qP_cuda<int8_t>(
//     const std::vector<int8_t>&, const std::vector<int8_t>&, const std::string&);
// extern template std::vector<int16_t> qP_cuda<int16_t>(
//     const std::vector<int8_t>&, const std::vector<int8_t>&, const std::string&);
/**
 * Profile generation for CUDA acceleration
 * 
 * @param read_num Numeric representation of the read sequence
 * @param scoring_matrix Flattened scoring matrix (nxn)
 * @param precision Computation precision ("byte" for int8_t or "word" for int16_t)
 * @return Flattened profile table suitable for CUDA processing
 */
template <typename T>
std::vector<T> qP_cuda(const std::vector<int8_t>& read_num, 
                       const std::vector<int8_t>& scoring_matrix,
                       const std::string& precision) {
    // Validate input parameters
    const auto readLen = static_cast<int32_t>(read_num.size());
    const auto n = static_cast<int32_t>(sqrt(scoring_matrix.size()));
    
    if (n * n != static_cast<int32_t>(scoring_matrix.size())) {
        std::cerr << "ERROR: Scoring matrix size is not a perfect square" << std::endl;
        throw std::runtime_error("Scoring matrix size is not a perfect square");
    }
    
    // Calculate bias (only needed for int8_t, bias = abs(min neg. value of matrix))
    T bias = 0;
    if constexpr (std::is_same_v<T, int8_t>) {
        auto min_element = *std::min_element(scoring_matrix.begin(), scoring_matrix.end());
        bias = min_element < 0 ? static_cast<T>(abs(min_element)) : 0;
    }
    
    // For CUDA, we'll create a flattened array in a row-major format that's easy to process
    // Format: [nt][pos] where nt is the nucleotide type and pos is the position in read
    try {
        // Allocate flattened profile array (n rows x readLen columns)
        std::vector<T> profile(n * readLen);
        
        // Generate profile in a format suitable for CUDA
        for (int32_t nt = 0; nt < n; ++nt) {
            for (int32_t pos = 0; pos < readLen; ++pos) {
                size_t profile_idx = nt * readLen + pos;
                
                // Get the correct score from the scoring matrix
                int matrix_idx = nt * n + read_num[pos];
                
                // Apply bias for byte precision
                if constexpr (std::is_same_v<T, int8_t>) {
                    profile[profile_idx] = scoring_matrix[matrix_idx] + bias;
                } else {
                    profile[profile_idx] = scoring_matrix[matrix_idx];
                }
            }
        }
        
        return profile;
        
    } catch (const std::length_error& e) {
        std::cerr << "ERROR: Vector allocation failed with length_error: " << e.what() << std::endl;
        std::cerr << "  Attempted to allocate profile of size: " << n * readLen << std::endl;
        std::cerr << "  Consider using smaller sequences or more memory" << std::endl;
        throw; // Re-throw to be handled by the caller
    } catch (const std::bad_alloc& e) {
        std::cerr << "ERROR: Memory allocation failed with bad_alloc: " << e.what() << std::endl;
        throw; // Re-throw to be handled by the caller
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception during profile generation: " << e.what() << std::endl;
        throw; // Re-throw to be handled by the caller
    }
}

#endif // SSW_COMMON_H