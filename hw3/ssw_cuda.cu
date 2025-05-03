/**
 * @file ssw_cuda.cu
 * @brief CUDA-accelerated Smith-Waterman algorithm
 * 
 * This implementation converts the SIMD-based Striped Smith-Waterman algorithm
 * to use CUDA parallel processing on GPUs.
 */

#include "ssw_common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstdint>

// Helper for error checking CUDA calls
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
}

// Alignment result structure
struct alignment_end {
    uint16_t score;
    int32_t ref;   // reference ending position
    int32_t read;  // read ending position
};

// Device function for saturated subtraction
// template <typename T>
// __device__ inline T device_saturated_sub(const T a, const T b) {
//     T result = a - b;
//     return (result > a) ? T(0) : result; // Check for underflow
// }
template<typename T, typename U>
__device__ T device_saturated_sub(T a, const U b) {
    return (a > static_cast<T>(b)) ? a - static_cast<T>(b) : 0;
}

/**
 * @brief CUDA kernel for Smith-Waterman alignment
 * 
 * Each thread processes one reference position in a grid-like pattern.
 * Multiple blocks handle different sections of the reference sequence.
 * 
 * @tparam T Value type (int8_t or int16_t)
 * @param ref_num Reference sequence
 * @param ref_dir Reference direction (0: forward, 1: reverse)
 * @param refLen Length of reference sequence
 * @param readLen Length of read/query sequence
 * @param weight_gapO Gap open penalty
 * @param weight_gapE Gap extension penalty
 * @param profile Query profile (flattened)
 * @param terminate Termination score
 * @param bias Bias value for int8_t version
 * @param maxColumn Output array for maximum scores at each ref position
 * @param end_read_column Output array for read positions at each max score
 */
template <typename T>
__global__ void ssw_kernel(const int8_t* ref_num,
                          int8_t ref_dir,
                          int32_t refLen,
                          int32_t readLen,
                          const uint8_t weight_gapO,
                          const uint8_t weight_gapE,
                          const T* profile,
                          T terminate,
                          T bias,
                          T* maxColumn,
                          int32_t* end_read_column) {
    
    // Calculate reference position for this thread
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (i >= refLen) return;
    
    // If we're in reverse mode, adjust the position
    int32_t pos;
    if (ref_dir == 1) {
        pos = refLen - 1 - i;
    } else {
        pos = i;
    }
    
    // Initialize H, E, and F vectors for this reference position
    T* H = new T[readLen]();
    T* E = new T[readLen]();
    T* F = new T[readLen]();
    
    // Get profile offset for current reference base
    int32_t nt = ref_num[pos];
    const T* vP = profile + (nt * readLen);
    
    // Local max tracking
    T max_score = 0;
    int32_t max_pos = -1;
    
    // Process each read position
    for (int32_t j = 0; j < readLen; j++) {
        // Calculate H[j]
        T H_prev = (j > 0) ? H[j-1] : 0;
        T H_diag = (j > 0) ? H_prev : 0;
        
        // Add profile score
        H[j] = H_diag + vP[j];
        
        // Subtract bias if using int8_t
        if (std::is_same<T, int8_t>::value) {
            H[j] = H[j] - bias;
        }
        
        // Calculate E (gap in reference)
        E[j] = max(device_saturated_sub(E[j], weight_gapE), 
                  device_saturated_sub(H[j], weight_gapO));
        
        // Calculate F (gap in query)
        F[j] = max(device_saturated_sub(F[j], weight_gapE), 
                  device_saturated_sub(H[j], weight_gapO));
        
        // Get max of H, E, F
        H[j] = max(H[j], max(E[j], F[j]));
        
        // Update local max
        if (H[j] > max_score) {
            max_score = H[j];
            max_pos = j;
        }
        
        // If at end of read, potentially update globals
        if (j == readLen - 1) {
            maxColumn[pos] = max_score;
            end_read_column[pos] = max_pos;
        }
        
        // Early termination if reached target score
        if (max_score >= terminate && terminate > 0) {
            break;
        }
    }
    
    // Free temp memory
    delete[] H;
    delete[] E;
    delete[] F;
}

/**
 * @brief Optimized version of SW kernel using shared memory
 * 
 * This version uses shared memory for better performance, especially for 
 * shorter read sequences that can fit in shared memory.
 * 
 * @tparam T Value type (int8_t or int16_t)
 */
template <typename T>
__global__ void ssw_shared_kernel(const int8_t* ref_num,
                                int8_t ref_dir,
                                int32_t refLen,
                                int32_t readLen,
                                const uint8_t weight_gapO,
                                const uint8_t weight_gapE,
                                const T* profile,
                                T terminate,
                                T bias,
                                T* maxColumn,
                                int32_t* end_read_column) {
    
    // Shared memory for temp arrays (better for shorter reads)
    extern __shared__ char shared_mem[];
    T* shared_H = (T*)shared_mem;
    T* shared_E = shared_H + readLen;
    
    // Calculate reference position for this thread
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (i >= refLen) return;
    
    // If we're in reverse mode, adjust the position
    int32_t pos;
    if (ref_dir == 1) {
        pos = refLen - 1 - i;
    } else {
        pos = i;
    }
    
    // Initialize shared memory
    for (int32_t j = 0; j < readLen; j++) {
        shared_H[j] = 0;
        shared_E[j] = 0;
    }
    
    // Get profile offset for current reference base
    int32_t nt = ref_num[pos];
    const T* vP = profile + (nt * readLen);
    
    // Local max tracking
    T max_score = 0;
    int32_t max_pos = -1;
    T F = 0;
    
    // Process each read position
    for (int32_t j = 0; j < readLen; j++) {
        // Calculate H[j]
        T H_prev = (j > 0) ? shared_H[j-1] : 0;
        T H_diag = (j > 0) ? H_prev : 0;
        
        // Add profile score
        shared_H[j] = H_diag + vP[j];
        
        // Subtract bias if using int8_t
        if (std::is_same<T, int8_t>::value) {
            shared_H[j] = shared_H[j] - bias;
        }
        
        // Calculate E (gap in reference)
        shared_E[j] = max(device_saturated_sub(shared_E[j], weight_gapE), 
                        device_saturated_sub(shared_H[j], weight_gapO));
        
        // Calculate F (gap in query)
        F = max(device_saturated_sub(F, weight_gapE), 
              device_saturated_sub(shared_H[j], weight_gapO));
        
        // Get max of H, E, F
        shared_H[j] = max(shared_H[j], max(shared_E[j], F));
        
        // Update local max
        if (shared_H[j] > max_score) {
            max_score = shared_H[j];
            max_pos = j;
        }
    }
    
    // Update global memory
    maxColumn[pos] = max_score;
    end_read_column[pos] = max_pos;
}

/**
 * @brief Find the maximum score and its position
 * 
 * @tparam T Value type (int8_t or int16_t)
 * @param maxColumn Array of maximum scores
 * @param refLen Length of reference sequence
 * @param bias Bias value for int8_t version
 * @return pair<uint16_t, int32_t> Maximum score and its position
 */
template <typename T>
std::pair<uint16_t, int32_t> find_max_score(const std::vector<T>& maxColumn, int32_t refLen, T bias) {
    T max = 0;
    int32_t max_pos = -1;
    
    for (int32_t i = 0; i < refLen; i++) {
        if (maxColumn[i] > max) {
            max = maxColumn[i];
            max_pos = i;
        }
    }
    
    // Handle overflow for int8_t
    uint16_t final_score;
    if (std::is_same<T, int8_t>::value) {
        auto max_with_bias = static_cast<uint16_t>(max) + static_cast<uint16_t>(bias);
        final_score = max_with_bias >= 255 ? 255 : max;
    } else {
        final_score = max;
    }
    
    return std::make_pair(final_score, max_pos);
}

/**
 * @brief Find the second best alignment
 * 
 * @tparam T Value type (int8_t or int16_t)
 * @param maxColumn Array of maximum scores
 * @param end_ref Best reference ending position
 * @param refLen Reference length
 * @param maskLen Mask length for finding second best
 * @return pair<uint16_t, int32_t> Second best score and its position
 */
template <typename T>
std::pair<uint16_t, int32_t> find_second_best(const std::vector<T>& maxColumn, 
                                            int32_t end_ref, 
                                            int32_t refLen, 
                                            int32_t maskLen) {
    T second_max = 0;
    int32_t second_max_pos = -1;
    
    // Search before the best alignment
    int32_t edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
    for (int32_t i = 0; i < edge; i++) {
        if (maxColumn[i] > second_max) {
            second_max = maxColumn[i];
            second_max_pos = i;
        }
    }
    
    // Search after the best alignment
    edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
    for (int32_t i = edge + 1; i < refLen; i++) {
        if (maxColumn[i] > second_max) {
            second_max = maxColumn[i];
            second_max_pos = i;
        }
    }
    
    return std::make_pair(second_max, second_max_pos);
}

/**
 * @brief Main CUDA Smith-Waterman implementation
 * 
 * @tparam T Value type (int8_t or int16_t)
 * @param ref_num Reference sequence
 * @param ref_dir Reference direction
 * @param read_num Query sequence
 * @param weight_gapO Gap open penalty
 * @param weight_gapE Gap extension penalty
 * @param profile Query profile
 * @param terminate Termination score
 * @param bias Bias value for int8_t
 * @param maskLen Mask length for second best
 * @return vector<alignment_end> Best and second best alignments
 */
template <typename T>
std::vector<alignment_end> ssw_cuda_template(const std::vector<int8_t>& ref_num,
                                           int8_t ref_dir,
                                           const std::vector<int8_t>& read_num,
                                           const uint8_t weight_gapO,
                                           const uint8_t weight_gapE,
                                           const std::vector<T>& profile,
                                           T terminate,
                                           T bias,
                                           int32_t maskLen) {
    // Get sizes
    const auto refLen = static_cast<int32_t>(ref_num.size());
    const auto readLen = static_cast<int32_t>(read_num.size());
    
    // Use provided maskLen or default to readLen/2
    if (maskLen <= 0) {
        maskLen = readLen / 2;
    }
    
    // Allocate device memory
    int8_t *d_ref_num;
    T *d_profile, *d_maxColumn;
    int32_t *d_end_read_column;
    
    // Allocate and copy reference sequence to device
    CHECK_CUDA_ERROR(cudaMalloc(&d_ref_num, refLen * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ref_num, ref_num.data(), refLen * sizeof(int8_t), cudaMemcpyHostToDevice));
    
    // Allocate and copy profile to device 
    // Profile format: [nt_type][read_position]
    size_t profile_size = 5 * readLen * sizeof(T);  // 5 nucleotide types
    CHECK_CUDA_ERROR(cudaMalloc(&d_profile, profile_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_profile, profile.data(), profile_size, cudaMemcpyHostToDevice));
    
    // Allocate output arrays on device
    CHECK_CUDA_ERROR(cudaMalloc(&d_maxColumn, refLen * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_end_read_column, refLen * sizeof(int32_t)));
    
    // Initialize output arrays to zero
    CHECK_CUDA_ERROR(cudaMemset(d_maxColumn, 0, refLen * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemset(d_end_read_column, 0, refLen * sizeof(int32_t)));
    
    // Configure kernel parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (refLen + threadsPerBlock - 1) / threadsPerBlock;
    
    // Determine which kernel to use based on readLen
    if (readLen <= 1024) {  // Use shared memory version for shorter reads
        size_t sharedMemSize = 2 * readLen * sizeof(T);
        ssw_shared_kernel<T><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_ref_num, ref_dir, refLen, readLen, 
            weight_gapO, weight_gapE, d_profile, terminate, bias,
            d_maxColumn, d_end_read_column
        );
    } else {  // Use global memory version for longer reads
        ssw_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(
            d_ref_num, ref_dir, refLen, readLen, 
            weight_gapO, weight_gapE, d_profile, terminate, bias,
            d_maxColumn, d_end_read_column
        );
    }
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    std::vector<T> maxColumn(refLen);
    std::vector<int32_t> end_read_column(refLen);
    
    CHECK_CUDA_ERROR(cudaMemcpy(maxColumn.data(), d_maxColumn, refLen * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(end_read_column.data(), d_end_read_column, refLen * sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    // Find best alignment
    auto best = find_max_score(maxColumn, refLen, bias);
    uint16_t max_score = best.first;
    int32_t end_ref = best.second;
    int32_t end_read = -1;
    
    // Get read position if we found a valid alignment
    if (end_ref >= 0 && end_ref < refLen) {
        end_read = end_read_column[end_ref];
    }
    
    // Find second best alignment
    auto second_best = find_second_best(maxColumn, end_ref, refLen, maskLen);
    uint16_t second_score = second_best.first;
    int32_t second_ref = second_best.second;
    
    // Free device memory
    cudaFree(d_ref_num);
    cudaFree(d_profile);
    cudaFree(d_maxColumn);
    cudaFree(d_end_read_column);
    
    // Prepare return value
    std::vector<alignment_end> results(2);
    
    // Set best alignment
    results[0].score = max_score;
    results[0].ref = end_ref;
    results[0].read = end_read;
    
    // Set second best alignment
    results[1].score = second_score;
    results[1].ref = second_ref;
    results[1].read = 0;  // We don't track read position for second best alignment
    
    return results;
}

/**
 * @brief Smith-Waterman for byte precision (int8_t)
 */
std::vector<alignment_end> ssw_byte_cuda(const std::vector<int8_t>& ref_num,
                                        int8_t ref_dir,
                                        const std::vector<int8_t>& read_num,
                                        const uint8_t weight_gapO,
                                        const uint8_t weight_gapE,
                                        const std::vector<int8_t>& profile,
                                        uint8_t terminate,
                                        uint8_t bias,
                                        int32_t maskLen = 0) {
    return ssw_cuda_template<int8_t>(ref_num, ref_dir, read_num, weight_gapO, weight_gapE, profile, terminate, bias, maskLen);
}

/**
 * @brief Smith-Waterman for word precision (int16_t)
 */
std::vector<alignment_end> ssw_word_cuda(const std::vector<int8_t>& ref_num,
                                        int8_t ref_dir,
                                        const std::vector<int8_t>& read_num,
                                        const uint8_t weight_gapO,
                                        const uint8_t weight_gapE,
                                        const std::vector<int16_t>& profile,
                                        uint16_t terminate = 0,
                                        int32_t maskLen = 0) {
    // For int16_t, bias is not used, pass 0
    return ssw_cuda_template<int16_t>(ref_num, ref_dir, read_num, weight_gapO, weight_gapE, profile, terminate, 0, maskLen);
}

/**
 * @brief Structure to hold alignment results
 */
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

/**
 * Main function for the Smith-Waterman alignment algorithm with CUDA acceleration
 * 
 * @param profile_byte Byte-sized profile for CUDA
 * @param profile_word Word-sized profile for CUDA
 * @param ref_num Reference sequence
 * @param read_num Read sequence
 * @param scoring_matrix Flattened scoring matrix
 * @param weight_gapO Gap opening penalty
 * @param weight_gapE Gap extension penalty 
 * @param maskLen Length of the mask for the optimal alignment
 * @param bias Bias for byte calculations
 * @return Unique pointer to alignment structure containing results
 */
std::unique_ptr<s_align> ssw_main_cuda(
    const std::vector<int8_t>& profile_byte,
    const std::vector<int16_t>& profile_word,
    const std::vector<int8_t>& ref_num,
    const std::vector<int8_t>& read_num,
    const std::vector<int8_t>& scoring_matrix,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const int32_t maskLen,
    const uint8_t bias) {

    // Initialize result structure
    auto r = std::make_unique<s_align>();
    r->ref_begin1 = -1;
    r->read_begin1 = -1;
    r->flag = 0;

    // Check for valid maskLen
    if (maskLen < 15) {
        std::cerr << "When maskLen < 15, the function doesn't return 2nd best alignment information." << std::endl;
    }

    bool word = false;
    int32_t band_width = 0;
    std::vector<alignment_end> bests;
    std::vector<alignment_end> bests_reverse;

    // Find the alignment scores and ending positions
    if (!profile_byte.empty()) {
        try {
            // Call CUDA version instead of SIMD version
            bests = ssw_byte_cuda(ref_num, 0, read_num, weight_gapO, weight_gapE, 
                                profile_byte, -1, bias, maskLen);
            
            // Print some information about the results if available
            if (bests.empty()) {
                std::cerr << "DEBUG C++:   No alignments returned" << std::endl;
            }
        } catch (const std::length_error& e) {
            std::cerr << "ERROR: Vector length error in ssw_byte_cuda: " << e.what() << std::endl;
            throw; // Re-throw to be handled by the caller
        } catch (const std::bad_alloc& e) {
            std::cerr << "ERROR: Memory allocation failed in ssw_byte_cuda: " << e.what() << std::endl;
            throw; // Re-throw to be handled by the caller
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in ssw_byte_cuda: " << e.what() << std::endl;
            throw; // Re-throw to be handled by the caller
        }
        
        if (!profile_word.empty() && bests[0].score == 255) {
            // Switch to word precision if byte precision reached max score
            bests = ssw_word_cuda(ref_num, 0, read_num, weight_gapO, weight_gapE, 
                                profile_word, -1, maskLen);
            word = true;
        } else if (bests[0].score == 255) {
            std::cerr << "Please set 2 to the score_size parameter of the initialization, "
                      << "otherwise the alignment results will be incorrect." << std::endl;
            return nullptr;
        }
    } else if (!profile_word.empty()) {
        bests = ssw_word_cuda(ref_num, 0, read_num, weight_gapO, weight_gapE, 
                            profile_word, -1, maskLen);
        word = true;
    } else {
        std::cerr << "Please initialize profiles before calling ssw_main_cuda." << std::endl;
        return nullptr;
    }

    // Check if meaningful alignment was found
    if (bests[0].score <= 0) {
        return r;
    }

    // Set results from the best alignment
    r->score1 = bests[0].score;
    r->ref_end1 = bests[0].ref;   // 0-based, always count from the input seq begin
    r->read_end1 = bests[0].read; // 0-based, count from the alignment begin
    
    std::vector<int8_t> ref_num_subset;
    // Make sure r->ref_end1 is valid and not negative
    if (r->ref_end1 >= 0 && r->ref_end1 < static_cast<int32_t>(ref_num.size())) {
        // Create a subset containing elements from 0 to ref_end1 (inclusive)
        ref_num_subset.assign(ref_num.begin(), ref_num.begin() + r->ref_end1 + 1);
    } else {
        // Fallback to using the whole reference if ref_end1 is invalid
        ref_num_subset = ref_num;
        std::cerr << "WARNING: Invalid r->ref_end1, using full reference" << std::endl;
    }

    // Store second-best score if available
    if (maskLen >= 15) {
        r->score2 = bests[1].score;
        r->ref_end2 = bests[1].ref;
    } else {
        r->score2 = 0;
        r->ref_end2 = -1;
    }

    // Get seq_reverse function from ssw_common.cpp
    auto read_reverse = seq_reverse(read_num, r->read_end1);

    if (!word) {
        // Create byte-sized profile for the reverse read
        std::vector<int8_t> reverse_profile = qP_cuda<int8_t>(read_reverse, scoring_matrix, "byte");
        
        try {
            bests_reverse = ssw_byte_cuda(ref_num_subset, 1, read_reverse, weight_gapO, weight_gapE, 
                                        reverse_profile, r->score1, bias, maskLen);
            if (bests_reverse.empty()) {
                std::cerr << "DEBUG:   No reverse alignments returned" << std::endl;
            }
        } catch (const std::length_error& e) {
            std::cerr << "ERROR: Vector length error in ssw_byte_cuda (reverse): " << e.what() << std::endl;
            throw;
        } catch (const std::bad_alloc& e) {
            std::cerr << "ERROR: Memory allocation failed in ssw_byte_cuda (reverse): " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in ssw_byte_cuda (reverse): " << e.what() << std::endl;
            throw;
        }
    } else {
        // Create word-sized profile for the reverse read
        std::vector<int16_t> reverse_profile = qP_cuda<int16_t>(read_reverse, scoring_matrix, "word");
        
        bests_reverse = ssw_word_cuda(ref_num_subset, 1, read_reverse, weight_gapO, weight_gapE, 
                                    reverse_profile, r->score1, maskLen);
    }

    r->ref_begin1 = bests_reverse[0].ref;
    r->read_begin1 = r->read_end1 - bests_reverse[0].read;

    if (r->score1 > bests_reverse[0].score) {
        std::cerr << "Warning: The alignment path of one pair of sequences may miss a small part." << std::endl;
        r->flag = 2;
    }

    // Generate cigar
    const int32_t align_ref_len = r->ref_end1 - r->ref_begin1 + 1;
    const int32_t align_read_len = r->read_end1 - r->read_begin1 + 1;
    band_width = abs(align_ref_len - align_read_len) + 1;

    // Extract slice of reference and read sequences for banded alignment
    std::vector<int8_t> ref_slice(ref_num.begin() + r->ref_begin1, 
                                ref_num.begin() + r->ref_end1 + 1);
    std::vector<int8_t> read_slice(read_num.begin() + r->read_begin1, 
                                 read_num.begin() + r->read_end1 + 1);

    // Call banded_sw to get path (cigar)
    auto path = banded_sw(ref_slice, read_slice, r->score1, 
                         weight_gapO, weight_gapE, band_width, scoring_matrix);

    if (!path.empty()) {
        r->cigar = path;
    }

    return r;
}