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
 * @brief Custom atomicMax for int8_t type
 * CUDA doesn't provide built-in atomicMax for int8_t
 */
__device__ inline int8_t atomicMaxInt8(int8_t* address, int8_t val) {
    // Convert address to int
    unsigned int* address_as_int = (unsigned int*)((void*)address);
    
    // Calculate byte position within the int
    unsigned int byte_pos = ((unsigned long long)address & 3) * 8;
    
    // Create mask
    unsigned int mask = 0xFF << byte_pos;
    
    // Read current value
    unsigned int old = *address_as_int;
    unsigned int assumed;
    
    do {
        assumed = old;
        // Extract current value at byte position
        int8_t current = (int8_t)((old >> byte_pos) & 0xFF);
        
        // If current value is already greater or equal, no update needed
        if (current >= val) break;
        
        // Create new value with updated byte
        unsigned int new_val = (old & ~mask) | ((unsigned int)val << byte_pos);
        
        // Try to update
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
    
    // Return old value
    return (int8_t)((old >> byte_pos) & 0xFF);
}

/**
 * @brief Custom atomicMax for int16_t type
 * CUDA doesn't provide built-in atomicMax for int16_t
 */
__device__ inline int16_t atomicMaxInt16(int16_t* address, int16_t val) {
    // Convert address to int
    unsigned int* address_as_int = (unsigned int*)((void*)address);
    
    // Calculate half-word position within the int
    unsigned int byte_pos = ((unsigned long long)address & 2) * 8;
    
    // Create mask
    unsigned int mask = 0xFFFF << byte_pos;
    
    // Read current value
    unsigned int old = *address_as_int;
    unsigned int assumed;
    
    do {
        assumed = old;
        // Extract current value at half-word position
        int16_t current = (int16_t)((old >> byte_pos) & 0xFFFF);
        
        // If current value is already greater or equal, no update needed
        if (current >= val) break;
        
        // Create new value with updated half-word
        unsigned int new_val = (old & ~mask) | ((unsigned int)val << byte_pos);
        
        // Try to update
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
    
    // Return old value
    return (int16_t)((old >> byte_pos) & 0xFFFF);
}

/**
 * @brief Custom atomic exchange for int8_t type
 */
__device__ inline int8_t atomicExchInt8(int8_t* address, int8_t val) {
    // Convert address to int
    unsigned int* address_as_int = (unsigned int*)((void*)address);
    
    // Calculate byte position within the int
    unsigned int byte_pos = ((unsigned long long)address & 3) * 8;
    
    // Create mask
    unsigned int mask = 0xFF << byte_pos;
    
    // Read current value
    unsigned int old = *address_as_int;
    unsigned int assumed;
    
    do {
        assumed = old;
        // Create new value with updated byte
        unsigned int new_val = (old & ~mask) | ((unsigned int)val << byte_pos);
        
        // Try to update
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
    
    // Return old value
    return (int8_t)((old >> byte_pos) & 0xFF);
}

/**
 * @brief Custom atomic exchange for int16_t type
 */
__device__ inline int16_t atomicExchInt16(int16_t* address, int16_t val) {
    // Convert address to int
    unsigned int* address_as_int = (unsigned int*)((void*)address);
    
    // Calculate half-word position within the int
    unsigned int byte_pos = ((unsigned long long)address & 2) * 8;
    
    // Create mask
    unsigned int mask = 0xFFFF << byte_pos;
    
    // Read current value
    unsigned int old = *address_as_int;
    unsigned int assumed;
    
    do {
        assumed = old;
        // Create new value with updated half-word
        unsigned int new_val = (old & ~mask) | ((unsigned int)val << byte_pos);
        
        // Try to update
        old = atomicCAS(address_as_int, assumed, new_val);
    } while (assumed != old);
    
    // Return old value
    return (int16_t)((old >> byte_pos) & 0xFFFF);
}

/**
 * @brief Main CUDA kernel for Smith-Waterman alignment
 * 
 * This kernel processes one row of the dynamic programming matrix at a time,
 * following a similar approach as the SIMD version.
 * 
 * PARALLELIZATION: Each thread processes one reference position (one column of the DP matrix)
 * 
 * @tparam T Value type (int8_t or int16_t)
 */
template <typename T>
__global__ void ssw_kernel(
    const int8_t* ref_num,
    int8_t ref_dir,
    int32_t refLen,
    int32_t readLen,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const T* profile,
    T terminate,
    T bias,
    T* maxColumn,
    int32_t* end_read_column,
    int32_t* max_pos_ref,   // Reference position of max score
    int32_t* max_pos_read,  // Read position of max score
    T* max_score            // Global max score
) {
    // Thread index = reference position to process
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= refLen) return;
    
    // If we're in reverse direction, adjust position
    int32_t pos;
    if (ref_dir == 1) {
        pos = refLen - 1 - i;
    } else {
        pos = i;
    }
    
    // Allocate arrays for H, E (and F is scalar)
    // Use shared memory if readLen is small enough, otherwise use global memory
    extern __shared__ char shared_mem[];
    T* H;
    T* E;
    
    // Check if shared memory allocation is enough
    if (readLen * 2 * sizeof(T) <= 48 * 1024) { // Typical shared memory size per block
        // Use shared memory
        H = (T*)shared_mem + threadIdx.x * readLen * 2;
        E = H + readLen;
    } else {
        // Allocate in global memory
        H = new T[readLen];
        E = new T[readLen];
    }
    
    // Initialize arrays
    for (int32_t j = 0; j < readLen; ++j) {
        H[j] = 0;
        E[j] = 0;
    }
    
    // Get profile offset for current reference base
    int32_t nt = ref_num[pos];
    const T* vP = profile + (nt * readLen);
    
    // Initialize max tracking
    T max_val = 0;
    int32_t max_val_pos = -1;
    T F = 0;
    
    // Process all query positions
    for (int32_t j = 0; j < readLen; ++j) {
        // Calculate H[j]
        T H_diag = (j > 0) ? H[j-1] : 0;
        
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
        F = max(device_saturated_sub(F, weight_gapE), 
                device_saturated_sub(H[j], weight_gapO));
        
        // Get max of H, E, F
        H[j] = max(H[j], max(E[j], F));
        
        // Update local max
        if (H[j] > max_val) {
            max_val = H[j];
            max_val_pos = j;
        }
    }
    
    // Lazy_F loop - equivalent to SIMD version
    // This is a key part of the algorithm!
    for (int32_t k = 0; k < readLen; ++k) {
        bool early_terminate = true;
        T prev_F = 0;
        
        for (int32_t j = 0; j < readLen; ++j) {
            // Shift F value
            T curr_F = (j > 0) ? prev_F : 0;
            prev_F = F;
            
            // Compare with H and update
            if (H[j] < curr_F) {
                H[j] = curr_F;
                
                // Update local max if needed
                if (curr_F > max_val) {
                    max_val = curr_F;
                    max_val_pos = j;
                }
                
                // Calculate new F
                T H_gap = device_saturated_sub(H[j], weight_gapO);
                F = device_saturated_sub(curr_F, weight_gapE);
                F = max(F, H_gap);
                
                // Check if we need another iteration
                if (F > H_gap) {
                    early_terminate = false;
                }
            }
        }
        
        // If no updates were needed, we can exit the loop
        if (early_terminate) break;
    }
    
    // Update maxColumn and end_read_column
    maxColumn[pos] = max_val;
    end_read_column[pos] = max_val_pos;
    
    // Update global max using custom atomic operations
    if (max_val > 0) {
        // Use type-specific atomic operations
        T old_max;
        if (std::is_same<T, int8_t>::value) {
            old_max = atomicMaxInt8((int8_t*)max_score, max_val);
            
            // If we updated the max, also update positions
            if (max_val > old_max) {
                atomicExch(max_pos_ref, pos);
                atomicExch(max_pos_read, max_val_pos);
                
                // Debug output for new max
                printf("New max score at ref[%d] read[%d]: %d\n", pos, max_val_pos, (int)max_val);
            }
        } else if (std::is_same<T, int16_t>::value) {
            old_max = atomicMaxInt16((int16_t*)max_score, max_val);
            
            // If we updated the max, also update positions
            if (max_val > old_max) {
                atomicExch(max_pos_ref, pos);
                atomicExch(max_pos_read, max_val_pos);
                
                // Debug output for new max
                printf("New max score at ref[%d] read[%d]: %d\n", pos, max_val_pos, (int)max_val);
            }
        }
    }
    
    // Print debug info for every position
    // printf("CUDA Thread %d (ref pos %d): col_max = %d\n", i, pos, (int)max_val);
    
    // Free memory if allocated in global memory
    if (readLen * 2 * sizeof(T) > 48 * 1024) {
        delete[] H;
        delete[] E;
    }
}

/**
 * @brief Main CUDA Smith-Waterman implementation
 * 
 * @tparam T Value type (int8_t or int16_t)
 */
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
    int32_t maskLen) {
    
    // Calculate sizes
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
    T *d_max_score;
    int32_t *d_max_pos_ref, *d_max_pos_read;
    
    // Allocate and copy reference sequence
    CHECK_CUDA_ERROR(cudaMalloc(&d_ref_num, refLen * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ref_num, ref_num.data(), refLen * sizeof(int8_t), cudaMemcpyHostToDevice));
    
    // Allocate and copy profile
    size_t profile_size = profile.size() * sizeof(T);
    CHECK_CUDA_ERROR(cudaMalloc(&d_profile, profile_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_profile, profile.data(), profile_size, cudaMemcpyHostToDevice));
    
    // Allocate output arrays
    CHECK_CUDA_ERROR(cudaMalloc(&d_maxColumn, refLen * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_end_read_column, refLen * sizeof(int32_t)));
    
    // Allocate global max tracking
    CHECK_CUDA_ERROR(cudaMalloc(&d_max_score, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_max_pos_ref, sizeof(int32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_max_pos_read, sizeof(int32_t)));
    
    // Initialize output arrays
    CHECK_CUDA_ERROR(cudaMemset(d_maxColumn, 0, refLen * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemset(d_end_read_column, 0, refLen * sizeof(int32_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_max_score, 0, sizeof(T)));
    
    // Initialize max positions with sentinel values
    int32_t sentinel = -1;
    CHECK_CUDA_ERROR(cudaMemcpy(d_max_pos_ref, &sentinel, sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_max_pos_read, &sentinel, sizeof(int32_t), cudaMemcpyHostToDevice));
    
    // Configure kernel - PARALLELIZATION: Each thread handles one reference position
    int threadsPerBlock = 256;
    int blocksPerGrid = (refLen + threadsPerBlock - 1) / threadsPerBlock;
    
    // Calculate shared memory size per block
    size_t sharedMemSize = std::min(static_cast<size_t>(readLen * 2 * sizeof(T) * threadsPerBlock), 
                                  static_cast<size_t>(48 * 1024)); // 48KB typical max shared memory
    
    // Launch kernel - PARALLELIZATION: Each thread computes one column of the DP matrix
    ssw_kernel<T><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_ref_num, ref_dir, refLen, readLen,
        weight_gapO, weight_gapE, d_profile, terminate, bias,
        d_maxColumn, d_end_read_column,
        d_max_pos_ref, d_max_pos_read, d_max_score
    );
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    std::vector<T> maxColumn(refLen);
    std::vector<int32_t> end_read_column(refLen);
    T max_score;
    int32_t max_pos_ref, max_pos_read;
    
    CHECK_CUDA_ERROR(cudaMemcpy(maxColumn.data(), d_maxColumn, refLen * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(end_read_column.data(), d_end_read_column, refLen * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&max_score, d_max_score, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&max_pos_ref, d_max_pos_ref, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&max_pos_read, d_max_pos_read, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    // Handle overflow for int8_t
    uint16_t final_score;
    if (std::is_same<T, int8_t>::value) {
        auto max_with_bias = static_cast<uint16_t>(max_score) + static_cast<uint16_t>(bias);
        final_score = max_with_bias >= 255 ? 255 : max_score;
    } else {
        final_score = max_score;
    }
    
    // Find second best alignment
    T second_max = 0;
    int32_t second_max_pos = -1;
    
    // Search before the best alignment
    int32_t edge = (max_pos_ref - maskLen) > 0 ? (max_pos_ref - maskLen) : 0;
    for (int32_t i = 0; i < edge; i++) {
        if (maxColumn[i] > second_max) {
            second_max = maxColumn[i];
            second_max_pos = i;
        }
    }
    
    // Search after the best alignment
    edge = (max_pos_ref + maskLen) > refLen ? refLen : (max_pos_ref + maskLen);
    for (int32_t i = edge + 1; i < refLen; i++) {
        if (maxColumn[i] > second_max) {
            second_max = maxColumn[i];
            second_max_pos = i;
        }
    }
    
    // Debug output
    std::cout << "CUDA Final results:" << std::endl;
    std::cout << "  Best score: " << final_score << " at ref=" << max_pos_ref
              << ", read=" << max_pos_read << std::endl;
    // std::cout << "  Second best score: " << second_max << " at ref=" << second_max_pos << std::endl;
    
    // Prepare return values
    std::vector<alignment_end> results(2);
    
    // Set best alignment
    results[0].score = final_score;
    results[0].ref = max_pos_ref;
    results[0].read = max_pos_read;
    
    // Set second best alignment
    results[1].score = second_max;
    results[1].ref = second_max_pos;
    results[1].read = 0;  // We don't track read position for second best alignment
    
    // Free device memory
    cudaFree(d_ref_num);
    cudaFree(d_profile);
    cudaFree(d_maxColumn);
    cudaFree(d_end_read_column);
    cudaFree(d_max_score);
    cudaFree(d_max_pos_ref);
    cudaFree(d_max_pos_read);
    
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

    std::cout << "r->ref_begin1 = " << r->ref_begin1 << std::endl;
    std::cout << "r->read_begin1 = " << r->read_begin1 << std::endl;

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