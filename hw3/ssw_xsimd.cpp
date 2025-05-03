// ssw_xsimd.cpp

#include "ssw_xsimd.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <iomanip>
#include <type_traits>
#include <optional>
#include <cmath>

using namespace std;

struct alignment_end {
    uint16_t score;
    int32_t ref;    // reference ending position
    int32_t read;   // read ending position
};

struct s_align{
	uint16_t score1;
	uint16_t score2;
	int32_t ref_begin1;
	int32_t ref_end1;
	int32_t	read_begin1;
	int32_t read_end1;
	int32_t ref_end2;
	vector<pair<char, int>> cigar;
    uint16_t flag;
};

/**
 * Convert the coordinate in the scoring matrix into the coordinate in one line of the band.
 * 
 * @param u Reference to the output coordinate in the band
 * @param w Band width
 * @param i Row index in the original matrix
 * @param j Column index in the original matrix
 */
inline void set_u(int32_t& u, int32_t w, int32_t i, int32_t j) {
    int32_t x = i - w;
    x = x > 0 ? x : 0;
    u = j - x + 1;
}

/**
 * Convert the coordinate in the direction matrix into the coordinate in one line of the band.
 * 
 * @param u Reference to the output index in the direction line
 * @param w Band width
 * @param i Row index in the original matrix
 * @param j Column index in the original matrix
 * @param p Direction type (0, 1, or 2)
 */
inline void set_d(int32_t& u, int32_t w, int32_t i, int32_t j, int32_t p) {
    int32_t x = i - w;
    x = x > 0 ? x : 0;
    x = j - x;
    u = x * 3 + p;
}

/**
 * @brief Rounds up to next power of 2
 * 
 * @param x Value to round up
 * @return Next power of 2 greater than or equal to x
 */
template<typename T>
T kroundup32(T x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    ++x;
    return x;
}


template <typename T>
inline T saturated_sub(const T& a, const T& b) {
    T result = a - b;
    return xsimd::max(result, T(0));
}

/**
 * @brief Generates CIGAR string from traceback matrix
 * 
 * Reconstructs alignment path from the traceback matrix and returns it as
 * a vector of CIGAR operations.
 * 
 * @param traceback Traceback matrix
 * @param i Ending row index
 * @param j Ending column index
 * @return Vector of CIGAR operations
 */
vector<pair<char, int>> generate_cigar(
    const vector<vector<int8_t>>& traceback,
    int i, int j) {
    
    vector<pair<char, int>> cigar;
    char current_op = 0;
    int current_length = 0;
    
    while (i > 0 || j > 0) {
        char op;
        
        // Determine operation based on traceback matrix
        // 0 = diagonal (match/mismatch), 1 = up (insertion), 2 = left (deletion)
        if (i > 0 && j > 0 && traceback[i][j] == 0) {
            op = 'M';  // Match/mismatch
            i--; j--;
        } else if (j > 0 && traceback[i][j] == 1) {
            op = 'I';  // Insertion
            j--;
        } else if (i > 0 && traceback[i][j] == 2) {
            op = 'D';  // Deletion
            i--;
        } else {
            break;  // Should not happen with valid traceback
        }
        
        // Add to CIGAR
        if (op == current_op) {
            current_length++;
        } else {
            if (current_op != 0) {
                cigar.push_back(make_pair(current_op, current_length));
            }
            current_op = op;
            current_length = 1;
        }
    }
    
    // Add the last operation
    if (current_op != 0) {
        cigar.push_back(make_pair(current_op, current_length));
    }
    
    // Reverse the CIGAR since we traced from end to beginning
    reverse(cigar.begin(), cigar.end());
    
    return cigar;
}

// Baseline (non-SIMD) Smith-Waterman algorithm based on SSW implementation
AlignmentResult smith_waterman_baseline(const string& ref, const string& read, const ScoringMatrix& scoring) {
    AlignmentResult result;
    int refLen = ref.length();
    int readLen = read.length();
    
    // Initialize scoring and traceback matrices
    vector<vector<int>> scores(refLen + 1, vector<int>(readLen + 1, 0));
    vector<vector<int8_t>> traceback(refLen + 1, vector<int8_t>(readLen + 1, 0));
    
    // Separate matrices for gap tracking (affine gap penalty)
    vector<vector<int>> ins_scores(refLen + 1, vector<int>(readLen + 1, 0));
    vector<vector<int>> del_scores(refLen + 1, vector<int>(readLen + 1, 0));
    
    int max_score = 0;
    int max_i = 0, max_j = 0;
    
    // Fill the matrices
    for (int i = 1; i <= refLen; i++) {
        for (int j = 1; j <= readLen; j++) {
            // Calculate match/mismatch score
            int diag_score = scores[i-1][j-1] + (ref[i-1] == read[j-1] ? scoring.match : scoring.mismatch);
            
            // Calculate gap scores with affine penalty
            int new_ins = scores[i][j-1] + scoring.gap_open;
            int extend_ins = ins_scores[i][j-1] + scoring.gap_extend;
            ins_scores[i][j] = max(new_ins, extend_ins);
            
            int new_del = scores[i-1][j] + scoring.gap_open;
            int extend_del = del_scores[i-1][j] + scoring.gap_extend;
            del_scores[i][j] = max(new_del, extend_del);
            
            // Find the maximum score
            int max_val = max({0, diag_score, ins_scores[i][j], del_scores[i][j]});
            scores[i][j] = max_val;
            
            // Set traceback pointer
            if (max_val == 0) {
                traceback[i][j] = -1;  // End of alignment
            } else if (max_val == diag_score) {
                traceback[i][j] = 0;   // Diagonal (match/mismatch)
            } else if (max_val == ins_scores[i][j]) {
                traceback[i][j] = 1;   // Up (insertion in read)
            } else if (max_val == del_scores[i][j]) {
                traceback[i][j] = 2;   // Left (deletion in read)
            }
            
            // Update maximum score if necessary
            if (max_val > max_score) {
                max_score = max_val;
                max_i = i;
                max_j = j;
            }
        }
    }
    
    // Find second-best score using approach similar to SSW
    // Mask region around the best alignment and find next best
    int maskLen = readLen / 2; // Adjust based on your specific case
    int second_best_score = 0;
    
    // Edge cases for left side of the alignment
    int edge = (max_i - maskLen) > 0 ? (max_i - maskLen) : 0;
    for (int i = 0; i < edge; i++) {
        for (int j = 1; j <= readLen; j++) {
            if (scores[i][j] > second_best_score) {
                second_best_score = scores[i][j];
            }
        }
    }
    
    // Edge cases for right side of the alignment
    edge = (max_i + maskLen) > refLen ? refLen : (max_i + maskLen);
    for (int i = edge + 1; i <= refLen; i++) {
        for (int j = 1; j <= readLen; j++) {
            if (scores[i][j] > second_best_score) {
                second_best_score = scores[i][j];
            }
        }
    }
    
    // Set alignment result
    result.score1 = max_score;
    result.score2 = second_best_score;
    result.ref_end = max_i - 1;
    result.read_end = max_j - 1;
    
    // Trace back to find beginning of alignment
    int i = max_i, j = max_j;
    while (i > 0 && j > 0 && traceback[i][j] != -1) {
        if (traceback[i][j] == 0) {      // Diagonal
            i--; j--;
        } else if (traceback[i][j] == 1) { // Insertion
            j--;
        } else if (traceback[i][j] == 2) { // Deletion
            i--;
        } else {
            break; // End of alignment
        }
    }
    
    result.ref_begin = i;
    result.read_begin = j;
    
    // Generate CIGAR string
    result.cigar = generate_cigar(traceback, max_i, max_j);
    
    return result;
}

template <typename T, typename A = xsimd::default_arch>
auto qP_template(const vector<int8_t>& read_num, 
                 const vector<int8_t>& scoring_matrix) {
    using batch_type = xsimd::batch<T, A>;
    constexpr size_t batch_size = batch_type::size;
    const auto readLen = static_cast<int32_t>(read_num.size());
    const auto n = static_cast<int32_t>(sqrt(scoring_matrix.size()));
    
    if (n * n != static_cast<int32_t>(scoring_matrix.size())) {
        cerr << "ERROR: Scoring matrix size is not a perfect square" << endl;
        throw runtime_error("Scoring matrix size is not a perfect square");
    }
    
    // bias (only needed for int8_t, bias = abs(min neg. value of matrix))
    T bias = 0;
    if constexpr (is_same_v<T, int8_t>) {
        auto min_element = *std::min_element(scoring_matrix.begin(), scoring_matrix.end());
        bias = min_element < 0 ? static_cast<T>(abs(min_element)) : 0;
    }
    
    // Calculate segment length based on SIMD register width (done at compile time when possible)
    const auto segLen = (readLen + batch_size - 1) / batch_size;

    // Add memory allocation protection
    try {
        // Allocate memory for the profile
        // cerr << "DEBUG: Attempting to allocate profile vector with size: " << n * segLen << endl;
        vector<batch_type> profile;
        
        // Try to reserve memory before allocation to avoid reallocation
        profile.reserve(n * segLen);
        profile.resize(n * segLen);
        
        // cerr << "DEBUG: Profile allocation successful" << endl;
        auto* t = reinterpret_cast<T*>(profile.data());
        
        // Generate query profile in stripped pattern
        for (int32_t nt = 0; nt < n; ++nt) {
            for (int32_t i = 0; i < segLen; ++i) {
                int32_t j = i;
                for (int32_t segNum = 0; segNum < batch_size; ++segNum) {
                    // Calculate vector index for debugging
                    size_t profile_idx = ((nt * segLen + i) * batch_size) + segNum;
                    
                    // Check if we're about to go out of bounds
                    if (j >= readLen) {
                        // Use if constexpr for compile-time branch selection
                        if constexpr (is_same_v<T, int8_t>) {
                            *t++ = bias;
                        } else {
                            *t++ = 0;
                        }
                    } else {
                        // Verify valid indices for scoring matrix lookup
                        int matrix_idx = nt * n + read_num[j];
                        if (matrix_idx < 0 || matrix_idx >= static_cast<int32_t>(scoring_matrix.size())) {
                            cerr << "ERROR: Invalid matrix index: " << matrix_idx 
                                << " (max: " << scoring_matrix.size() - 1 
                                << ") at nt=" << nt << ", read_num[" << j << "]=" 
                                << static_cast<int>(read_num[j]) << endl;
                            throw out_of_range("Scoring matrix access out of bounds");
                        }
                        
                        // Use if constexpr for compile-time branch selection
                        if constexpr (is_same_v<T, int8_t>) {
                            *t++ = scoring_matrix[matrix_idx] + bias;
                        } else {
                            *t++ = scoring_matrix[matrix_idx];
                        }
                    }
                    j += segLen;
                    
                    // Periodically report progress for very large sequences
                    if (profile_idx > 0 && profile_idx % 1000000 == 0) {
                        // cerr << "DEBUG: Processed " << profile_idx << " elements" << endl;
                    }
                }
            }
        }
        
        return profile;
        
    } catch (const std::length_error& e) {
        cerr << "ERROR: Vector allocation failed with length_error: " << e.what() << endl;
        cerr << "  Attempted to allocate profile of size: " << n * segLen << endl;
        cerr << "  Consider using smaller sequences or more memory" << endl;
        throw; // Re-throw to be handled by the caller
    } catch (const std::bad_alloc& e) {
        cerr << "ERROR: Memory allocation failed with bad_alloc: " << e.what() << endl;
        throw; // Re-throw to be handled by the caller
    } catch (const std::exception& e) {
        cerr << "ERROR: Exception during profile generation: " << e.what() << endl;
        throw; // Re-throw to be handled by the caller
    }
}

/**
 * Query profile for byte precision (int8_t)
 */
template <typename A = xsimd::default_arch>
auto qP_byte(const vector<int8_t>& read_num, 
             const vector<int8_t>& scoring_matrix) {
    return qP_template<int8_t, A>(read_num, scoring_matrix);
}

/**
 * Query profile for word precision (int16_t)
 */
template <typename A = xsimd::default_arch>
auto qP_word(const vector<int8_t>& read_num, 
             const vector<int8_t>& scoring_matrix) {
    return qP_template<int16_t, A>(read_num, scoring_matrix);
}

/**
 * @brief Striped Smith-Waterman algorithm with xsimd vectorization
 * 
 * Record the highest score of each reference position.
 * Return the alignment score and ending position of the best alignment, 2nd best alignment, etc.
 * Gap begin and gap extension are different.
 * weight_match > 0, all other weights < 0.
 * The returned positions are 0-based.
 * 
 * @tparam T Value type (int8_t or int16_t)
 * @tparam A Architecture type for xsimd (auto-detected if not specified)
 * @param ref_num Reference sequence encoded as numbers
 * @param ref_dir Direction of reference (0: forward, 1: reverse)
 * @param read_num Query sequence encoded as numbers
 * @param weight_gapO Gap open penalty
 * @param weight_gapE Gap extension penalty
 * @param profile Query profile
 * @param terminate Termination score (0 to disable)
 * @param bias Value to add to avoid negative numbers (only used for int8_t version)
 * @param maskLen Length of mask for the second best alignment
 * @return vector<alignment_end> Alignment results (best and second best)
 */
template <typename T, typename A = xsimd::default_arch>
vector<alignment_end> ssw_template(const vector<int8_t>& ref_num,
                               int8_t ref_dir,
                               const vector<int8_t>& read_num,
                               const uint8_t weight_gapO,
                               const uint8_t weight_gapE,
                               const vector<typename xsimd::batch<T, A>>& profile,
                               T terminate,
                               T bias,
                               int32_t maskLen) {
    using batch_type = xsimd::batch<T, A>;
    constexpr size_t batch_size = batch_type::size;
    
    // Calculate sizes
    const auto refLen = static_cast<int32_t>(ref_num.size());
    const auto readLen = static_cast<int32_t>(read_num.size());
    
    // Determine segLen based on the precision (batch_size)
    const auto segLen = (readLen + batch_size - 1) / batch_size;
    
    // Use provided maskLen or default to readLen/2 if not specified
    if (maskLen <= 0) {
        maskLen = readLen / 2;
    }

    // 直接找最大元素（沒有平行到）
    auto max_batch = [](auto& m, auto& vm) {
        using batch_type = std::remove_reference_t<decltype(vm)>;
        using value_type = typename batch_type::value_type;

        value_type max_val = vm.get(0);
        for (size_t i = 1; i < batch_type::size; ++i) {
            value_type val = vm.get(i);
            if (val > max_val) {
                max_val = val;
            }
        }
        
        m = max_val;
    };

    
    // Initialize max score and positions
    T max = 0;
    int32_t end_read = readLen - 1;
    int32_t end_ref = -1;
    
    // maxColumn: largest score of each reference position
    // end_read_column: "read_end" of the largest score alignment of each reference position
    vector<T> maxColumn(refLen, 0);
    vector<int32_t> end_read_column(refLen, 0);
    
    // Define 0 vector
    batch_type vZero = batch_type(static_cast<T>(0));
    
    // Allocate SIMD vectors
    vector<batch_type> vHStore(segLen);
    vector<batch_type> vHLoad(segLen);
    vector<batch_type> vE(segLen);
    vector<batch_type> vHmax(segLen);
    
    // Create gap penalties
    batch_type vGapO = batch_type(static_cast<T>(weight_gapO));
    batch_type vGapE = batch_type(static_cast<T>(weight_gapE));
    
    // Create bias vector (only used for int8_t)
    batch_type vBias;
    if constexpr (is_same_v<T, int8_t>) {
        vBias = batch_type(static_cast<T>(bias));
    }
    
    // Trace variables
    batch_type vMaxScore = vZero;
    batch_type vMaxMark = vZero;
    
    // Outer loop bounds
    int32_t begin = 0, end = refLen, step = 1;
    if (ref_dir == 1) {
        begin = refLen - 1;
        end = -1;
        step = -1;
    }
    
    // Outer loop to process the reference sequence
    for (int32_t i = begin; i != end; i += step) {
        batch_type vF = vZero;
        batch_type vMaxColumn = vZero;
        
        // Initialize H with previous end value
        batch_type vH = vHStore[segLen - 1];
        
        // Shift the 128-bit value in vH left
        if constexpr (is_same_v<T, int8_t>) {
            vH = xsimd::slide_left<1>(vH);
        } else if constexpr (is_same_v<T, int16_t>) {
            vH = xsimd::slide_left<2>(vH);
        } else {
            throw runtime_error("Unsupported type for SIMD alignment");
        }
        
        // Swap the two H buffers
        swap(vHLoad, vHStore);

        // Get profile pointer for current reference base
        const batch_type* vP = profile.data() + ref_num[i] * segLen;
        

        
        // Inner loop to process the query sequence
        for (int32_t j = 0; j < segLen; j++) {
            // Add profile score to vH
            vH = vH + vP[j];
            
            // Subtract bias if using uint8_t
            if constexpr (is_same_v<T, int8_t>) {
                vH = vH - vBias;
            }
            
            // Get max from vH, vE and vF
            auto e = vE[j];
            vH = xsimd::max(vH, e);
            vH = xsimd::max(vH, vF);
            vMaxColumn = xsimd::max(vMaxColumn, vH);
            
            // Save vH
            vHStore[j] = vH;
            
            // Update vE value (use saturation arithmetic, result >= 0)
            auto vH_gap = saturated_sub(vH, vGapO);
            e = saturated_sub(e, vGapE);
            e = xsimd::max(e, vH_gap);
            vE[j] = e;
            
            // Update vF value (use saturation arithmetic, result >= 0)
            vF = saturated_sub(vF, vGapE);
            vF = xsimd::max(vF, vH_gap);
            
            // Load the next vH
            vH = vHLoad[j];
        }
        
        // Lazy_F loop
        for (int32_t k = 0; k < batch_size; k++) {
            // Shift vF
            if constexpr (is_same_v<T, int8_t>) {
                vF = xsimd::slide_left<1>(vF);
            } else {
                vF = xsimd::slide_left<2>(vF);
            }
                
            for (int32_t j = 0; j < segLen; j++) {
                vH = vHStore[j];
                vH = xsimd::max(vH, vF);
                vMaxColumn = xsimd::max(vMaxColumn, vH);
                vHStore[j] = vH;
                
                // Calculate new vF (use saturation arithmetic, result >= 0)
                auto vH_gap = saturated_sub(vH, vGapO);
                vF = saturated_sub(vF, vGapE);
                
                // Early termination check
                if constexpr (is_same_v<T, int8_t>) {
                    auto vTemp = saturated_sub(vF, vH_gap);
                    auto mask = xsimd::eq(vTemp, vZero); // 0xffff
                    if (xsimd::all(mask)) {
                        // fprintf(stderr, "DEBUG C++: goto end\n");
                        goto end;
                    }
                } else {
                    auto mask = xsimd::gt(vF, vH_gap);
                    if (!xsimd::any(mask)) goto end;
                }
            }
        }
        
end:
        // Update max score
        vMaxScore = xsimd::max(vMaxScore, vMaxColumn);
        auto vTemp = xsimd::eq(vMaxMark, vMaxScore);

        // Check if we have a new maximum
        if (!xsimd::all(vTemp)) {
            T temp;
            vMaxMark = vMaxScore;
            max_batch(temp, vMaxScore);
            vMaxScore = vMaxMark;
            
            if (temp > max) {
                max = temp;
                
                // Overflow check for uint8_t
                if constexpr (is_same_v<T, int8_t>) {
                    auto max_with_bias = static_cast<uint16_t>(max) + static_cast<uint16_t>(bias);
                    if (max_with_bias >= 255) break;
                }
                
                end_ref = i;
                vHmax = vHStore;
            }
        }
        
        // Record max score of current column
        T col_max;
        max_batch(col_max, vMaxColumn);
        maxColumn[i] = col_max;
        
        // Check for termination
        if (col_max == terminate) break;
    }
    
    // Trace the alignment ending position on read
    T* t = reinterpret_cast<T*>(vHmax.data());
    const int32_t column_len = segLen * batch_size;
    for (int32_t i = 0; i < column_len; i++) {
        if (t[i] == max) {
            int32_t temp = i / batch_size + (i % batch_size) * segLen;
            if (temp < end_read) end_read = temp;
        }
    }
    
    // Prepare return value with best and second best alignment
    vector<alignment_end> results(2);
    
    // Set best alignment
    if constexpr (is_same_v<T, int8_t>) {
        auto max_with_bias = static_cast<uint16_t>(max) + static_cast<uint16_t>(bias);
        results[0].score = max_with_bias >= 255 ? 255 : max;
    } else {
        results[0].score = max;
    }
    results[0].ref = end_ref;
    results[0].read = end_read;
    
    // Find second best alignment
    results[1].score = 0;
    results[1].ref = 0;
    results[1].read = 0;
    
    // Search before the best alignment
    int32_t edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
    for (int32_t i = 0; i < edge; i++) {
        if (maxColumn[i] > results[1].score) {
            results[1].score = maxColumn[i];
            results[1].ref = i;
        }
    }
    
    // Search after the best alignment
    edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
    for (int32_t i = edge + 1; i < refLen; i++) {
        if (maxColumn[i] > results[1].score) {
            results[1].score = maxColumn[i];
            results[1].ref = i;
        }
    }
    
    // cerr << "DEBUG C++: Final values:" << endl;
    // cerr << "DEBUG C++: results[0]: score=" << results[0].score << ", ref=" << results[0].ref
    //     << ", read=" << results[0].read << endl;
    // cerr << "DEBUG C++: results[1]: score=" << results[1].score << ", ref=" << results[1].ref
    //     << ", read=" << results[1].read << endl;

    return results;
}

/**
 * Smith-Waterman for byte precision (int8_t)
 */
template <typename A = xsimd::default_arch>
vector<alignment_end> ssw_byte(const vector<int8_t>& ref_num,
                             int8_t ref_dir,
                             const vector<int8_t>& read_num,
                             const uint8_t weight_gapO,
                             const uint8_t weight_gapE,
                             const vector<xsimd::batch<int8_t, A>>& profile,
                             uint8_t terminate,
                             uint8_t bias,
                             int32_t maskLen = 0) {
    return ssw_template<int8_t, A>(ref_num, ref_dir, read_num, weight_gapO, weight_gapE, profile, terminate, bias, maskLen);
}

/**
 * Smith-Waterman for word precision (int16_t)
 */
template <typename A = xsimd::default_arch>
vector<alignment_end> ssw_word(const vector<int8_t>& ref_num,
                             int8_t ref_dir,
                             const vector<int8_t>& read_num,
                             const uint8_t weight_gapO,
                             const uint8_t weight_gapE,
                             const vector<xsimd::batch<int16_t, A>>& profile,
                             uint16_t terminate = 0,
                             int32_t maskLen = 0) {
    // For int16_t, bias is not used, pass 0
    return ssw_template<int16_t, A>(ref_num, ref_dir, read_num, weight_gapO, weight_gapE, profile, terminate, 0, maskLen);
}

/**
 * @brief Reverses a sequence
 * 
 * @param seq_num Sequence to reverse
 * @param seq_end Ending position (0-based)
 * @return Reversed sequence
 */
static vector<int8_t> seq_reverse(const vector<int8_t>& seq_num,
                                int32_t seq_end) {
    // seq_end is 0-based alignment ending position
    vector<int8_t> reverse(seq_end + 1, 0);
    int32_t start = 0;
    
    while (start <= seq_end) {
        reverse[start] = seq_num[seq_end];
        reverse[seq_end] = seq_num[start];
        ++start;
        --seq_end;
    }
    
    return reverse;
}

/**
 * @brief Performs banded Smith-Waterman alignment
 * 
 * @param ref_num Reference sequence
 * @param read_num Read sequence
 * @param score Target score for alignment
 * @param weight_gapO Gap opening penalty (will be used as negative)
 * @param weight_gapE Gap extension penalty (will be used as negative)
 * @param band_width Width of the band for alignment
 * @param scoring_matrix Scoring matrix for matches/mismatches
 * @return Cigar string representing the alignment
 */
static vector<pair<char, int>> banded_sw(
    const vector<int8_t>& ref_num,
    const vector<int8_t>& read_num,
    int32_t score,
    const uint32_t weight_gapO,
    const uint32_t weight_gapE,
    int32_t band_width,
    const vector<int8_t>& scoring_matrix) {
    
    // Get dimensions from input vectors
    const int32_t refLen = static_cast<int32_t>(ref_num.size());
    const int32_t readLen = static_cast<int32_t>(read_num.size());
    
    // Validate and determine scoring matrix dimension
    const auto n = static_cast<int32_t>(sqrt(scoring_matrix.size()));
    if (n * n != static_cast<int32_t>(scoring_matrix.size())) {
        throw runtime_error("Scoring matrix size is not a perfect square");
    }

    // Initialize dynamic memory with vectors
    vector<uint32_t> c(16);
    int32_t i, j, e, f, temp1, temp2, s1 = 8, max = 0, len; // l = 0
    int64_t s2 = 1024;
    char op, prev_op;
    int32_t width, width_d;
    
    len = std::max(refLen, readLen);
    
    vector<int32_t> h_b(s1);
    vector<int32_t> e_b(s1);
    vector<int32_t> h_c(s1);
    vector<int8_t> direction(s2);
    int8_t* direction_line;

    do {
        width = band_width * 2 + 3;
        width_d = band_width * 2 + 1;
        
        // Resize vectors if needed
        if (width >= s1) {
            s1 = width;
            kroundup32(s1);  // Round up to next power of 2
            h_b.resize(s1);
            e_b.resize(s1);
            h_c.resize(s1);
        }
        
        if (width_d * readLen * 3 >= s2) {
            s2 = width_d * readLen * 3;
            kroundup32(s2);  // Round up to next power of 2
            direction.resize(s2);
        }
        
        direction_line = direction.data();
        
        // Initialize first row
        for (j = 1; j < width - 1; j++) {
            h_b[j] = 0;
        }
        
        // Main DP loop
        for (i = 0; i < readLen; i++) {
            int32_t beg = 0, end = refLen - 1, u = 0, edge;
            j = i - band_width;
            beg = std::max(beg, j);  // band start
            j = i + band_width;
            end = min(end, j);  // band end
            edge = min(end + 1, width - 1);
            
            f = h_b[0] = e_b[0] = h_b[edge] = e_b[edge] = h_c[0] = 0;
            direction_line = direction.data() + width_d * i * 3;

            for (j = beg; j <= end; j++) {
                int32_t b, e1, f1, d, de, df, dh;
                set_u(u, band_width, i, j);
                set_u(e, band_width, i - 1, j);
                set_u(b, band_width, i, j - 1);
                set_u(d, band_width, i - 1, j - 1);
                set_d(de, band_width, i, j, 0);
                set_d(df, band_width, i, j, 1);
                set_d(dh, band_width, i, j, 2);

                temp1 = i == 0 ? -weight_gapO : h_b[e] - weight_gapO;
                temp2 = i == 0 ? -weight_gapE : e_b[e] - weight_gapE;
                e_b[u] = std::max(temp1, temp2);
                direction_line[de] = temp1 > temp2 ? 3 : 2;

                temp1 = h_c[b] - weight_gapO;
                temp2 = f - weight_gapE;
                f = std::max(temp1, temp2);
                direction_line[df] = temp1 > temp2 ? 5 : 4;

                e1 = std::max(e_b[u], 0);
                f1 = std::max(f, 0);
                temp1 = std::max(e1, f1);
                temp2 = h_b[d] + scoring_matrix[ref_num[j] * n + read_num[i]];
                h_c[u] = std::max(temp1, temp2);

                max = std::max(max, h_c[u]);

                if (temp1 <= temp2) {
                    direction_line[dh] = 1;
                } else {
                    direction_line[dh] = e1 > f1 ? direction_line[de] : direction_line[df];
                }
            }
            
            for (j = 1; j <= u; j++) {
                h_b[j] = h_c[j];
            }
        }
        
        band_width *= 2;
    } while (max < score && band_width <= len);
    
    band_width /= 2;

    // Trace back
    i = readLen - 1;
    j = refLen - 1;
    e = 0;      // Count the number of M, D or I.
    // l = 0;      // Record length of current cigar
    op = prev_op = 'M';
    temp2 = 2;  // h
    direction_line = direction.data() + width_d * i * 3;
    
    vector<pair<char, int>> cigar_ops;
    
    while (i >= 0 && j > 0) {
        set_d(temp1, band_width, i, j, temp2);
        switch (direction_line[temp1]) {
            case 1:
                --i;
                --j;
                temp2 = 2;
                direction_line -= width_d * 3;
                op = 'M';
                break;
            case 2:
                --i;
                temp2 = 0;    // e
                direction_line -= width_d * 3;
                op = 'I';
                break;
            case 3:
                --i;
                temp2 = 2;
                direction_line -= width_d * 3;
                op = 'I';
                break;
            case 4:
                --j;
                temp2 = 1;
                op = 'D';
                break;
            case 5:
                --j;
                temp2 = 2;
                op = 'D';
                break;
            default:
                throw runtime_error("Trace back error: " + 
                                    to_string(direction_line[temp1]));
        }
        
        if (op == prev_op) {
            ++e;
        } else {
            cigar_ops.push_back(make_pair(prev_op, e));
            prev_op = op;
            e = 1;
        }
    }
    
    // Add final cigar operation
    if (op == 'M') {
        cigar_ops.push_back(make_pair(op, e + 1));
    } else {
        cigar_ops.push_back(make_pair(op, e));
        cigar_ops.push_back(make_pair('M', 1));
    }

    // Reverse cigar
    vector<pair<char, int>> result;
    result.reserve(cigar_ops.size());
    for (auto it = cigar_ops.rbegin(); it != cigar_ops.rend(); ++it) {
        result.push_back(*it);
    }
    
    return result;
}

/**
 * Main function for the Smith-Waterman alignment algorithm with SSE acceleration
 * 
 * @param profile_byte Byte-sized profile for SSE computations
 * @param profile_word Word-sized profile for SSE computations
 * @param ref_num Reference sequence
 * @param read_num Read sequence
 * @param weight_gapO Gap opening penalty
 * @param weight_gapE Gap extension penalty 
 * @param maskLen Length of the mask for the optimal alignment
 * @param bias Bias for byte calculations
 * @param n Size of the scoring matrix row/column
 * @return Unique pointer to alignment structure containing results
 */
template <typename A = xsimd::default_arch>
unique_ptr<s_align> ssw_main(
    const vector<xsimd::batch<int8_t, A>>& profile_byte,
    const vector<xsimd::batch<int16_t, A>>& profile_word,
    const vector<int8_t>& ref_num,
    const vector<int8_t>& read_num,
    const vector<int8_t>& scoring_matrix,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const int32_t maskLen,
    const uint8_t bias) {

    // Initialize result structure
    auto r = make_unique<s_align>();
    r->ref_begin1 = -1;
    r->read_begin1 = -1;
    r->flag = 0;

    // Check for valid maskLen
    if (maskLen < 15) {
        cerr << "When maskLen < 15, the function doesn't return 2nd best alignment information." << endl;
    }

    bool word = false;
    int32_t band_width = 0;
    vector<alignment_end> bests;
    vector<alignment_end> bests_reverse;

    // Find the alignment scores and ending positions
    if (!profile_byte.empty()) {
        try {
            // The actual function call
            bests = ssw_byte(ref_num, 0, read_num, weight_gapO, weight_gapE, 
                            profile_byte, -1, bias, maskLen);
            
            // Print some information about the results if available
            if (bests.empty()) {
                cerr << "DEBUG C++:   No alignments returned" << endl;
            }
        } catch (const std::length_error& e) {
            cerr << "ERROR: Vector length error in ssw_byte: " << e.what() << endl;
            throw; // Re-throw to be handled by the caller
        } catch (const std::bad_alloc& e) {
            cerr << "ERROR: Memory allocation failed in ssw_byte: " << e.what() << endl;
            throw; // Re-throw to be handled by the caller
        } catch (const std::exception& e) {
            cerr << "ERROR: Exception in ssw_byte: " << e.what() << endl;
            throw; // Re-throw to be handled by the caller
        }
        
        if (!profile_word.empty() && bests[0].score == 255) {
            bests = ssw_word(ref_num, 0, read_num, weight_gapO, weight_gapE, 
                           profile_word, -1, maskLen);
            word = true;
        } else if (bests[0].score == 255) {
            cerr << "Please set 2 to the score_size parameter of the initialization, "
                      << "otherwise the alignment results will be incorrect." << endl;
            return nullptr;
        }
    } else if (!profile_word.empty()) {
        bests = ssw_word(ref_num, 0, read_num, weight_gapO, weight_gapE, 
                       profile_word, -1, maskLen);
        word = true;
    } else {
        cerr << "Please initialize profiles before calling ssw_main." << endl;
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
    
    vector<int8_t> ref_num_subset;
    // Make sure r->ref_end1 is valid and not negative
    if (r->ref_end1 >= 0 && r->ref_end1 < static_cast<int32_t>(ref_num.size())) {
        // Create a subset containing elements from 0 to ref_end1 (inclusive)
        ref_num_subset.assign(ref_num.begin(), ref_num.begin() + r->ref_end1 + 1);
    } else {
        // Fallback to using the whole reference if ref_end1 is invalid
        ref_num_subset = ref_num;
        cerr << "WARNING: Invalid r->ref_end1, using full reference" << endl;
    }

    // Store second-best score if available
    if (maskLen >= 15) {
        r->score2 = bests[1].score;
        r->ref_end2 = bests[1].ref;
    } else {
        r->score2 = 0;
        r->ref_end2 = -1;
    }

    // Find the beginning position of the best alignment
    auto read_reverse = seq_reverse(read_num, r->read_end1);

    if (!word) {
        // Create byte-sized profile for the reverse read
        auto vP = qP_byte<A>(read_reverse, scoring_matrix);
        try {
            bests_reverse = ssw_byte(ref_num_subset, 1, read_reverse, weight_gapO, weight_gapE, 
                               vP, r->score1, bias, maskLen);
            if (bests_reverse.empty()) {
              cerr << "DEBUG:   No reverse alignments returned" << endl;
            }
        } catch (const std::length_error& e) {
            cerr << "ERROR: Vector length error in ssw_byte (reverse): " << e.what() << endl;
            throw;
        } catch (const std::bad_alloc& e) {
            cerr << "ERROR: Memory allocation failed in ssw_byte (reverse): " << e.what() << endl;
            throw;
        } catch (const std::exception& e) {
            cerr << "ERROR: Exception in ssw_byte (reverse): " << e.what() << endl;
            throw;
        }
    } else {
        // Create word-sized profile for the reverse read
        auto vP = qP_word<A>(read_reverse, scoring_matrix);
        bests_reverse = ssw_word(ref_num, 1, read_reverse, weight_gapO, weight_gapE, 
                               vP, r->score1, maskLen);
    }

    r->ref_begin1 = bests_reverse[0].ref;
    r->read_begin1 = r->read_end1 - bests_reverse[0].read;

    if (r->score1 > bests_reverse[0].score) {
        cerr << "Warning: The alignment path of one pair of sequences may miss a small part." << endl;
        r->flag = 2;
    }

    // Generate cigar
    const int32_t align_ref_len = r->ref_end1 - r->ref_begin1 + 1;
    const int32_t align_read_len = r->read_end1 - r->read_begin1 + 1;
    band_width = abs(align_ref_len - align_read_len) + 1;

    // Extract slice of reference and read sequences for banded alignment
    vector<int8_t> ref_slice(ref_num.begin() + r->ref_begin1, 
                                ref_num.begin() + r->ref_end1 + 1);
    vector<int8_t> read_slice(read_num.begin() + r->read_begin1, 
                                 read_num.begin() + r->read_end1 + 1);

    // Create scoring matrix (assuming it's already passed as a parameter to ssw_main)
    auto path = banded_sw(ref_slice, read_slice, r->score1, 
                         weight_gapO, weight_gapE, band_width, scoring_matrix);

    if (!path.empty()) {
        r->cigar = path;
    }

    return r;
}

AlignmentResult smith_waterman_striped_simd(
    const string& ref, 
    const string& read, 
    const ScoringMatrix& scoring) {
    
    AlignmentResult result;
    
    // Convert strings to numeric representation
    vector<int8_t> ref_num(ref.length());
    vector<int8_t> read_num(read.length());
    
    // Map nucleotides to their numeric values
    static const unordered_map<char, int8_t> nucleotide_map = {
        {'A', static_cast<int8_t>(NucleotideType::A)},
        {'C', static_cast<int8_t>(NucleotideType::C)},
        {'G', static_cast<int8_t>(NucleotideType::G)},
        {'T', static_cast<int8_t>(NucleotideType::T)},
        {'N', static_cast<int8_t>(NucleotideType::N)},
        {'a', static_cast<int8_t>(NucleotideType::A)},
        {'c', static_cast<int8_t>(NucleotideType::C)},
        {'g', static_cast<int8_t>(NucleotideType::G)},
        {'t', static_cast<int8_t>(NucleotideType::T)},
        {'n', static_cast<int8_t>(NucleotideType::N)}
    };
    
    // Convert sequences
    for (size_t i = 0; i < ref.length(); ++i) {
        auto it = nucleotide_map.find(ref[i]);
        ref_num[i] = (it != nucleotide_map.end()) ? it->second : static_cast<int8_t>(NucleotideType::N);
    }
    
    for (size_t i = 0; i < read.length(); ++i) {
        auto it = nucleotide_map.find(read[i]);
        read_num[i] = (it != nucleotide_map.end()) ? it->second : static_cast<int8_t>(NucleotideType::N);
    }
    
    // Create scoring matrix
    const int n = 5; // Size of nucleotide types (A, C, G, T, N)
    vector<int8_t> scoring_matrix(n * n, scoring.mismatch);
    
    // Set match scores on the diagonal
    for (int i = 0; i < n-1; ++i) {
        scoring_matrix[i * n + i] = scoring.match;
    }
    // N matches with anything
    for (int i = 0; i < n; ++i) {
        scoring_matrix[4 * n + i] = 0;
        scoring_matrix[i * n + 4] = 0;
    }
    
    // Calculate appropriate mask length (half of read length)
    int32_t readLen = static_cast<int32_t>(read.length());
    int32_t maskLen = readLen / 2;
    
    // Create byte profile
    // Create word profile (for scores that might overflow byte representation)
    auto profile_byte = qP_byte(read_num, scoring_matrix);
    auto profile_word = qP_word(read_num, scoring_matrix);

    // Get gap penalties from scoring structure
    uint8_t weight_gapO = static_cast<uint8_t>(abs(scoring.gap_open));
    uint8_t weight_gapE = static_cast<uint8_t>(abs(scoring.gap_extend));
    
    // Calculate appropriate bias for byte calculations
    uint8_t bias = 0;
    bias = static_cast<uint8_t>(abs(min(scoring.match, scoring.mismatch)));

    // Perform alignment
    auto alignment = ssw_main(
        profile_byte,
        profile_word,
        ref_num,
        read_num,
        scoring_matrix,
        weight_gapO,
        weight_gapE,
        maskLen,
        bias
    );

    // Check if alignment succeeded
    if (!alignment) {
        cerr << "Warning: Alignment failed between the provided sequences." << endl;
        return result;
    }
    
    // Transfer results
    result.score1 = alignment->score1;
    result.score2 = alignment->score2;
    result.ref_begin = alignment->ref_begin1;
    result.ref_end = alignment->ref_end1;
    result.read_begin = alignment->read_begin1;
    result.read_end = alignment->read_end1;
    result.cigar = alignment->cigar;
    return result;
}