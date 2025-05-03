/**
 * @file ssw_common.cpp
 * @brief Common functions for Smith-Waterman alignment
 * 
 * This file contains utility functions that are used by both SIMD and CUDA
 * implementations of the Smith-Waterman algorithm.
 */

#include "ssw_common.h"
#include "ssw_xsimd.h"
#include "ssw_cuda.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <utility>

// Helper macro for rounding up to next power of 2
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

// Helper macros for banded Smith-Waterman
#define set_u(u, w, i, j) { int x=(i)-(w); x=x>0?x:0; (u)=(j)-x+1; }
#define set_d(u, w, i, j, p) { int x=(i)-(w); x=x>0?x:0; x=(j)-x; (u)=((x)*(p)); }

/**
 * @brief Reverses a sequence
 * 
 * @param seq_num Sequence to reverse
 * @param seq_end Ending position (0-based)
 * @return Reversed sequence
 */
std::vector<int8_t> seq_reverse(const std::vector<int8_t>& seq_num, int32_t seq_end) {
    // seq_end is 0-based alignment ending position
    std::vector<int8_t> reverse(seq_end + 1, 0);
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
 * @brief Performs banded Smith-Waterman alignment traceback
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
std::vector<std::pair<char, int>> banded_sw(
    const std::vector<int8_t>& ref_num,
    const std::vector<int8_t>& read_num,
    int32_t score,
    const uint32_t weight_gapO,
    const uint32_t weight_gapE,
    int32_t band_width,
    const std::vector<int8_t>& scoring_matrix) {
    
    // Get dimensions from input vectors
    const int32_t refLen = static_cast<int32_t>(ref_num.size());
    const int32_t readLen = static_cast<int32_t>(read_num.size());
    
    // Validate and determine scoring matrix dimension
    const auto n = static_cast<int32_t>(sqrt(scoring_matrix.size()));
    if (n * n != static_cast<int32_t>(scoring_matrix.size())) {
        throw std::runtime_error("Scoring matrix size is not a perfect square");
    }

    // Initialize dynamic memory with vectors
    std::vector<uint32_t> c(16);
    int32_t i, j, e, f, temp1, temp2, s1 = 8, max = 0, len; // l = 0
    int64_t s2 = 1024;
    char op, prev_op;
    int32_t width, width_d;
    
    len = std::max(refLen, readLen);
    
    std::vector<int32_t> h_b(s1);
    std::vector<int32_t> e_b(s1);
    std::vector<int32_t> h_c(s1);
    std::vector<int8_t> direction(s2);
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
            end = std::min(end, j);  // band end
            edge = std::min(end + 1, width - 1);
            
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
    
    std::vector<std::pair<char, int>> cigar_ops;
    
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
                throw std::runtime_error("Trace back error: " + 
                                      std::to_string(direction_line[temp1]));
        }
        
        if (op == prev_op) {
            ++e;
        } else {
            cigar_ops.push_back(std::make_pair(prev_op, e));
            prev_op = op;
            e = 1;
        }
    }
    
    // Add final cigar operation
    if (op == 'M') {
        cigar_ops.push_back(std::make_pair(op, e + 1));
    } else {
        cigar_ops.push_back(std::make_pair(op, e));
        cigar_ops.push_back(std::make_pair('M', 1));
    }

    // Reverse cigar
    std::vector<std::pair<char, int>> result;
    result.reserve(cigar_ops.size());
    for (auto it = cigar_ops.rbegin(); it != cigar_ops.rend(); ++it) {
        result.push_back(*it);
    }
    
    return result;
}

/**
 * @brief Smith-Waterman alignment using CUDA acceleration
 * 
 * @param ref Reference sequence
 * @param read Read/query sequence
 * @param scoring Scoring parameters
 * @return AlignmentResult Alignment details
 */
AlignmentResult smith_waterman_striped_cuda(
    const std::string& ref, 
    const std::string& read, 
    const ScoringMatrix& scoring) {
    
    AlignmentResult result;
    
    // Convert strings to numeric representation
    std::vector<int8_t> ref_num(ref.length());
    std::vector<int8_t> read_num(read.length());
    
    // Map nucleotides to their numeric values
    static const std::unordered_map<char, int8_t> nucleotide_map = {
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
    std::vector<int8_t> scoring_matrix(n * n, scoring.mismatch);
    
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
    
    // Create profiles in CUDA-compatible format
    auto profile_byte = qP_cuda<int8_t>(read_num, scoring_matrix, "byte");
    auto profile_word = qP_cuda<int16_t>(read_num, scoring_matrix, "word");

    // Get gap penalties from scoring structure
    uint8_t weight_gapO = static_cast<uint8_t>(abs(scoring.gap_open));
    uint8_t weight_gapE = static_cast<uint8_t>(abs(scoring.gap_extend));
    
    // Calculate appropriate bias for byte calculations
    uint8_t bias = 0;
    bias = static_cast<uint8_t>(abs(std::min(scoring.match, scoring.mismatch)));

    // Perform alignment using CUDA
    auto alignment = ssw_main_cuda(
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
        std::cerr << "Warning: Alignment failed between the provided sequences." << std::endl;
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