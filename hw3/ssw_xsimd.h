// ssw_xsimd.h

#ifndef SSW_XSIMD_H
#define SSW_XSIMD_H

#include <string>
#include <vector>
#include <utility>
#include <xsimd/xsimd.hpp>

using namespace std;

// Define scoring matrix index as an enum class for type safety
enum class NucleotideType : int8_t {
    A = 0, C = 1, G = 2, T = 3, N = 4
};

// Structure to hold scoring parameters
struct ScoringMatrix {
    int match;        // Score for a match
    int mismatch;     // Score for a mismatch
    int gap_open;     // Score for opening a gap
    int gap_extend;   // Score for extending a gap
};

// Structure to hold alignment result
struct AlignmentResult {
    int score1;            // Best alignment score
    int score2;            // Second best alignment score
    int ref_begin;         // 0-based position where alignment begins on reference
    int ref_end;           // 0-based position where alignment ends on reference
    int read_begin;        // 0-based position where alignment begins on read
    int read_end;          // 0-based position where alignment ends on read

    // CIGAR operations as pairs of (operation, length)
    // Operations: 'M' for match/mismatch, 'I' for insertion, 'D' for deletion
    vector<pair<char, int>> cigar; // operation: numbers

    // Constructor with defaults
    AlignmentResult() : score1(0), score2(0), ref_begin(-1), ref_end(-1), 
                        read_begin(-1), read_end(-1) {}
};

/**
 * @brief Baseline Smith-Waterman algorithm (non-SIMD)
 * 
 * This implementation provides a standard Smith-Waterman algorithm without
 * any SIMD optimizations for benchmarking or verification purposes.
 * 
 * @param ref Reference sequence
 * @param read Query sequence
 * @param scoring Scoring parameters
 * @return Alignment result containing scores, positions and CIGAR string
 */
AlignmentResult smith_waterman_baseline(
    const string& ref, 
    const string& read, 
    const ScoringMatrix& scoring);

/**
 * @brief SIMD-accelerated Striped Smith-Waterman algorithm
 * 
 * @param ref Reference sequence as string
 * @param read Query sequence as string
 * @param scoring Scoring parameters for the alignment
 * @return Alignment result containing scores, positions and CIGAR string
 */
AlignmentResult smith_waterman_striped_simd(
    const string& ref, 
    const string& read, 
    const ScoringMatrix& scoring);

#endif // SSW_XSIMD_H