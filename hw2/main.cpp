// main.cpp
// Striped Smith-Waterman implementation using XSIMD
// Based on the original SSW library (Zhao et al.)

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "ssw_xsimd.h"

using namespace std;

// Simple FASTA parser
struct FastaSequence {
    string name;
    string sequence;
};

vector<FastaSequence> read_fasta(const string& filename) {
    vector<FastaSequence> sequences;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return sequences;
    }
    
    string line;
    FastaSequence current;
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        if (line[0] == '>') {
            // If we've been collecting a sequence, save it
            if (!current.name.empty() && !current.sequence.empty()) {
                sequences.push_back(current);
            }
            
            // Start a new sequence
            current.name = line.substr(1);
            // Remove any whitespace from the name
            current.name = current.name.substr(0, current.name.find_first_of(" \t"));
            current.sequence.clear();
        } else {
            // Add line to current sequence
            current.sequence += line;
        }
    }
    
    // Add the last sequence if there is one
    if (!current.name.empty() && !current.sequence.empty()) {
        sequences.push_back(current);
    }
    // if (current.name == "6:163296599:F:198;None;None/1") cout << "[Debug] Query: " << current.sequence << "\n" << endl;
    
    file.close();
    return sequences;
}

void print_alignment(const AlignmentResult& result, 
                    const string& ref_name, 
                    const string& read_name, 
                    const string& ref_seq, 
                    const string& read_seq,
                    double baseline_time,
                    double simd_time) {
    
    // Print alignment info
    cout << "target_name: " << ref_name << endl;
    cout << "query_name: " << read_name << endl;
    
    // Print timing information
    cout << "Baseline time: " << fixed << setprecision(1) << baseline_time << " ms" << endl;
    cout << "SIMD time: " << fixed << setprecision(1) << simd_time << " ms" << endl;
    cout << "Speedup: " << fixed << setprecision(1) << (baseline_time / simd_time) << "X" << endl;
    
    cout << "optimal_alignment_score: " << result.score1;
    if (result.score2 > 0) {
        cout << "    suboptimal_alignment_score: " << result.score2;
    }
    cout << "    strand: +"; // forward aligned
    
    if (result.ref_begin >= 0) {
        cout << "    target_begin: " << result.ref_begin + 1;
    }
    cout << "    target_end: " << result.ref_end + 1;
    
    if (result.read_begin >= 0) {
        cout << "    query_begin: " << result.read_begin + 1;
    }
    cout << "    query_end: " << result.read_end + 1 << endl;
    
    // If there's a CIGAR string, print the alignment
    if (!result.cigar.empty()) {
        string ref_aligned, match_line, read_aligned;
        int ref_pos = result.ref_begin;
        int read_pos = result.read_begin;
        
        for (auto cigar_op : result.cigar) {
            char op = cigar_op.first;
            int length = cigar_op.second;
            
            for (int i = 0; i < length; i++) {
                if (op == 'M') {  // Match or mismatch
                    ref_aligned += ref_seq[ref_pos];
                    read_aligned += read_seq[read_pos];
                    
                    if (ref_seq[ref_pos] == read_seq[read_pos]) {
                        match_line += '|';  // Match
                    } else {
                        match_line += '*';  // Mismatch
                    }
                    
                    ref_pos++;
                    read_pos++;
                } else if (op == 'I') {  // Insertion to reference
                    ref_aligned += '-';
                    read_aligned += read_seq[read_pos];
                    match_line += ' ';
                    read_pos++;
                } else if (op == 'D') {  // Deletion from reference
                    ref_aligned += ref_seq[ref_pos];
                    read_aligned += '-';
                    match_line += ' ';
                    ref_pos++;
                }
            }
        }
        
        // Print the alignment
        cout << "\nSeq1(Target): " << setw(8) << (result.ref_begin + 1) << "    "
                  << ref_aligned << "    " << setw(7) << (result.ref_end + 1) << endl;  // ATCGs
        cout << "                          " << match_line << endl;                     // "|", "*", " "s
        cout << "Seq2(Query): " << setw(9) << (result.read_begin + 1) << "    "
                  << read_aligned << "    " << setw(7) << (result.read_end + 1) << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <target.fasta> <query.fasta>" << endl;
        return 1;
    }

    cout << "XSIMD info: " << endl;
    #if XSIMD_WITH_AVX512F
    cout << "AVX-512F available\n";
    cout << "Register size: 512 bits (64 bytes)\n";
    #elif XSIMD_WITH_AVX2
    cout << "AVX2 available\n";
    cout << "Register size: 256 bits (32 bytes)\n";
    #elif XSIMD_WITH_SSE2
    cout << "SSE2 available\n";
    cout << "Register size: 128 bits (16 bytes)\n";
    #elif XSIMD_WITH_NEON
    cout << "NEON available\n"; 
    cout << "Register size: 128 bits (16 bytes)\n";
    #else
    cout << "No SIMD instruction set available\n";
    #endif

    // Default parameters
    int match = 2;
    int mismatch = 2;
    int gap_open = 3;
    int gap_extend = 1;
    
    // Read the FASTA files
    vector<FastaSequence> target_sequences = read_fasta(argv[1]);
    if (target_sequences.empty()) {
        cerr << "Failed to read target sequence from " << argv[1] << endl;
        return 1;
    }
    
    vector<FastaSequence> query_sequences = read_fasta(argv[2]);
    if (query_sequences.empty()) {
        cerr << "Failed to read query sequence from " << argv[2] << endl;
        return 1;
    }
    
    // Use the first sequence from each file
    FastaSequence& target_seq = target_sequences[0];
    FastaSequence& query_seq = query_sequences[0];
    
    // Initialize scoring matrix for nucleotides
    ScoringMatrix scoring_matrix;
    scoring_matrix.match = match;
    scoring_matrix.mismatch = -mismatch;
    scoring_matrix.gap_open = -gap_open;
    scoring_matrix.gap_extend = -gap_extend;
    
    // Run baseline (non-SIMD) alignment
    auto start_baseline = chrono::high_resolution_clock::now();
    AlignmentResult baseline_result = smith_waterman_baseline(
        target_seq.sequence, query_seq.sequence, scoring_matrix);
    auto end_baseline = chrono::high_resolution_clock::now();
    double baseline_time = chrono::duration<double, milli>(
        end_baseline - start_baseline).count();
    
    // Run SIMD-accelerated alignment
    auto start_simd = chrono::high_resolution_clock::now();
    AlignmentResult simd_result = smith_waterman_striped_simd(
        target_seq.sequence, query_seq.sequence, scoring_matrix);
    auto end_simd = chrono::high_resolution_clock::now();
    double simd_time = chrono::duration<double, milli>(
        end_simd - start_simd).count();
    
    // Print the alignment result
    cout << "\n--------XSIMD result--------" << endl;
    print_alignment(simd_result, target_seq.name, query_seq.name,
                   target_seq.sequence, query_seq.sequence, baseline_time, simd_time);

    // Print the alignment result
    // cout << "\n--------baseline result--------" << endl;
    // print_alignment(baseline_result, target_seq.name, query_seq.name,
    //                target_seq.sequence, query_seq.sequence, baseline_time, simd_time);
    
    return 0;
}