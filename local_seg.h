/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./local_seg.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef LOCAL_SEG_H
#define LOCAL_SEG_H

#include <vector>
#include <string>

using namespace std;

class Local_seg {
 public:
    Local_seg(string, string);
    bool load();
    ~Local_seg();
    void self_align();
    void output_local_min();
 private:
    int frame_num;
    vector < vector<int> > non_zero_index;
    vector < vector<double> > features;
    vector <double> feature_length;
    double* align_scores;
    string infile_name;
    string outfile_name;

    // utility functions
    vector<string> string_split(string);
    void allocate_align_scores();
};

#endif
