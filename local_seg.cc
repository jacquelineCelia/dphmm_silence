/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./local_seg.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>

#include "local_seg.h"
using namespace std;

Local_seg::Local_seg(string fin, string fout) {
   infile_name = fin;
   outfile_name = fout;
   frame_num = 0;
}

vector<string> Local_seg::string_split(string line) {
   vector<string> tokens;
   size_t pointer;
   pointer = line.find_first_of(' ');
   string sub_line = line.substr(pointer + 1);
   pointer = sub_line.find_first_of(' ');
   while (pointer != string::npos) {
      tokens.push_back(sub_line.substr(0, pointer));
      sub_line = sub_line.substr(pointer + 1); 
      pointer = sub_line.find_first_of(' ');
   }
   return tokens;
}

void Local_seg::allocate_align_scores() {
   align_scores = new double[frame_num - 1];
}

bool Local_seg::load() {
   ifstream fin;
   string line;
   fin.open(infile_name.c_str());
   if(fin.is_open()) {
      while (fin.good()) {
         getline(fin, line);
         if (line != "") {
            frame_num++;
            vector<string> tokens;
            tokens = string_split(line); 
            vector<string>::iterator iter;
            vector<int> index_list;
            vector<double> feature_frame(50);
            double length = 0.0;
            for (iter = tokens.begin(); iter != tokens.end(); ++iter) {
               string token = *iter;
               size_t pos = token.find(':');
               int index = atoi(token.substr(0, pos).c_str());
               index_list.push_back(index);
               double prob = atof(token.substr(pos + 1).c_str());
               feature_frame[index] = prob; 
               length += prob * prob;
            }
            feature_length.push_back(sqrt(length));
            non_zero_index.push_back(index_list);
            features.push_back(feature_frame);
         }
      }
      allocate_align_scores();
      fin.close();
      return true;
   }
   else {
      return false;
   }
}

void Local_seg::self_align() {
   for(int i = 0; i < frame_num - 1; ++i) {
      vector<int>::iterator iter;
      double score = 0.0;
      for (iter = non_zero_index[i].begin(); iter != non_zero_index[i].end(); ++iter) {
         score += features[i][*iter] * features[i+1][*iter];
      }
      score /= (feature_length[i] * feature_length[i+1]);
      align_scores[i] = score;
   }
}

void Local_seg::output_local_min() {
   ofstream fout(outfile_name.c_str(), ios::out);

   double thre = 0.0;
   for(int i = 0; i < frame_num - 1 ; ++i) {
      if (i == 0) {
         if (align_scores[i] < align_scores[i+1] - thre) {
            fout << '1' << endl;
         }
         else {
            fout << '0' << endl;
         }
      }
      else {
         if (align_scores[i - 1] >= align_scores[i] && \
             align_scores[i] < align_scores[i+1] ) {
            if (abs(align_scores[i - 1] - align_scores[i]) > thre || \
                abs(align_scores[i] - align_scores[i+1]) > thre) {
               fout << '1' << endl;
            }
            else {
               fout << '0' << endl;
            }
         }
         else {
            fout << '0' << endl;
         }
      }
   }
}

Local_seg::~Local_seg() {
   delete[] align_scores;
}
