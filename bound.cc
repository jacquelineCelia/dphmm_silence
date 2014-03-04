/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./bound.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <cstring>
#include "bound.h"

using namespace std;

int Bound::index_counter = 0;
int Bound::total_frames = 0;

Bound::Bound(const Bound& source) {
   index = source.get_index();
   frame_num = source.get_frame_num();
   start_frame = source.get_start_frame();
   end_frame = source.get_end_frame();
   frame_num = source.get_frame_num();
   dim = source.get_dim();
   utt_end = source.get_utt_end();
   phn_end = source.get_phn_end();
   data = new float* [frame_num];
   for(int i = 0; i < frame_num; ++i) {
      data[i] = new float[dim];
      const float* ptr = source.get_frame_i(i);
      memcpy(data[i], ptr, sizeof(float) * dim);
   }
   parent = source.get_parent();
}

const Bound& Bound::operator= (const Bound& source) {
   if (&source == this) {
      return *this;
   }
   else {
      for(int i = 0; i < frame_num; ++i) {
         delete[] data[i];
      }
      delete[] data;
   }
   index = source.get_index();
   frame_num = source.get_frame_num();
   start_frame = source.get_start_frame();
   end_frame = source.get_end_frame();
   dim = source.get_dim();
   utt_end = source.get_utt_end();
   phn_end = source.get_phn_end();
   parent = source.get_parent();
   
   data = new float* [frame_num];
   for(int i = 0; i < frame_num; ++i) {
      data[i] = new float[dim];
      const float* ptr = source.get_frame_i(i);
      memcpy(data[i], ptr, sizeof(float) * dim);
   }
   return *this;
}

// Initialize a bound
// give start_frame, end_frame, dim
Bound::Bound(int start, int end, int d, \
             bool s_utt_end) {
   start_frame = start;
   end_frame = end;
   dim = d;
   frame_num = end - start + 1;
   utt_end = s_utt_end;
   phn_end = false;
   data = new float*[frame_num];
   for(int i = 0; i < frame_num; ++i) {
      data[i] = new float[dim];
   }
}

void Bound::set_phn_end(bool s) {
   phn_end = s;
}

void Bound::set_utt_end(bool s) {
   utt_end = s;
}

void Bound::set_data(float** source) { 
   for(int i = 0; i < frame_num; ++i) {
      memcpy(data[i], source[i], sizeof(float) * dim);
   }
}

void Bound::set_index(int id) {
   index = id;
}

void Bound::set_parent(Segment* s_parent) {
   parent = s_parent;
}

void Bound::set_start_frame_index(const int index) {
   start_frame_index = index;
}

void Bound::show_data() {
   for (int i = 0; i < frame_num; ++i) {
      for (int j = 0; j < dim; ++j) {
         cout << "frame " << i << " dim " << j << " is " << data[i][j] << ' ' ;
      }
      cout << endl;
   }
}

Bound::~Bound() {
   for(int i = 0; i < frame_num; ++i) {
      delete[] data[i];
   }
   delete[] data;
}
