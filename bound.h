/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./bound.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef BOUND_H
#define BOUND_H

#include "segment.h"

class Segment;
class Bound {
   public:
      static int index_counter;
      static int total_frames;
      Bound(int, int, int, bool);
      Bound(const Bound&);
      const Bound& operator= (const Bound&);
      void set_data(float**);
      void set_index(int);
      void set_parent(Segment*);
      void set_phn_end(bool);
      void set_utt_end(bool);
      void set_start_frame_index(const int index);
      int get_frame_num() const {return frame_num;}
      int get_dim() const {return dim;}
      int get_index() const {return index;}
      int get_start_frame() const {return start_frame;}
      int get_end_frame() const {return end_frame;}
      int get_start_frame_index() const {return start_frame_index;}
      bool get_utt_end() const {return utt_end;}
      bool get_phn_end() const {return phn_end;}
      const float* get_frame_i(int i) const {return data[i];}
      Segment* get_parent() const {return parent;}
      void show_data(); 
      ~Bound();
   private:
      int index;
      float** data;
      int frame_num;
      int start_frame;
      int start_frame_index;
      int end_frame;
      int dim;
      bool utt_end;
      bool phn_end;
      Segment* parent;
};

#endif
