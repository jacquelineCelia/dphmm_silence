/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./segment.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef SEGMENT_H
#define SEGMENT_H

#include <string>
#include <vector>
#include "bound.h"

using namespace std;
class Bound;
class Segment {
   public:
      Segment(const Segment&);
      Segment(string, vector<Bound*>);
      const Segment& operator= (const Segment&);
      static int counter;
      void set_frame_num();
      void set_frame_data();
      void set_hidden_states(int*);
      void set_mixture_id(const int*);
      void set_cluster_id(int);
      void set_start_frame();
      void set_end_frame();
      void set_start_frame_index();
      void show_data();
      void set_member_parent();
      void write_class_label(const string&);
      vector<const float*> get_frame_data() const {return frame_data;}
      int get_frame_num() const;
      const float* get_frame_i_data(int) const;
      int get_hidden_states(int) const;
      int get_mixture_id(int) const;
      int get_cluster_id() const;
      int get_frame_index(const int offset) const \
                   {return (start_frame_index + offset);}
      string get_tag() const {return tag;}
      int get_start_frame() const {return start_frame;}
      int get_end_frame() const {return end_frame;}
      int get_dimension() const {return dimension;}
      vector<Bound*> get_members() const {return members;}
      const int* get_hidden_states_all() const {return hidden_states;}
      const int* get_mixture_id_all() const {return mixture_id;}
      int get_segment_num() const {return counter;}
      bool is_hashed() const {return hashed;}
      void change_hash_status(bool);
      void set_hash(const double);
      double get_hash() const {return hash_cluster_post;} 
      ~Segment();
   private:
      string tag;
      int start_frame;
      int start_frame_index;
      int end_frame;
      int cluster_id;
      int frame_num;
      int* hidden_states;
      int* mixture_id;
      int dimension;
      vector<const float*> frame_data;
      vector<Bound*> members;
      bool hashed;
      // store cluster posterior
      double hash_cluster_post;
};

#endif
