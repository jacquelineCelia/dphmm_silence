/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./segment.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <cstring>
#include "segment.h"

using namespace std;

int Segment::counter = 0;

// initialize a segment by assigning default values to
// frame_num
// cluster_id
// dimension
// To establish memory space for frame_data and hidden_states

Segment::Segment(string tag_name, vector<Bound*> mem) {
   cluster_id = -1;
   tag = tag_name;
   members = mem;
   set_frame_data();
   set_frame_num();
   set_start_frame();
   set_start_frame_index();
   set_end_frame();
   dimension = members[0] -> get_dim();
   hidden_states = new int[frame_num];
   mixture_id = new int[frame_num];
   hashed = false;
}

const Segment& Segment::operator= (const Segment& source) {
   if (this == &source) {
      return *this;
   }
   else {
      delete[] hidden_states;
      delete[] mixture_id;
   }
   tag = source.get_tag();
   members = source.get_members();
   set_member_parent();
   start_frame = source.get_start_frame();
   set_start_frame_index(); 
   end_frame = source.get_end_frame();
   cluster_id = source.get_cluster_id();
   frame_num = source.get_frame_num();
   dimension = source.get_dimension();
   mixture_id = new int [frame_num];
   hidden_states = new int [frame_num];
   frame_data = source.get_frame_data(); 
   memcpy(hidden_states, source.get_hidden_states_all(), \
     sizeof(int) * frame_num);
   memcpy(mixture_id, source.get_mixture_id_all(), \
     sizeof(int) * frame_num);
   hashed = source.is_hashed();
   hash_cluster_post = source.get_hash();
   return *this;
}

Segment::Segment(const Segment& source) {
   tag = source.get_tag();
   start_frame = source.get_start_frame();
   end_frame = source.get_end_frame();
   cluster_id = source.get_cluster_id();
   frame_num = source.get_frame_num();
   dimension = source.get_dimension();
   frame_data = source.get_frame_data(); 
   members = source.get_members();
   set_start_frame_index();
   set_member_parent();
   mixture_id = new int [frame_num];
   hidden_states = new int [frame_num];
   memcpy(hidden_states, source.get_hidden_states_all(), \
     sizeof(int) * frame_num);
   memcpy(mixture_id, source.get_mixture_id_all(), \
     sizeof(int) * frame_num);
   hashed = source.is_hashed();
   hash_cluster_post = source.get_hash(); 
}

void Segment::change_hash_status(bool new_status){
   hashed = new_status;
}

void Segment::set_hash(const double new_hash_cluster_post) {
   hash_cluster_post = new_hash_cluster_post;
}

void Segment::set_member_parent() {
   vector<Bound*>::iterator iter_memebers;
   iter_memebers = members.begin();
   for(; iter_memebers != members.end(); ++iter_memebers) {
      (*iter_memebers) -> set_parent(this);
   }
}

void Segment::set_start_frame() {
   vector<Bound*>::iterator iter;
   iter = members.begin();
   start_frame = (*iter) -> get_start_frame(); 
}

void Segment::set_end_frame() {
   vector<Bound*>::iterator iter;
   iter = --members.end();
   end_frame = (*iter) -> get_end_frame(); 
}

// To set frame_num using len read from a file
void Segment::set_frame_num() {
   frame_num = 0;
   vector<Bound*>::iterator iter;
   for(iter = members.begin(); iter != members.end(); ++iter) {
      frame_num += (*iter) -> get_frame_num();
   }
}

// To get frame_num
int Segment::get_frame_num() const {
   return frame_num;
}

void Segment::set_start_frame_index() {
   start_frame_index = members[0] -> get_start_frame_index(); 
}

// To set feature values
void Segment::set_frame_data() {
   vector<Bound*>::iterator iter;
   for(iter = members.begin(); iter != members.end(); ++iter) {
      int mem_len = (*iter) -> get_frame_num();
      for (int i = 0; i < mem_len; ++i) {
         frame_data.push_back((*iter) -> get_frame_i(i));
      }
   }
}

// get frame_i
const float* Segment::get_frame_i_data(int index) const {
   return frame_data[index];
}

// To set hidden_states
void Segment::set_hidden_states(int* source) {
   memcpy(hidden_states, source, sizeof(int) * frame_num);
}

// To get hidden_states
int Segment::get_hidden_states(int i) const {
   return hidden_states[i];
}

// To set cluster id
void Segment::set_cluster_id(int id) {
   cluster_id = id;
}

// To get cluster id
int Segment::get_cluster_id() const {
   return cluster_id;
}

// To set mixture_id
void Segment::set_mixture_id(const int* source) {
   memcpy(mixture_id, source, sizeof(int) * frame_num);
}

// To get mixture_id for frame i
int Segment::get_mixture_id(int id) const {
   return mixture_id[id];
}

void Segment::show_data() {
   for (int i = 0; i < frame_num; ++i) {
      for (int j = 0; j < dimension; ++j) {
         cout << "frame " << i << " dim " << j << " is " << frame_data[i][j] << ' ' ;
      }
      cout << endl;
   }
}

void Segment::write_class_label(const string& dir) {
   string path = dir + "/" + tag + ".algn";
   ofstream fout(path.c_str(), ios::app);
   fout << start_frame << " " << end_frame << " " << cluster_id << endl;
   fout.close();
}

// Free memories allocated for this object
Segment::~Segment() {
   delete[] hidden_states;
   delete[] mixture_id;
}
