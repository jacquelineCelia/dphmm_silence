/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./manager.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef MANAGER_H 
#define MANAGER_H 

#include <cstring>
#include <list>
#include "sampler.h" 
#include "cluster.h"
#include "segment.h"
#include "gmm.h"

using namespace std;

class Manager {
   public:
      Manager();
      bool load_gmm(const string&);
      // bool load_segments(const string&);
      bool load_bounds(const string&, const int);
      bool load_config(const string&);
      void init_sampler();
      void gibbs_sampling(const int, const string);
      // bool load_snapshot(const string&);
      bool load_gmm_from_file(ifstream&, Gmm&, bool);
      bool update_boundaries(const int);
      void update_clusters(const bool, const int);
      void load_data_to_matrix();
      string get_basename(string);
      bool state_snapshot(const string&);
      bool load_snapshot(const string&, const string&, \
        const string&, const int);
      bool load_in_model(const string&, const int);
      bool load_in_model_id(const string&);
      bool load_in_data(const string&, const int); 
      bool load_silence_model(const string&);
      Cluster* find_cluster(const int);
      ~Manager();
   private:
      Sampler sampler;
      list<Segment*> segments;
      vector<Cluster*> clusters;
      vector<Bound*> bounds;
      const float** data;
      Gmm s_gmm; 
      int s_dim;
      int s_state;
      int s_mixture;
      float s_dp_alpha;
      float s_beta_alpha;
      float s_beta_beta;
      float s_gamma_shape;
      float s_norm_kappa;
      float s_gamma_weight_alpha;
      float s_gamma_trans_alpha;
      float s_h0;
      int sil_num_state;
      int sil_num_mixture;
      float sil_self_trans_prob;
      int group_size;
      float new_class_threshold;
      vector<int> batch_groups;
};

#endif
