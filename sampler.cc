/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./sampler.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <cstring>
#include <ctime>                        // define time()
#include <cmath>
#include <mkl_vml.h>

/*
#include "boost/math/distributions/beta.hpp"  // distributions from boost
#include "boost/math/distributions/gamma.hpp" // for gamma_distribution.
#include "boost/math/distributions/normal.hpp" // for normal_distribution.
#include "boost/random/uniform_real.hpp" // for normal_distribution.
#include "boost/random/mersenne_twister.hpp"
*/

#include "sampler.h"
#include "segment.h"
#include "cluster.h"
#include "gmm.h"

#define BRNG VSL_BRNG_MT19937 
#define GAMMA_METHOD VSL_RNG_METHOD_GAMMA_GNORM
#define UNIFORM_METHOD VSL_RNG_METHOD_UNIFORM_STD
#define GAUSSIAN_METHOD VSL_RNG_METHOD_GAUSSIAN_ICDF 
#define SKIP true 

using namespace std;

// using namespace boost::math;
// using namespace boost::random;

float Sampler::annealing = 10.1;
int Sampler::offset = 0;

Sampler::Sampler() {
   UNIT = 0;
   MEAN = 1;
   PRE = 2;
   EMIT = 3;
}

bool Sampler::sample_boundary(Bound* bound) {
   if (bound -> get_utt_end()) {
      return true;
   }
   if (sample_index_from_distribution(boundary_prior)){
      return true;
   }
   return false;
}

bool Sampler::decluster(Segment* ptr, vector<Cluster*>& clusters) {
   vector<Cluster*>::iterator iter_clusters;
   for(iter_clusters = clusters.begin(); \
     iter_clusters != clusters.end(); ++iter_clusters) {
      if (ptr -> get_cluster_id() == (*iter_clusters) -> get_cluster_id()){
         (*iter_clusters) -> remove_members(ptr);
         return true;
      }
   }
   return false;
}

bool Sampler::clean_cluster(Segment* ptr, vector<Cluster*>& clusters) {
   vector<Cluster*>::iterator iter_clusters;
   for(iter_clusters = clusters.begin(); \
     iter_clusters != clusters.end(); ++iter_clusters) {
      if (ptr -> get_cluster_id() == (*iter_clusters) -> get_cluster_id()){
         if ((*iter_clusters)->get_member_num()== 0) {
            delete *iter_clusters;
            clusters.erase(iter_clusters);
            Cluster::counter--;
         }
         return true;
      }
   }
   return false;
}

void Sampler::set_precompute_status(Cluster* model, const bool new_status) {
   model -> set_precompute_status(new_status);
}

bool Sampler::sample_boundary(vector<Bound*>::iterator iter, \
                              list<Segment*>& segments, \
                              vector<Cluster*>& clusters) {
   if (!(*iter) -> get_phn_end()) {
      Segment* parent = (*iter) -> get_parent();
      string tag = parent -> get_tag();
      Segment* h0 = new Segment(*parent);
      // Create segment for h0
      // Create segments for h1
      vector<Bound*> siblings = parent -> get_members();
      vector<Bound*>::iterator iter_siblings;
      iter_siblings = siblings.begin();
      vector<Bound*> h1_l_members;
      for(; (*iter_siblings) != (*iter); ++iter_siblings) {
         h1_l_members.push_back(*iter_siblings);
      }
      h1_l_members.push_back(*iter);
      Segment* h1_l = new Segment(tag, h1_l_members);
      vector<Bound*> h1_r_members;
      ++iter_siblings;
      for(; iter_siblings != siblings.end(); ++iter_siblings) {
         h1_r_members.push_back(*iter_siblings);
      }
      Segment* h1_r = new Segment(tag, h1_r_members);
      if (!decluster(parent, clusters)) {
         cout << "Cannot remove " << parent -> get_tag() 
           << " frame " << parent -> get_start_frame() << " to frame " \
           << parent -> get_end_frame() << endl;
         return false;
      }
      segments.pop_front();
      Segment::counter--;
      delete parent;
      SampleBoundInfo info = sample_h0_h1(h0, h1_l, h1_r, clusters);
      if (info.get_boundary_decision()) {
         is_boundary(h1_l, h1_r, h0, segments, clusters, info, iter);
      }
      else {
         is_not_boundary(h1_l, h1_r, h0, segments, clusters, info, iter);
      }
   }
   else {
      if (!(*iter) -> get_utt_end()) {
         Segment* parent = (*iter) -> get_parent();
         Segment* next_parent = (*(iter + 1)) -> get_parent();
         string tag = parent -> get_tag();
         // Create Segment* h0
         vector<Bound*> h0_members = parent -> get_members();
         vector<Bound*> cousins = next_parent -> get_members();
         vector<Bound*>::iterator iter_cousin;
         iter_cousin = cousins.begin();
         for(; iter_cousin != cousins.end(); ++iter_cousin) {
            h0_members.push_back(*iter_cousin);
         }
         Segment* h0 = new Segment(tag, h0_members);
         // Create Segment* h1_l and hl_r
         Segment* h1_l = new Segment(*parent);
         Segment* h1_r = new Segment(*next_parent);
         if (!decluster(parent, clusters)) {
            cout << "Cannot remove segment..." << endl;
            return false;
         }
         if (!decluster(next_parent, clusters)) {
            cout << "Cannot remove semgnet..." << endl;
            return false;
         }
         segments.pop_front();
         segments.pop_front();
         delete parent;
         delete next_parent;
         Segment::counter -= 2;
         SampleBoundInfo info = sample_h0_h1(h0, h1_l, h1_r, clusters);
         if (info.get_boundary_decision()) {
            is_boundary(h1_l, h1_r, h0, segments, clusters, info, iter);
         }
         else {
            is_not_boundary(h1_l, h1_r, h0, segments, clusters, info, iter);
         }
      }
      else {
         Segment* parent = (*iter) -> get_parent();
         Segment* new_segment = new Segment(*parent);
         if (!decluster(parent, clusters)) {
            cout << "Cannot remove segment..." << endl;
            return false;
         }
         segments.pop_front();
         delete parent;
         --Segment::counter;
         Cluster* new_c;
         if (new_segment -> is_hashed()) {
            for(int i = 0; i < clusters.size(); ++i) {
               if (new_segment -> get_cluster_id() == \
                 clusters[i] -> get_cluster_id()) {
                  new_c = clusters[i];
               }
            }
            encluster(*new_segment, clusters, new_c);
         }
         else {
            if (!clean_cluster(new_segment, clusters)) {
               cout << "Cannot clean clusters..." << endl;
               return false;
            }
            new_c = sample_just_cluster(*new_segment, clusters);
            sample_more_than_cluster(*new_segment, clusters, new_c);
         }
         new_segment -> change_hash_status(false);
         segments.push_back(new_segment);
         ++Segment::counter;
      }
   }
   return true;
}

void Sampler::is_boundary(Segment* h1_l, Segment* h1_r, \
                          Segment* h0, \
                          list<Segment*>& segments, \
                          vector<Cluster*>& clusters, \
                          SampleBoundInfo& info, \
                          vector<Bound*>::iterator iter) {
   sample_more_than_cluster(*h1_r, clusters, info.get_c_h1_r());
   // delete c_h0 if it was a new cluster
   vector<Bound*> members = h1_l -> get_members();
   vector<Bound*>::iterator iter_members = members.begin();
   for(; iter_members != members.end(); ++iter_members) {
      (*iter_members) -> set_parent(h1_l);
   }
   members = h1_r -> get_members();
   iter_members = members.begin();
   for(; iter_members != members.end(); ++iter_members) {
      (*iter_members) -> set_parent(h1_r);
   }
   segments.push_back(h1_l);
   segments.push_front(h1_r);
   Segment::counter += 2;
   (*iter) -> set_phn_end(true);
   h1_l -> change_hash_status(false);
   if (h0 -> is_hashed()) {
      if (!clean_cluster(h0, clusters)) {
         cout << "is_boundary" << endl;
         cout << "Cannot clean clusters..." << endl;
      }
   }
   else if (info.get_c_h0() -> get_cluster_id() == -1) {
      delete info.get_c_h0();
   }
   delete h0;
}

void Sampler::is_not_boundary(Segment* h1_l, Segment* h1_r, \
                             Segment* h0, \
                             list<Segment*>& segments, \
                             vector<Cluster*>& clusters, \
                             SampleBoundInfo& info,
                             vector<Bound*>::iterator iter) {
   if (h0 -> is_hashed()) {
      encluster(*h0, clusters, info.get_c_h0());
   }
   else {
      sample_more_than_cluster(*h0, clusters, info.get_c_h0());
   }
   segments.push_front(h0);
   vector<Bound*> members = h0 -> get_members();
   vector<Bound*>::iterator iter_members = members.begin();
   for(; iter_members != members.end(); ++iter_members) {
      (*iter_members) -> set_parent(h0);
   }
   Segment::counter++;
   (*iter) -> set_phn_end(false);
   if (h1_r -> is_hashed()) {
      if (!clean_cluster(h1_r, clusters)) {
         cout << "is not boundary h1_r" << endl;
         cout << "Cannot clean clusters..." << endl;
      }
   } 
   else if (info.get_c_h1_r() -> get_cluster_id() == -1) {
      delete info.get_c_h1_r();
   }
   if(!decluster(h1_l, clusters)) {
      cout << "Cannot remove segment..." << endl;
   }
   if (!clean_cluster(h1_l, clusters)) {
      cout << "is not boundary" << endl;
      cout << "Cannot clean clusters..." << endl;
   }
   delete h1_l;
   delete h1_r;
}

/*
Cluster* Sampler::sample_from_hash_for_cluster(Segment* data, \
                                   vector<Cluster*>& clusters) {
   vector<double> posteriors = data -> get_hash(); 
   posteriors.pop_back();
   Cluster* new_cluster = sample_cluster_from_base(); 
   double prior, likelihood;
   prior = get_dp_prior(new_cluster);
   likelihood = new_cluster -> compute_likelihood(*data);
   posteriors.push_back(prior + likelihood);
   if (posteriors.size() != clusters.size() + 1) {
      cout << posteriors.size() << " " << clusters.size() << endl;
      cout << data -> get_tag() << " " << data -> get_start_frame() << " " << data -> get_end_frame() << endl;
      cout << "member length " << (data -> get_members()).size() << endl;
      cout << "Size mismatch between cluster and posteriors." << endl;
   }
   int new_c = sample_index_from_log_distribution(posteriors);
   data -> set_hash(posteriors[new_c]);
   data -> change_hash_status(true); 
   if ((unsigned int) new_c != clusters.size()) {
      delete new_cluster;
      return clusters[new_c];
   }
   else {
      return new_cluster;
   }
}
*/

SampleBoundInfo Sampler::sample_h0_h1(Segment* h0, \
                 Segment* h1_l, Segment* h1_r, \
                 vector<Cluster*>& clusters) {
   double boundary_posterior_arr[2];
   SampleBoundInfo info;
   Cluster* c_h0;
   if (h0 -> is_hashed()) {
      for (int i = 0; i < clusters.size(); ++i) {
         if(h0 -> get_cluster_id() == clusters[i] -> get_cluster_id()) {
            c_h0 = clusters[i];
         }
      }
   }
   else {
      if (h0 -> get_cluster_id() != -1) {
         if (!clean_cluster(h0, clusters)) {
            cout << "sample h0 h1 condi 1" << endl;
            cout << "Cannot clean clusters..." << endl;
         }
      }
      c_h0 = sample_just_cluster(*h0, clusters);
      /*
      if (c_h0 -> get_cluster_id() == -1) {
         cout << "c_h0 is new" << endl;
      }
      */
   }
   boundary_posterior_arr[0] = h0 -> get_hash();
   info.set_c_h0(c_h0);
   Cluster* c_h1_l;
   if (h1_l -> is_hashed()) {
      for (int i = 0; i < clusters.size(); ++i) {
         if(h1_l -> get_cluster_id() == clusters[i] -> get_cluster_id()) {
            c_h1_l = clusters[i];
         }
      }
      encluster(*h1_l, clusters, c_h1_l);
   }
   else {
      if (h1_l -> get_cluster_id() != -1 && \
        h1_l -> get_cluster_id() != c_h0 -> get_cluster_id()) {
         if (!clean_cluster(h1_l, clusters)) {
            cout << "sample h0 h1 condi2" << endl;
            cout << "Cannot clean clusters..." << endl;
         }
      }
      c_h1_l = sample_just_cluster(*h1_l, clusters);
      /*
      if (c_h1_l -> get_cluster_id()) {
         cout << "c_h1_l is new" << endl;
      }
      */
      sample_more_than_cluster(*h1_l, clusters, c_h1_l);
   }
   boundary_posterior_arr[1] = h1_l -> get_hash(); 
   info.set_c_h1_l(c_h1_l);
   Cluster* c_h1_r;
   if (h1_r -> is_hashed()) {
      for (int i = 0; i < clusters.size(); ++i) {
         if(h1_r -> get_cluster_id() == clusters[i] -> get_cluster_id()) {
            c_h1_r = clusters[i];
         }
      }
   }
   else {
      if (h1_r -> get_cluster_id() != -1 && \
        h1_r -> get_cluster_id() != c_h0 -> get_cluster_id()) {
         if (!clean_cluster(h1_r, clusters)) {
            cout << "sample h0 h1 condi3" << endl;
            cout << "Cannot clean clusters..." << endl;
         }
      }
      c_h1_r = sample_just_cluster(*h1_r, clusters);
      /*
      if (c_h1_r -> get_cluster_id() == -1) {
         cout << "c_h1_r is new" << endl;
      }
      */
   }
   boundary_posterior_arr[1] += h1_r -> get_hash();
   boundary_posterior_arr[0] += boundary_prior_log[0];
   boundary_posterior_arr[1] += boundary_prior_log[1];
   info.set_c_h1_r(c_h1_r);
   // cout << boundary_posterior_arr[0] << " " << boundary_posterior_arr[1] << endl;
   vector<double> boundary_posterior(boundary_posterior_arr, \
     boundary_posterior_arr + 2);
   // sample the decision
   // for the picked one, add it to the list<Segment*>
   // sample hidden states and mixture id
   // append it to the cluster's member list.
   // delete the unused Segment*
   // set the boundary info to (*iter)
   int boundary_decision = sample_index_from_log_distribution(\
                                              boundary_posterior);
   info.set_boundary_decision(boundary_decision);
   return info;
}

Cluster* Sampler::sample_cluster_from_others(vector<Cluster*>& clusters) {
   Cluster* new_cluster = new Cluster(state_num, mixture_num, dim);
   vector<vector<float> > new_trans;
   vector<vector<float> > pseudo_trans;
   for (int i = 0; i < state_num; ++i) {
      vector<float> pseudo_state_trans;
      for (int j = 0; j < state_num + 1; ++j) {
         pseudo_state_trans.push_back(0.0);
      }
      pseudo_trans.push_back(pseudo_state_trans);
   }
   sample_trans(new_trans, pseudo_trans);
   new_cluster -> update_trans(new_trans);
   double uni_class[clusters.size()];
   double subsurface = 0;
   double log_pro = log(1.0/clusters.size());
   for (int i = 0; i < clusters.size() - 1; ++i) {
      subsurface += 1.0 / clusters.size();
      uni_class[i] = log_pro; 
   }
   uni_class[clusters.size() - 1] = log(1 - subsurface);
   vector<double> posteriors(uni_class, uni_class + clusters.size());
   unsigned int c = sample_index_from_log_distribution(posteriors);
   float random_unit = sample_from_unit();
   float sign = random_unit >= 0.5 ? 1 : -1;
   for (int i = 0; i < state_num; ++i) {
      Gmm new_state(mixture_num, dim);
      for (int j = 0; j < mixture_num; ++j) {
         float random_unit = sample_from_unit();
         float sign = random_unit >= 0.5 ? 1 : -1;
         const float* other_mean = \
                (clusters[c] -> get_emission(i)).get_mixture(j).get_mean();
         const float* other_var = \
                (clusters[c] -> get_emission(i)).get_mixture(j).get_var();
         const float other_weight = \
                (clusters[c] -> get_emission(i)).get_mixture(j).get_weight();
         float avg_mean = 0;
         for (int d = 0; d < dim; ++d) {
            avg_mean += other_mean[d];
         }
         avg_mean /= dim;
         float new_mean[dim];
         float new_var[dim];
         for (int d = 0; d < dim; ++d) {
            new_var[d] = 1.0;
            new_mean[d] = other_mean[d] + annealing * sign * avg_mean;
         }
         new_state.get_mixture(j).update_mean(new_mean);
         new_state.get_mixture(j).update_var(new_var);
         new_state.get_mixture(j).update_det();
         new_state.get_mixture(j).update_weight(other_weight);
      }
      new_cluster -> update_emission(new_state, i);
   }
   return new_cluster;
}

Cluster* Sampler::sample_just_cluster(Segment& data, \
                      vector<Cluster*>& clusters, bool pick_new){
   double prior = 0.0;
   double likelihood = 0.0;
   int num_clusters = clusters.size();
   double posterior_arr[num_clusters + 1];
   // Compute poterior P(c|data, clusters);
   for(int i = 0; i < num_clusters; ++i) {
      prior = get_dp_prior(clusters[i]);
      likelihood = clusters[i] -> compute_likelihood(data, offset);
      posterior_arr[i] = prior + likelihood;
   }
   // Sample one cluster to approximate the infinite integral
   Cluster* new_cluster;
   // new_cluster = sample_cluster_from_base();
   if (clusters.size() == 0) {
      new_cluster = sample_cluster_from_base();
   }
   else {
      if (Sampler::annealing != 10.1) {
         new_cluster = sample_cluster_from_others(clusters);
      }
      else {
         new_cluster = sample_cluster_from_base();
      }
   }
   // new_cluster = sample_cluster_from_base();
   prior = get_dp_prior(new_cluster);
   likelihood = new_cluster -> compute_likelihood(data, offset);
   posterior_arr[num_clusters] = prior + likelihood;
   vector<double> posteriors(posterior_arr, posterior_arr + num_clusters + 1);
   int new_c = pick_new ? (int) clusters.size() : sample_index_from_log_distribution(posteriors);
   data.set_hash(calculator.sum_logs(posterior_arr, num_clusters + 1));
   if ((unsigned int) new_c != clusters.size()) {
      delete new_cluster;
      return clusters[new_c];
   }
   else {
      return new_cluster;
   }
}

Cluster* Sampler::sample_cluster_from_base() {
   Cluster* new_cluster = new Cluster(state_num, mixture_num, dim);
   sample_hmm_parameters(*new_cluster);
   return new_cluster;
}


void Sampler::sample_more_than_cluster(Segment& data, \
                             vector<Cluster*>& clusters, \
                             Cluster* picked_cluster) {
   // Update the cluster id for the data
   // sample hidden_states for the data
   // sample misture id for the data
   // append data to the cluster
   sample_hidden_states(data, picked_cluster);
   /*
   for (int i = 0; i < data.get_frame_num(); ++i) {
      cout << data.get_hidden_states(i) << " ";
   }
   cout << endl;
   */
   sample_mixture_id(data, picked_cluster);
   if (picked_cluster -> get_cluster_id() == -1) {
      picked_cluster -> set_cluster_id();
      clusters.push_back(picked_cluster);
      Cluster::counter++;
   }
   data.set_cluster_id(picked_cluster -> get_cluster_id());
   picked_cluster -> append_member(&data);
   data.change_hash_status(true);
}

void Sampler::encluster(Segment& data, \
                        vector<Cluster*>& clusters, \
                        Cluster* picked_cluster) {
   picked_cluster -> append_member(&data);
}

int Sampler::sample_index_from_distribution(vector<double> posteriors) {
   vector<double>::iterator iter;
   double sum = 0.0;
   int new_c = 0;
   double random_from_unit = -1;
   for(iter = posteriors.begin(); iter != posteriors.end(); ++iter) {
      sum += *iter;
   }
   for(iter = posteriors.begin(); iter != posteriors.end(); ++iter) {
      *iter /= sum; 
   }
   // sample a random number from a uniform dist [0,1]
   random_from_unit = sample_from_unit(); 
   // figure out the new cluster 
   sum = posteriors[new_c];
   while (random_from_unit > sum) {
      sum += posteriors[++new_c];
   }
   return new_c;
}

int Sampler::sample_index_from_log_distribution(vector<double> posteriors) {
   double marginal_max = calculator.find_log_max(posteriors);
   // Normalize the poteriors first
   double sum = 0.0;
   double random_from_unit = -1;
   int new_c = 0;
   vector<double>::iterator iter;
   for(iter = posteriors.begin(); iter != posteriors.end(); ++iter) {
      *iter = exp(*iter - marginal_max);
      sum += *iter;
   }
   for(iter = posteriors.begin(); iter != posteriors.end(); ++iter) {
      *iter /= sum; 
   }
   // sample a random number from a uniform dist [0,1]
   random_from_unit = sample_from_unit(); 
   // figure out the new cluster 
   sum = posteriors[new_c];
   while (random_from_unit > sum) {
      sum += posteriors[++new_c];
   }
   return new_c;
}

float Sampler::sample_from_unit() {
   float* ran_num;
   ran_num = storage.get_random_samples(UNIT);
   while (*ran_num == 1 || *ran_num == 0) {
      ran_num = storage.get_random_samples(UNIT);
   }
   return *ran_num;
}

void Sampler::sample_mixture_id(Segment& data, Cluster* model) {
   const int frame_num = data.get_frame_num();
   int* new_mixture_id = new int[frame_num];
   if (mixture_num == 1) {
      for (int i = 0; i < frame_num; ++i) {
         new_mixture_id[i] = 0;
      }
   }
   else {
      for (int i = 0 ; i < frame_num; ++i) {
         int s_t = data.get_hidden_states(i);
         const float* const frame_i = data.get_frame_i_data(i);
         int frame_i_index = data.get_frame_index(i) - offset;
         vector<double> posterior = \
               model -> compute_gmm_mixture_posterior_weight(\
                 s_t, frame_i, frame_i_index);
         new_mixture_id[i] = sample_index_from_log_distribution(posterior);
      }
   }
   data.set_mixture_id(new_mixture_id);
   delete[] new_mixture_id;
}

void Sampler::sample_pseudo_state_seq(Cluster* model, \
                                      int* state_storage, \
                                      int len) {
   const int frame_num = len; 
   int* new_hidden_states = new int[frame_num];
   new_hidden_states[0] = 0;
   for(int i = 1; i < frame_num - 1; ++i) {
      if (new_hidden_states[i - 1] == state_num - 1) {
         new_hidden_states[i] = new_hidden_states[i - 1];
      }
      else {
         vector<double> prior;
         double state_trans_p = 0.0;
         int s_t_1 = new_hidden_states[i - 1];
         for (int s_t = s_t_1; s_t < state_num; ++s_t) {
            state_trans_p = model -> \
               get_state_trans_prob(s_t_1, s_t);
            prior.push_back(state_trans_p);
         }
         int s = sample_index_from_log_distribution(prior);
         new_hidden_states[i] = s_t_1 + s;
      }
   }
   if (frame_num > 1) {
      new_hidden_states[frame_num - 1] = state_num - 1;
   }
   memcpy(state_storage, new_hidden_states, sizeof(int) * len);
   delete[] new_hidden_states;
}

bool Sampler::hidden_state_valid_check(const int* states, 
  const int frame_num) {
   int last_state_counter = 0;
   for (int i = 0; i < frame_num; ++i) {
      if (states[i] == state_num - 1) {
         ++last_state_counter;
      }
   }
   if (last_state_counter > frame_num / 2 && frame_num >= 12) {
      return true;
   }
   else {
      return false;
   }
}

// the sequence must start with state 0 and end at (state_num - 1)
void Sampler::sample_hidden_states(Segment& data, Cluster* model) {
   const int frame_num = data.get_frame_num();
   int* new_hidden_states = new int[frame_num];
   new_hidden_states[0] = 0;
   // sample hidden states from posterior
   if (data.get_cluster_id() != -1) {
      bool resample = true;
      int trial_num = 0;
      while (resample && trial_num < 5) {
         for(int i = 1; i < frame_num - 1; ++i) {
            if (new_hidden_states[i - 1] == state_num - 1) {
               new_hidden_states[i] = state_num - 1;
            }
            else {
               // every computed item is in log
               vector<double> posteriors;
               double likelihood = 0.0;
               double prior = 0.0;
               int s_t_1 = new_hidden_states[i - 1];
               if (!SKIP) {
                  for(int s_t = s_t_1; s_t <= s_t_1 + 1 && s_t < state_num; \
                  ++s_t) {
                     prior = model -> get_state_trans_prob(s_t_1, s_t);
                     likelihood = model -> \
                        compute_emission_likelihood(\
                          s_t, data.get_frame_i_data(i), \
                           data.get_frame_index(i) - offset);
                     posteriors.push_back(likelihood + prior);
                  }
                  new_hidden_states[i] = \
                     sample_index_from_log_distribution(posteriors) + s_t_1;
               }
               else {
                  for(int s_t = s_t_1; s_t < state_num; ++s_t) {
                     prior = model -> get_state_trans_prob(s_t_1, s_t);
                     likelihood = model -> \
                        compute_emission_likelihood(\
                          s_t, data.get_frame_i_data(i), \
                           data.get_frame_index(i) - offset);
                     posteriors.push_back(likelihood + prior);
                     // posteriors.push_back(likelihood);
                  }
                  new_hidden_states[i] = \
                     sample_index_from_log_distribution(posteriors) + s_t_1;
               }
            }
         }
         ++trial_num;
         // resample = hidden_state_valid_check(new_hidden_states, frame_num);
         resample = false;
      }
   }
   // sample hidden states from prior
   else {
      int base = frame_num / state_num;
      int extra = frame_num % state_num;
      int ptr = 0;
      for (int s = 0; s < state_num; ++s) {
         for (int b = 0; b < base; ++b) {
            new_hidden_states[ptr] = s;
            ++ptr;
         }
         if (s == 1) {
            for (int b = 0; b < extra; ++b) {
               new_hidden_states[ptr] = s;
               ++ptr;
            }
         }
      }
      new_hidden_states[0] = 0;
      /*
      for(int i = 1; i < frame_num - 1; ++i) {
         if (new_hidden_states[i - 1] == state_num - 1) {
            new_hidden_states[i] = new_hidden_states[i - 1];
         }
         else {
            vector<double> prior;
            double state_trans_p = 0.0;
            int s_t_1 = new_hidden_states[i - 1];
            for (int s_t = s_t_1; s_t <= s_t_1 + 1 && s_t < state_num; ++s_t) {
               state_trans_p = model -> \
                  get_state_trans_prob(s_t_1, s_t);
               prior.push_back(state_trans_p);
            }
            int s = sample_index_from_log_distribution(prior);
            new_hidden_states[i] = s_t_1 + s;
         }
      }*/
   }
   if (frame_num > 1) {
      new_hidden_states[frame_num - 1] = state_num - 1;
   }
   data.set_hidden_states(new_hidden_states);
   delete[] new_hidden_states;
}
/*
float Sampler::sample_from_beta(const int* counts) {
   float random_from_unit = sample_from_unit(); 
   beta_distribution<> dist(beta_alpha + counts[0], beta_beta + counts[1]);
   return quantile(dist, random_from_unit); 
}
*/
const float* Sampler::sample_from_gaussian(int index, \
                                      const float* new_var, \
                                      const float* mean_count, \
                                      const float weight_count) {
   const float* prior_mean = gmm.get_mixture(index).get_mean();
   float* new_mean = new float[dim];
   for(int i = 0; i < dim; ++i) {
      // normal_distribution(RealType mean = 0, RealType sd = 1);
      float updated_mean = (norm_kappa * prior_mean[i] + mean_count[i]) \
                            / (norm_kappa + weight_count);
      float updated_kappa = norm_kappa + weight_count;
      float std = sqrt( 1 / (new_var[i] * updated_kappa));
      vsRngGaussian(GAUSSIAN_METHOD, \
        stream, 1, new_mean + i, updated_mean, std);
      /*
      float random_from_unit = sample_from_unit();
      normal_distribution<> dist(updated_mean, std);
      new_mean[i] = quantile(dist, random_from_unit);
      */
   }
   return new_mean;
}
// update gamma_rate, b, according to
// b0 + 1/2 \sum_i^n {(x_i - \bar{x})^2} + \frac{k0n( \bar{x} - u0)^2}{2(k0+n)}
// b0 + 1/2 * \sum_i^n {x_i^2} - 1/2*n*\bar{x}^2 + the last item above 
float Sampler::update_gamma_rate(float b0, \
                                  float s, \
                                  float u, \
                                  float n, \
                                  float u0) {
   float updated_gamma_rate;
   if (!n) {
      return b0;
   }
   updated_gamma_rate = b0 + 0.5 * (s - u * u / n) + \
     (norm_kappa * n * (u/n - u0) * (u/n - u0) / (2*(norm_kappa + n)));
   return updated_gamma_rate;
}

const float* Sampler::sample_from_gamma(int index, \
                                   const float* var_count, \
                                   const float* mean_count, \
                                   const float weight_count ) {
   // prior
   const float* gamma_rate = gmm.get_mixture(index).get_var(); 
   const float* prior_mean = gmm.get_mixture(index).get_mean();
   float* new_var = new float[dim];
   // cout << "sampling for lamda" << endl;
   for (int i = 0; i < dim; ++i) {
      float updated_gamma_rate = \
        update_gamma_rate(gamma_rate[i], var_count[i], \
                          mean_count[i], weight_count, \
                          prior_mean[i]);
      float updated_gamma_shape = gamma_shape + weight_count/2;
      vsRngGamma(GAMMA_METHOD, stream, 1, new_var + i, \
        updated_gamma_shape, 0, 1 / updated_gamma_rate);
      /*
      float random_from_unit = sample_from_unit();
      gamma_distribution<> dist(updated_gamma_shape, 1 / updated_gamma_rate);
      new_var[i] = quantile(dist, random_from_unit);
      */
      // To store precision instead of var
      // new_var[i] = 1/new_var[i];
   }
   return new_var;
}

const float* Sampler::sample_from_gamma_for_multidim(const float* count, \
                                                     int multidim, \
                                                     float* prior) {
   float* portion = new float[multidim];
   float total = 0.0;
   // cout << "sampling for weight" << endl;
   for (int i = 0; i < multidim; ++i) {
      vsRngGamma(GAMMA_METHOD, stream, 1, portion + i, \
        prior[i] + count[i], 0, 1);
      total += portion[i]; 
   }
   for (int i = 0 ; i < multidim; ++i) {
      portion[i] /= total;
   }
   return portion;
}

/*
const float* Sampler::sample_from_gamma_for_weight(const float* count) {
   int mixture_num = gmm.get_mixture_num();
   float* weight = new float[mixture_num];
   float total = 0.0;
   // cout << "sampling for weight" << endl;
   for (int i = 0; i < mixture_num; ++i) {
      gamma_distribution<> dist(gamma_weight_alpha + count[i], 1);
      float random_from_unit = sample_from_unit();
      weight[i] = quantile(dist, random_from_unit);
      total += weight[i]; 
   }
   for (int i = 0 ; i < mixture_num; ++i) {
      weight[i] /= total;
   }
   return weight;
}
*/

void Sampler::sample_trans(vector<vector<float> >& trans, \
                           vector<vector<float> >& pseudo_trans) {
   if (!SKIP) {
      for(int i = 0; i < state_num; ++i) {
         int num_to_states = 2; 
         float counts[num_to_states];
         for (int j = 0; j < num_to_states; ++j) {
            counts[j] = pseudo_trans[i][i + j];
         }
         float trans_prior[num_to_states];
         for (int j = 0; j < num_to_states; ++j) {
            trans_prior[j] = gamma_trans_alpha;
         }
         if (i == 1) {
            trans_prior[0] = 5 * gamma_trans_alpha;
         }
         const float* new_trans = sample_from_gamma_for_multidim( \
            counts, num_to_states, trans_prior);
         vector<float> new_state_trans(new_trans, new_trans + num_to_states);
         for (int j = 0; j < num_to_states; ++j) {
            new_state_trans[j] = log(new_state_trans[j]);
         }
         delete[] new_trans;
         vector<float>::iterator iter;
         for (int j = 0; j < i; ++j) {
            iter = new_state_trans.begin();
            new_state_trans.insert(iter, 0.0); 
         }
         if (i != state_num - 1 && i != 0) {
            new_state_trans.push_back(0.0);
         }
         trans.push_back(new_state_trans);
      }
   }
   else {
      for(int i = 0; i < state_num; ++i) {
         int num_to_states = state_num - i;
         if (i == state_num - 1) {
            ++num_to_states;
         }
         if (i == 0) {
            ++num_to_states;
         }
         float counts[num_to_states];
         for (int j = 0; j < num_to_states; ++j) {
            counts[j] = pseudo_trans[i][i + j];
         }
         float trans_prior[num_to_states];
         for (int j = 0; j < num_to_states; ++j) {
            trans_prior[j] = gamma_trans_alpha;
         }
         if (i == 0) {
            trans_prior[0] = 3 * gamma_trans_alpha;
            trans_prior[1] = 3 * gamma_trans_alpha;
         }
         else if (i == 1) {
            trans_prior[0] = 3 * gamma_trans_alpha;
         }
         const float* new_trans = sample_from_gamma_for_multidim( \
            counts, num_to_states, trans_prior);
         vector<float> new_state_trans(new_trans, new_trans + num_to_states);
         for (int j = 0; j < num_to_states; ++j) {
            new_state_trans[j] = log(new_state_trans[j]);
         }
         delete[] new_trans;
         vector<float>::iterator iter;
         for (int j = 0; j < i; ++j) {
            iter = new_state_trans.begin();
            new_state_trans.insert(iter, 0.0); 
         }
         if (i != state_num - 1 && i != 0) {
            new_state_trans.push_back(0.0);
         }
         trans.push_back(new_state_trans);
      }
   }
}

void Sampler::sample_hmm_parameters(Cluster& model) {
   vector<vector<float> > new_trans;
   vector<vector<float> > pseudo_trans;
   for (int i = 0; i < state_num; ++i) {
      vector<float> pseudo_state_trans;
      for (int j = 0; j < state_num + 1; ++j) {
         pseudo_state_trans.push_back(0.0);
      }
      pseudo_trans.push_back(pseudo_state_trans);
   }
   // sample from prior
   if (model.get_cluster_id() == -1) {
      // sample the trans matrix
      sample_trans(new_trans, pseudo_trans);
      // sample the gmms
      float pseudo_vector[dim];
      float pseudo_count[model.get_mixture_num()];
      for (int i = 0; i < dim; ++i) {
         pseudo_vector[i] = 0.0;
      }
      for (int i = 0; i < model.get_mixture_num(); ++i) {
         pseudo_count[i] = 0.0;
      }
      for(int i = 0; i < state_num; ++i) {
         Gmm new_gmm;
         new_gmm.init(model.get_mixture_num(), model.get_dim());
         // sample the mixture
         // sample the weight 
         float weight_prior[model.get_mixture_num()];
         for (int w = 0; w < model.get_mixture_num(); ++w) {
            weight_prior[w] = gamma_weight_alpha;
         }
         const float* weight = sample_from_gamma_for_multidim( \
           pseudo_count, model.get_mixture_num(), weight_prior);
         for(int j = 0; j < model.get_mixture_num(); ++j) {
            new_gmm.get_mixture(j).update_weight(log(weight[j]));
            // sample the var vector
            const float* new_var = storage.get_random_samples(PRE); 
            // sample the mean vector
            const float* new_mean = storage.get_random_samples(MEAN); 
            new_gmm.get_mixture(j).update_mean(new_mean);
            new_gmm.get_mixture(j).update_var(new_var);
            new_gmm.get_mixture(j).update_det();
            delete[] new_var;
            delete[] new_mean;
         }
         delete[] weight;
         model.update_emission(new_gmm, i);
      }
   }
   // sample from poterior
   else {
      // pseudo_trans can be used to store counts of transition
      sample_trans(new_trans, model.get_cache_trans());
      Gmm gmm_true_var[state_num];
      for (int i = 0; i < state_num; ++i) {
         gmm_true_var[i].init(model.get_mixture_num(), model.get_dim());
      }
      if (model.get_age() > 0) {
         for (int i = 0; i < state_num; ++i) {
            for (int j = 0; j < model.get_mixture_num(); ++j) {
               float new_mean[model.get_dim()];
               float new_var[model.get_dim()];
               float n = model.get_cache_weight(i, j);
               memcpy(new_var, model.get_cache_var(i, j), \
                 sizeof(float) * model.get_dim());
               memcpy(new_mean, model.get_cache_mean(i, j), \
                  sizeof(float) * model.get_dim());
               if (n > 50) {
                  for (int d = 0; d < model.get_dim(); ++d) {
                     new_mean[d] = new_mean[d] / n;
                     new_var[d] = new_var[d] - n * new_mean[d] * new_mean[d]; 
                     new_var[d] = (n - 1) / new_var[d];
                  }
               }
               gmm_true_var[i].get_mixture(j).update_mean(new_mean);
               gmm_true_var[i].get_mixture(j).update_var(new_var);
            }
         }
      }
      for(int i = 0; i < state_num; ++i) { 
         Gmm new_gmm;
         new_gmm.init(model.get_mixture_num(), model.get_dim());
         float* weight_count = new float[model.get_mixture_num()];
         for (int j = 0 ; j < model.get_mixture_num(); ++j) {
            weight_count[j] = model.get_cache_weight(i, j);
            const float* mean_count = model.get_cache_mean(i, j); 
            const float* new_var;
            if (model.get_age() > 0 && weight_count[j] > 50) {
               new_var = gmm_true_var[i].get_mixture(j).get_var();
               new_gmm.get_mixture(j).update_var(new_var);
            }
            else {
               const float* var_count = model.get_cache_var(i, j); 
               new_var = sample_from_gamma( \
                  j, var_count, mean_count, weight_count[j]);
               new_gmm.get_mixture(j).update_var(new_var);
               delete[] new_var;
            }
            const float* new_mean = sample_from_gaussian( \
              j, new_gmm.get_mixture(j).get_var(), \
              mean_count, weight_count[j]);
            // update mean and variance
            new_gmm.get_mixture(j).update_mean(new_mean);
            new_gmm.get_mixture(j).update_det();
            delete[] new_mean;
         }
         float weight_prior[model.get_mixture_num()];
         for (int w = 0; w < model.get_mixture_num(); ++w) {
            weight_prior[w] = gamma_weight_alpha;
         }
         const float* new_weight = sample_from_gamma_for_multidim( \
                     weight_count, model.get_mixture_num(), weight_prior);
         for (int j = 0; j < model.get_mixture_num(); ++j) {
            new_gmm.get_mixture(j).update_weight(log(new_weight[j]));
         }
         delete[] new_weight;
         delete[] weight_count;
         model.update_emission(new_gmm, i);
      }
   }
   model.update_trans(new_trans);
}

void Sampler::precompute(Cluster* model, int total, const float** data) {
   model -> precompute(total, data);
}

double Sampler::get_dp_prior(Cluster* model) const {
   int member_num = model->get_member_num();
   int data_num = Segment::counter;
   if (model-> get_cluster_id() == -1) {
      return log((dp_alpha / (data_num - 1 + dp_alpha)));
   }
   else {
      if (member_num == 0) {
         return -300;
      }
      return log((member_num / (data_num - 1 + dp_alpha)));
   }
}

void Sampler::init_prior(const int s_dim, \
                         const int s_state, \
                         const int s_mixture, \
                         const float s_dp_alpha, \
                         const float s_beta_alpha, \
                         const float s_beta_beta, \
                         const float s_gamma_shape, \
                         const float s_norm_kappa, \
                         const float s_gamma_weight_shape, \
                         const float s_gamma_trans_alpha, \
                         Gmm s_gmm, \
                         const float s_h0) {
   dim = s_dim;
   state_num = s_state;
   mixture_num = s_mixture;
   dp_alpha = s_dp_alpha;
   beta_alpha = s_beta_alpha;
   beta_beta = s_beta_beta;
   gmm.init(mixture_num, dim);
   gmm = s_gmm;
   // prior for boundaries
   boundary_prior.push_back(s_h0);
   float s_h1 = 1 - s_h0;
   boundary_prior.push_back(s_h1);
   boundary_prior_log.push_back(log(s_h0));
   boundary_prior_log.push_back(log(s_h1));
   gamma_shape = s_gamma_shape;
   gamma_weight_alpha = s_gamma_weight_shape;
   gamma_trans_alpha = s_gamma_trans_alpha;
   norm_kappa = s_norm_kappa;
   for (int i = 0; i < gmm.get_mixture_num(); ++i) {
      const float* prior_var = gmm.get_mixture(i).get_var();
      float* gamma_rate = new float[dim];
      for (int j = 0; j < dim; ++j) {
         gamma_rate[j] = gamma_shape / prior_var[j];
      }
      gmm.get_mixture(i).update_var(gamma_rate);
      delete[] gamma_rate;
   }
   mixture_num = gmm.get_mixture_num(); 
   // generator.seed(static_cast<unsigned int>(time(0)));  
   unsigned int SEED = time(0);
   vslNewStream(&stream, BRNG,  SEED);
   storage.init(dim, gamma_shape, norm_kappa, 1, gmm.get_mixture(0)); 
}

Sampler::~Sampler() {
   vslDeleteStream(&stream);
}
