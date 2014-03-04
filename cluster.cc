/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./cluster.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/

#include <iostream>
#include <fstream>
#include <cfloat>
#include <cmath>
#include "cluster.h"

using namespace std;

int Cluster::counter = 0;
int Cluster::aval_id = 0;

Cluster::Cluster(int s_num, int m_num, int v_dim) {
// Cluster::Cluster() {
   state_num = s_num;
   mixture_num = m_num;
   vector_dim = v_dim;
   member_num = 0;
   precompute_status = false;
   // trans = new float[state_num];
   for (int i = 0; i < state_num; ++i) {
      vector<float> state_trans;
      for (int j = 0; j < state_num + 1; ++j) {
         state_trans.push_back(0.0);
      }
      trans.push_back(state_trans);
   }
   for (int i = 0; i < state_num; ++i) {
      vector<float> state_trans;
      for (int j = 0; j < state_num + 1; ++j) {
         state_trans.push_back(0.0);
      }
      cache_trans.push_back(state_trans);
   }
   id = -1;
   for(int i = 0; i < state_num; ++i) {
      Gmm prototype; 
      prototype.init(mixture_num, vector_dim);
      emissions.push_back(prototype);
      caches.push_back(prototype);
   }
   age = 0;
}

void Cluster::init(const int s_state_num, \
                   const int s_mixture_num, \
                   const int s_vector_dim) {
   state_num = s_state_num;
   mixture_num = s_mixture_num;
   vector_dim = s_vector_dim;
   for (int i = 0 ; i < state_num; ++i) {
      Gmm new_gmm(mixture_num, vector_dim);
      emissions.push_back(new_gmm);
   }
   for (int i = 0 ; i < state_num; ++i) {
      vector<float> inner_trans;
      for (int j = 0 ; j < state_num + 1; ++j) {
         inner_trans.push_back(0);
      }
      trans.push_back(inner_trans);
   }
   id = -1;
   age = 0;
}


void Cluster::increase_trans(const int i, const int j) {
   ++cache_trans[i][j];
}

void Cluster::decrease_trans(const int i, const int j) {
   --cache_trans[i][j];
}

void Cluster::increase_cache(const int s_t, \
  const int m_t, const float* data) {
   caches[s_t].get_mixture(m_t).add_sample_count_mean(data, 1);
   caches[s_t].get_mixture(m_t).add_sample_count_var(data, 1);
   caches[s_t].get_mixture(m_t).add_sample_count_weight(1);
}

void Cluster::decrease_cache(const int s_t, \
  const int m_t, const float* data) {
   caches[s_t].get_mixture(m_t).sub_sample_count_mean(data, 1);
   caches[s_t].get_mixture(m_t).sub_sample_count_var(data, 1);
   caches[s_t].get_mixture(m_t).sub_sample_count_weight(1);
}

void Cluster::update_emission(Gmm& gmm, int index) {
   emissions[index] = gmm;
}

const float* Cluster::get_cache_mean(const int s_t, const int m_t) {
   return caches[s_t].get_mixture(m_t).get_mean();
}

const float* Cluster::get_cache_var(const int s_t, const int m_t) {
   return caches[s_t].get_mixture(m_t).get_var();
}

int Cluster::get_cache_weight(const int s_t, const int m_t) {
   return caches[s_t].get_mixture(m_t).get_weight();
}

void Cluster::update_trans(vector<vector<float> > new_trans) {
   trans = new_trans;
}

void Cluster::append_member(Segment* data) {
   int frame_num = data -> get_frame_num();
   for(int i = 0; i < frame_num - 1; ++i) {
      int s_t = data -> get_hidden_states(i);
      int s_t_1 = data -> get_hidden_states(i + 1);
      int m_t = data -> get_mixture_id(i);
      const float* frame_i = data -> get_frame_i_data(i);
      increase_trans(s_t, s_t_1);
      increase_cache(s_t, m_t, frame_i);
   }
   // deal with the last frame
   int i = frame_num - 1;
   int s_t = data -> get_hidden_states(i);
   int m_t = data -> get_mixture_id(i);
   const float* frame_i = data -> get_frame_i_data(i);
   increase_trans(s_t, state_num);
   increase_cache(s_t, m_t, frame_i);
   ++member_num;
}

void Cluster::remove_members(Segment* data) {
   int frame_num = data -> get_frame_num();
   for(int i = 0; i < frame_num - 1; ++i) {
      int s_t = data -> get_hidden_states(i);
      int s_t_1 = data -> get_hidden_states(i + 1);
      int m_t = data -> get_mixture_id(i);
      const float* frame_i = data -> get_frame_i_data(i);
      decrease_trans(s_t, s_t_1);
      decrease_cache(s_t, m_t, frame_i);
   }
   // deal with the last frame
   int i = frame_num - 1;
   int s_t = data -> get_hidden_states(i);
   int m_t = data -> get_mixture_id(i);
   const float* frame_i = data -> get_frame_i_data(i);
   decrease_trans(s_t, state_num);
   decrease_cache(s_t, m_t, frame_i);
   --member_num;
}

void Cluster::set_precompute_status(const bool new_status) {
   precompute_status = new_status;
   for (int i = 0; i < state_num; ++i) {
      emissions[i].set_precompute_status(new_status);
   }
}

void Cluster::set_cluster_id(int s_id) {
   id = s_id;
}

void Cluster::set_cluster_id() {
   id = aval_id;
   aval_id++;
}

int Cluster::get_member_num() const {
   return member_num;
}

int Cluster::get_cluster_id() const {
   return id;
}

// Compute P(d|HMM)
double Cluster::compute_likelihood(const Segment& data, const int offset){
   double cur_scores[state_num];
   double pre_scores[state_num];
   for (int i = 0; i < state_num; ++i) {
      cur_scores[i] = 0.0;
      pre_scores[i] = 0.0;
   }
   pre_scores[0] = compute_emission_likelihood(0, \
                                               data.get_frame_i_data(0),\
                                               data.get_frame_index(0) - offset);
   if (data.get_frame_num() > 1) {
      for (int cur_state = 0; cur_state < state_num; ++cur_state) {
         // double a;
         // a = log(trans[0][cur_state]);
         double prob_emit = compute_emission_likelihood(cur_state, \
            data.get_frame_i_data(1), data.get_frame_index(1) - offset);
         // cur_scores[cur_state] = pre_scores[0] + a + prob_emit; 
         cur_scores[cur_state] = pre_scores[0] + prob_emit; 
      }
      for (int i = 0; i < state_num; ++i) {
         pre_scores[i] = cur_scores[i];
      }
      for (int i = 2; i < data.get_frame_num(); ++i) {
         for (int cur_state = 0; cur_state < state_num; ++cur_state) {
            cur_scores[cur_state] = 0.0;
            /*
            double log_arr[cur_state + 1];
            for (int pre_state = 0; pre_state <= cur_state; ++pre_state) {
               // double a;
               // a = trans[pre_state][cur_state];
               // log_reg.push_back(pre_scores[pre_state] + log(a));
               log_arr[pre_state] = pre_scores[pre_state];
            }
            */
            double prob_emit_i = compute_emission_likelihood(cur_state, \
               data.get_frame_i_data(i), data.get_frame_index(i) - offset);
            if (cur_state == 0) {
               cur_scores[cur_state] = pre_scores[0];
            }
            else {
               cur_scores[cur_state] = calculator.sum_logs(\
                 pre_scores, cur_state + 1);
            }
            cur_scores[cur_state] += prob_emit_i;
         }
         for (int j = 0; j < state_num; ++j ) {
            pre_scores[j] = cur_scores[j];
         }
      }
   }
   if (data.get_frame_num() > 1) {
      return pre_scores[state_num - 1]; // + log(trans[state_num - 1][state_num]); 
   }
   else {
      return pre_scores[0]; // + log(trans[0][state_num]);
   }
}

void Cluster::precompute(int total, const float** data) {
   for(int i = 0; i < state_num; ++i) {
      emissions[i].precompute(total, data);
   }
}

// Compute P(x|Guassian) 
double Cluster::compute_emission_likelihood(int state, \
                                            const float* data, \
                                            const int index) {
   if (precompute_status) {
      return emissions[state].compute_likelihood(index);
   }
   else {
      return emissions[state].compute_likelihood(data); 
   }
}

vector<double> Cluster::compute_gmm_mixture_posterior_weight(int state, \
                                            const float* data, int index) {
   if (precompute_status) {
      return emissions[state].compute_posterior_weight(index);
   }
   else {
      return emissions[state].compute_posterior_weight(data);
   }
}

void Cluster::set_state_mixture_weight(const int s, \
  const int m, const float w) {
   emissions[s].set_mixture_weight(m, w);
}

void Cluster::set_state_mixture_det(const int s, \
  const int m, const float det) {
   emissions[s].set_mixture_det(m, det); 
}

void Cluster::set_state_mixture_mean(const int s, \
  const int m, const float* mean) {
   emissions[s].set_mixture_mean(m, mean);
}

void Cluster::set_trans(const float* s_trans) {
   for (int i = 0 ; i < state_num; ++i) {
      for (int j = 0 ; j < state_num + 1; ++j) {
         trans[i][j] = s_trans[i * (state_num + 1) + j];
      }
   }
}

void Cluster::set_state_mixture_var(const int s, \
  const int m, const float* var) {
   emissions[s].set_mixture_var(m, var);
}

float Cluster::get_state_trans_prob(int from, int to) const {
   return trans[from][to];
}

void Cluster::show_member_len() {
   cout << "I am Cluster " << id << 
     " and I have " << member_num << " members." << endl;
}
void Cluster::state_snapshot(const string& fn) {
   ofstream fout(fn.c_str(), ios::app);
   // write out members' info
   int member_len = member_num; 
   // write out member number
   fout.write(reinterpret_cast<char*> (&member_len), sizeof(int));
   // write state number
   fout.write(reinterpret_cast<char*> (&state_num), sizeof(int));
   // write mixture_num
   fout.write(reinterpret_cast<char*> (&mixture_num), sizeof(int));
   // write vector_dim
   fout.write(reinterpret_cast<char*> (&vector_dim), sizeof(int));
   // write out trans info
   float copy_trans[state_num * (state_num + 1)];
   for (int i = 0; i < state_num; ++i) {
      for (int j = 0; j < state_num + 1; ++j) {
         copy_trans[i * (state_num + 1) + j] = trans[i][j]; 
      }
   }
   fout.write(reinterpret_cast<char*> (copy_trans), sizeof(float) * state_num * (state_num + 1)); 
   // write gmm's info
   for (int i = 0; i < state_num; ++i) {
      for (int j = 0; j < mixture_num; ++j) {
         float w_i = emissions[i].get_mixture(j).get_weight();
         float det_i = emissions[i].get_mixture(j).get_det();
         const float* mean_i = emissions[i].get_mixture(j).get_mean(); 
         const float* var_i = emissions[i].get_mixture(j).get_var();
         fout.write(reinterpret_cast<char*> (&w_i), sizeof(float)); 
         fout.write(reinterpret_cast<char*> (&det_i), sizeof(float)); 
         fout.write(reinterpret_cast<const char*> (mean_i), sizeof(float) * vector_dim);
         fout.write(reinterpret_cast<const char*> (var_i), sizeof(float) * vector_dim);
      }
   }
   // write out each member
   /*
   vector<Segment*>::iterator iter_segments;
   for(iter_segments = members.begin(); iter_segments != \
     members.end(); ++iter_segments) {
      // write tag
      string tag = (*iter_segments) -> get_tag();
      int tag_len = tag.length() + 1;
      // write tag length
      fout.write(reinterpret_cast<char*> (&tag_len), sizeof(int));
      fout.write(reinterpret_cast<const char*> (tag.c_str()), tag_len);
      // write start frame
      int start = (*iter_segments) -> get_start_frame();
      fout.write(reinterpret_cast<char*> (&start), sizeof(int));
      // write end frame
      int end = (*iter_segments) -> get_end_frame();
      fout.write(reinterpret_cast<char*> (&end), sizeof(int));
      // write cluster id
      int cluster_id = (*iter_segments) -> get_cluster_id();
      fout.write(reinterpret_cast<char*> (&cluster_id), sizeof(int));
      // write frame number
      int frame_num = (*iter_segments) -> get_frame_num();
      fout.write(reinterpret_cast<char*> (&frame_num), sizeof(int));
      // write hidden states
      const int* hidden_states = (*iter_segments) -> get_hidden_states_all();
      fout.write(reinterpret_cast<const char*> (hidden_states), \
        sizeof(int) * frame_num);
      // write mixture id
      const int* mixture_id = (*iter_segments) -> get_mixture_id_all();
      fout.write(reinterpret_cast<const char*> (mixture_id), \
        sizeof(int) * frame_num);
      for (int i = 0 ; i < frame_num; ++i) {
         const float* frame_i = (*iter_segments) -> get_frame_i_data(i);
         fout.write(reinterpret_cast<const char*> (frame_i), \
           sizeof(float) * vector_dim);
      }
   }
   */
   fout.close();
}

Cluster::~Cluster(){
   /*
   if (id != -1) {
      cout << "cluster " << id << " has been destructed" << endl;
   }
   */
}

