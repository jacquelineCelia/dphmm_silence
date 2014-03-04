/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./manager.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>

#include "manager.h"
#include "sampler.h"
#include "cluster.h"
#include "segment.h"
#include "gmm.h"

using namespace std;

Manager::Manager() {
}

bool Manager::load_config(const string& fnconfig) {
   ifstream fconfig(fnconfig.c_str(), ifstream::in);
   if (!fconfig.good()) {return false;}
   fconfig >> s_dim;
   if (!fconfig.good()) {return false;}
   fconfig >> s_state;
   if (!fconfig.good()) {return false;}
   fconfig >> s_mixture;
   if (!fconfig.good()) {return false;}
   fconfig >> s_dp_alpha;
   if (!fconfig.good()) {return false;}
   fconfig >> s_beta_alpha;
   if (!fconfig.good()) {return false;}
   fconfig >> s_beta_beta;
   if (!fconfig.good()) {return false;}
   fconfig >> s_gamma_shape;
   if (!fconfig.good()) {return false;}
   fconfig >> s_norm_kappa;
   if (!fconfig.good()) {return false;}
   fconfig >> s_gamma_weight_alpha;
   if (!fconfig.good()) {return false;}
   fconfig >> s_gamma_trans_alpha;
   if (!fconfig.good()) {return false;}
   fconfig >> s_h0;
   if (!fconfig.good()) {return false;}
   fconfig >> sil_num_state;
   if (!fconfig.good()) {return false;}
   fconfig >> sil_num_mixture;
   if (!fconfig.good()) {return false;}
   fconfig >> sil_self_trans_prob;
   if (!fconfig.good()) {return false;}
   fconfig >> new_class_threshold; 
   fconfig.close();
   return true;
}

bool Manager::load_silence_model(const string& fn_sil) {
    ifstream fsil(fn_sil.c_str(), ios::binary);
    if (!fsil.is_open()) {
        return false;
    }
    // load silence model
    Cluster* sil_cluster = new Cluster(sil_num_state, sil_num_mixture, s_dim); 
    sil_cluster -> set_cluster_id();
    clusters.push_back(sil_cluster);
    Cluster::counter++;
    vector<vector<float> > trans_probs;
    for (int i = 0; i < sil_num_state; ++i) {
        vector<float> trans_prob(sil_num_state + 1, 0);
        trans_prob[0]= log(sil_self_trans_prob);
        trans_prob[1] = log(1 - sil_self_trans_prob);
        trans_probs.push_back(trans_prob);
    }
    sil_cluster -> update_trans(trans_probs); 
    Gmm new_state(sil_num_mixture, s_dim);
    for (int m = 0; m < sil_num_mixture; ++m) {
        float weight;
        fsil.read(reinterpret_cast<char*> (&weight), sizeof(float));
        vector<float> mean(s_dim, 0);
        vector<float> pre(s_dim, 0);
        fsil.read(reinterpret_cast<char*> (&mean[0]), sizeof(float) * s_dim);
        fsil.read(reinterpret_cast<char*> (&pre[0]), sizeof(float) * s_dim);
        new_state.get_mixture(m).update_mean(&mean[0]);
        new_state.get_mixture(m).update_var(&pre[0]);
        new_state.get_mixture(m).update_det();
        new_state.get_mixture(m).update_weight(weight);
    }
    for (int i = 0 ; i < sil_num_state; ++i) {
        sil_cluster -> update_emission(new_state, i);
    }
    fsil.close();
    return true;
}

bool Manager::load_gmm_from_file(ifstream& fgmm, Gmm& gmm, bool inverse) {
   if (!fgmm.good()) {
      return false;
   }
   cout << "Reading out gmms..." << endl;
   for (int i = 0; i < s_mixture; ++i) {
      float w_i;
      float mean_i[s_dim];
      float var_i[s_dim];
      fgmm.read(reinterpret_cast<char*>(&w_i), sizeof(float));
      fgmm.read(reinterpret_cast<char*>(mean_i), sizeof(float) * s_dim);
      fgmm.read(reinterpret_cast<char*>(var_i), sizeof(float) * s_dim);
      gmm.get_mixture(i).update_weight(w_i);
      gmm.get_mixture(i).update_mean(mean_i);
      if (!inverse) {
         for (int j = 0; j < s_dim; ++j) {
            var_i[j] = 1 / var_i[j];
         }
      }
      /*
      for (int j = 0; j < s_dim; ++j) {
         cout << var_i[j] << endl;
      }
      */
      gmm.get_mixture(i).update_var(var_i);
      gmm.get_mixture(i).update_det();
   }
   return true;
}

bool Manager::load_gmm(const string& fngmm) {
   cout << "Loading gmm" << endl;
   ifstream fgmm(fngmm.c_str(), ios::binary);
   if (!fgmm.good()) {
      return false;
   }
   s_gmm.init(s_mixture, s_dim);
   if (!load_gmm_from_file(fgmm, s_gmm, true)) {
      return false;
   }
   fgmm.close();
   return true;
}

string Manager::get_basename(string s) {
   size_t found_last_slash, found_last_period;
   found_last_slash = s.find_last_of("/");
   found_last_period = s.find_last_of(".");
   return s.substr(found_last_slash + 1, \
     found_last_period - 1 - found_last_slash);
}

bool Manager::load_bounds(const string& fnbound_list, const int g_size) {
   group_size = g_size;
   ifstream fbound_list(fnbound_list.c_str(), ifstream::in);
   if (!fbound_list.is_open()) {
      return false;
   }
   cout << "file opened" << endl;
   int input_counter = 0;
   while (fbound_list.good()) {
      string fn_index;
      string fn_data;
      fbound_list >> fn_index;
      fbound_list >> fn_data;
      if (fn_index != "" && fn_data != "") {
         ++input_counter;
         ifstream findex(fn_index.c_str(), ifstream::in);
         ifstream fdata(fn_data.c_str(), ifstream::binary);
         string basename = get_basename(fn_data);
         vector<Bound*> a_seg;
         if (!findex.is_open()) {
            return false;
         }
         if (!fdata.is_open()) {
            return false;
         }
         int start = 0;
         int end = 0;
         int cluster_id = -1;
         cout << "Loading " << fn_data << "..." << endl;
         int total_frame_num;
         findex >> total_frame_num;
         while (end != total_frame_num - 1) {
            findex >> start;
            findex >> end;
            findex >> cluster_id;
            int frame_num = end - start + 1;
            float** frame_data = new float*[frame_num];
            for (int i = 0; i < frame_num; ++i) {
               frame_data[i] = new float[s_dim];
               fdata.read(reinterpret_cast<char*>(frame_data[i]), \
                    sizeof(float) * s_dim);
            }
            bool utt_end = false;
            if (end == total_frame_num - 1) {
               utt_end = true;
            }
            Bound* new_bound = new Bound(start, end, s_dim, utt_end);
            new_bound -> set_data(frame_data);
            new_bound -> set_index(Bound::index_counter);
            new_bound -> set_start_frame_index(Bound::total_frames);
            ++Bound::index_counter;
            Bound::total_frames += end - start + 1;
            for(int i = 0; i < frame_num; ++i) {
               delete[] frame_data[i];
            }
            delete[] frame_data;
            if (frame_num) {
               bounds.push_back(new_bound);
               a_seg.push_back(new_bound);
               // Sampler::sample_boundary(Bound*) is for sampling from prior
               bool phn_end = sampler.sample_boundary(new_bound);
               new_bound -> set_phn_end(phn_end);
               if (phn_end) {
                  Segment* new_segment = new Segment(basename, a_seg);
                  ++Segment::counter;
                  segments.push_back(new_segment);
                  Cluster* new_c;
                  float uniform_real = sampler.sample_from_unit();
                  if (uniform_real < new_class_threshold) {
                      new_c = sampler.sample_just_cluster(*new_segment, clusters, true);
                  }
                  else {
                      new_c = sampler.sample_just_cluster(*new_segment, clusters, false);
                  }
                  if (cluster_id == 0) {
                      new_c = clusters[0];
                  }
                  sampler.sample_more_than_cluster(*new_segment, clusters, new_c);
                  vector<Bound*>::iterator iter_members = a_seg.begin();
                  for(; iter_members != a_seg.end(); ++iter_members) {
                     (*iter_members) -> set_parent(new_segment);
                  }
                  new_segment -> change_hash_status(false);
                  cout << Segment::counter << " segments..." << endl;
                  cout << Cluster::counter << " clusters..." << endl;
                  a_seg.clear();
               }
            }
            else {
               delete new_bound;
               Bound::index_counter--;
            }
         }
         if (a_seg.size()) {
             cout << "Not cleaned" << endl;
             Segment* new_segment = new Segment(basename, a_seg);
             ++Segment::counter;
             segments.push_back(new_segment);
             Cluster* new_c = sampler.sample_just_cluster(*new_segment, clusters);
             sampler.sample_more_than_cluster(*new_segment, clusters, new_c);
             vector<Bound*>::iterator iter_members = a_seg.begin();
             for(; iter_members != a_seg.end(); ++iter_members) {
                (*iter_members) -> set_parent(new_segment);
             }
             cout << Segment::counter << " segments..." << endl;
             cout << Cluster::counter << " clusters..." << endl;
             a_seg.clear();
         }
         vector<Bound*>::iterator to_last = bounds.end();
         to_last--;
         (*to_last) -> set_utt_end(true);
         (*to_last) -> set_phn_end(true);
         findex.close();
         fdata.close();
      }
      if (!(input_counter % group_size) && fn_index != "" && fn_data != "") {
         batch_groups.push_back(bounds.size());
         cout << input_counter << " and " << bounds.size() << endl;
      }
      if (!(input_counter % 10)) {
         update_clusters(false, 0);   
      }
   }
   if (input_counter % group_size) {
      batch_groups.push_back(bounds.size());
      cout << input_counter << " and " << bounds.size() << endl;
   }
   fbound_list.close();
   load_data_to_matrix(); 
   return true;
}

void Manager::load_data_to_matrix() {
   data = new const float*[Bound::total_frames];
   list<Segment*>::iterator iter;
   int ptr = 0;
   for(iter = segments.begin(); iter != segments.end(); ++iter) { 
      int frame_num = (*iter) -> get_frame_num();
      for (int i = 0; i < frame_num; ++i) {
         data[ptr++] = (*iter) -> get_frame_i_data(i);
      }
   }
}

/*
bool Manager::load_segments(const string& fndata_list) {
   ifstream fdata_list(fndata_list.c_str(), ifstream::in);
   if (!fdata_list.is_open()) {
      return false;
   }
   while (fdata_list.good()) { 
      string fn_index;
      string fn_data;
      fdata_list >> fn_index; 
      fdata_list >> fn_data;
      if (fn_index != "" && fn_data != "") {
         ifstream findex(fn_index.c_str(), ifstream::in);
         ifstream fdata(fn_data.c_str(), ios::binary);
         string basename = get_basename(fn_data);
         if (!findex.is_open()) {
            return false;
         }
         if (!fdata.is_open()) {
            return false;
         }
         int start = 0;
         int end = 0;
         cout << "Loading " << fn_data << "..." << endl;
         while (findex.good()) {
            findex >> end;
            cout << "Loading segment " << start << "-" << end << "..." << endl; 
            int frame_num = end - start + 1;
            if (frame_num >= 3) {
               float** frame_data = new float* [frame_num];
               for (int i = 0; i < frame_num; ++i) {
                  frame_data[i] = new float[s_dim];
                  fdata.read(reinterpret_cast<char*>(frame_data[i]), \
                  sizeof(float) * s_dim);
               }
               Segment* new_segment = new Segment(frame_num, s_dim, \
                 basename, start, end);
               new_segment -> set_frame_data(frame_data);
               ++Segment::counter;
               segments.push_back(new_segment);
               sampler.sample_cluster(*new_segment, clusters);
               cout << Segment::counter << " segments..." << endl;
               cout << Cluster::counter << " clusters..." << endl;
               for (int i = 0; i < frame_num; ++i) {
                  delete[] frame_data[i];
               }
               delete[] frame_data;
               // new_segment -> show_data();
            }
            start = end + 1;
         }
         fdata.close();
         findex.close();
      }
   }
   fdata_list.close();
   return true;
}
*/
void Manager::init_sampler() {
   sampler.init_prior(s_dim, \
     s_state, s_mixture, \
     s_dp_alpha, \
     s_beta_alpha, s_beta_beta, \
     s_gamma_shape, \
     s_norm_kappa, \
     s_gamma_weight_alpha, \
     s_gamma_trans_alpha, \
     s_gmm, s_h0); 
}

bool Manager::update_boundaries(const int group_ptr) {
   cout << "Total bounds is " << Bound::index_counter << endl;
   cout << "Total segs is " << Segment::counter << endl;
   vector<Bound*>::iterator iter_bounds = bounds.begin();
   int i = group_ptr == 0 ? 0 : batch_groups[group_ptr - 1];
   Sampler::offset = bounds[i] -> get_start_frame_index();
   for (; i < batch_groups[group_ptr]; ++i) {
   //for(iter_bounds = bounds.begin(); iter_bounds != bounds.end(); \
   //  ++iter_bounds) {
      if (!sampler.sample_boundary(iter_bounds + i, segments, clusters)) {
         Segment* parent = (*iter_bounds) -> get_parent();
         cout << "Cannot update bound " << parent -> get_tag() 
              << "frame " << parent -> get_start_frame() << " to "
              << parent -> get_end_frame() << endl;
         return false;
      }
   }
   return true;
}

void Manager::update_clusters(const bool to_precompute, const int group_ptr) {
   vector<Cluster*>::iterator iter_clusters = clusters.begin();
   for (int k = 0; k < clusters.size(); ++k) {
      // clusters[k] -> show_member_len();
      if (clusters[k] -> get_age() >= 500000000000 && \
       clusters[k] -> get_member_num() <= 100) {
         int cluster_id = clusters[k] -> get_cluster_id();
         int member_num = clusters[k] -> get_member_num(); 
         delete clusters[k]; 
         clusters.erase(iter_clusters + k);
         --Cluster::counter;
         Segment::counter -= member_num;
         list<Segment*>::iterator iter_segments = segments.begin();
         for (; iter_segments != segments.end(); ++iter_segments) {
            if ((*iter_segments) -> get_cluster_id() == cluster_id) {
               ++Segment::counter;
               Cluster* new_c = \
                  sampler.sample_just_cluster(*(*iter_segments), clusters);
               sampler.sample_more_than_cluster(\
                  *(*iter_segments), clusters, new_c); 
            }
         }
      }
      else {
         sampler.sample_hmm_parameters(*clusters[k]);
         clusters[k] -> update_age();
         if (to_precompute) {
            vector<Bound*>::iterator iter_bounds = bounds.begin();
            int i = group_ptr == 0 ? 0 : batch_groups[group_ptr - 1];
            Bound* start_bound = *(iter_bounds + i);
            Bound* end_bound = *(iter_bounds + batch_groups[group_ptr] - 1);
            int start_index = start_bound -> get_start_frame_index();
            int end_index = end_bound -> get_start_frame_index() + \
                         end_bound -> get_frame_num();
            const float* sec_data[end_index - start_index];
            for (int j = start_index; j < end_index; ++j) {
               sec_data[j - start_index] = data[j];
            }
            sampler.precompute(clusters[k], end_index - start_index, sec_data);
            sampler.set_precompute_status(clusters[k], to_precompute);
         }
      }
   }
}

void Manager::gibbs_sampling(const int num_iter, const string result_dir) {
   for (int i = 0; i <= num_iter; ++i) {
      /*
      if (i <= 10000) {
         Sampler::annealing = 10 - (i / (num_iter / 10));
      }
      else {
         Sampler::annealing = 10.1;
      }
      */
      Sampler::annealing = 10.1;
      cout << "starting the " << i << " th iteration..." << endl;
      cout << "Total number of clusters is " << Cluster::counter << \
        ", to double check " << clusters.size() << endl;
      cout << "Updating clusters..." << endl;
      update_clusters(true, (i % batch_groups.size()));
      cout << "Updating boundaries..." << endl;
      if (!update_boundaries(i % batch_groups.size())) {
         cout << "Cannot update boundaries..." << endl;
         return;
      }
      if (!(i % 100) && i != 0) {
         list<Segment*>::iterator iter_segments; 
         for (iter_segments = segments.begin(); iter_segments != segments.end(); \
               ++iter_segments) {
            stringstream num_to_string;
            num_to_string << i;
            (*iter_segments) -> write_class_label(result_dir + "/" + num_to_string.str());
         }
      }
      if (!(i % 100) && i != 0) {
         stringstream num_to_string;
         num_to_string << i;
         string fsnapshot = result_dir + "/" + \
                            num_to_string.str() + "/snapshot";
         cout << "Writing out to " << fsnapshot << " ..." << endl;
         if (!state_snapshot(fsnapshot)) {
            cout << "Cannot open " << fsnapshot << 
              ". Please make sure the path exists" << endl; 
            return;
         }
         vector<Cluster*>::iterator iter_clusters;
         for (iter_clusters = clusters.begin(); iter_clusters != clusters.end(); \
           ++iter_clusters) {
            (*iter_clusters) -> state_snapshot(fsnapshot);
         }
      }
      /*
      if (((num_iter - i <= 1000) && !(i % 100)) || i == 10 || (i % 500 == 0)) {
         stringstream num_to_string;
         num_to_string << i;
         string fsnapshot = result_dir + "/" + \
                            num_to_string.str() + "/snapshot";
         cout << "Writing out to " << fsnapshot << " ..." << endl;
         if (!state_snapshot(fsnapshot)) {
            cout << "Cannot open " << fsnapshot << 
              ". Please make sure the path exists" << endl; 
            return;
         }
         for (iter_clusters = clusters.begin(); iter_clusters != clusters.end(); \
           ++iter_clusters) {
            (*iter_clusters) -> state_snapshot(fsnapshot);
         }
      }
      */
   }
}

bool Manager::state_snapshot(const string& fn) {
   ofstream fout(fn.c_str(), ios::out);
   if (!fout.good()) {
      return false;
   }
   int data_counter = Segment::counter;
   fout.write(reinterpret_cast<char*> (&data_counter), sizeof(int));
   int cluster_counter = Cluster::counter;
   fout.write(reinterpret_cast<char*> (&cluster_counter), sizeof(int));
   /*
   int aval_id = Cluster::aval_id;
   fout.write(reinterpret_cast<char*>(&aval_id), sizeof(int));
   int cluster_counter = Cluster::counter;
   fout.write(reinterpret_cast<char*>(&cluster_counter), sizeof(int));
   int aval_data = Segment::counter;
   fout.write(reinterpret_cast<char*>(&aval_data), sizeof(int));
   int num_clusters = clusters.size();
   fout.write(reinterpret_cast<char*>(&num_clusters), sizeof(int));
   cout << num_clusters << endl;
   */
   fout.close();
   return true;
}

bool Manager::load_in_model(const string& fname, const int threshold) {
   ifstream fin(fname.c_str(), ios::binary);
   int data_num;
   int cluster_num;
   if (!fin.good()) {
      cout << fname << " cannot be opened." << endl;
      return false;
   }
   fin.read(reinterpret_cast<char*> (&data_num), sizeof(int));
   fin.read(reinterpret_cast<char*> (&cluster_num), sizeof(int));
   cout << "number of clusters " << cluster_num << endl;
   for (int i = 0; i < cluster_num; ++i) {
      int member_num;
      int state_num;
      int mixture_num;
      int vector_dim;
      fin.read(reinterpret_cast<char*> (&member_num), sizeof(int));
      fin.read(reinterpret_cast<char*> (&state_num), sizeof(int));
      fin.read(reinterpret_cast<char*> (&mixture_num), sizeof(int));
      fin.read(reinterpret_cast<char*> (&vector_dim), sizeof(int));
      Cluster* new_cluster = new Cluster(state_num, mixture_num, vector_dim);
      new_cluster -> set_member_num(member_num);
      float trans[state_num * (state_num + 1)];
      fin.read(reinterpret_cast<char*> (trans), sizeof(float) * \
        state_num * (state_num + 1));
      new_cluster -> set_trans(trans);
      for (int j = 0; j < state_num; ++j) {
         for (int k = 0 ; k < mixture_num; ++k) {
            float w;
            float det;
            float mean[vector_dim];
            float var[vector_dim];
            fin.read(reinterpret_cast<char*> (&w), \
              sizeof(float));
            fin.read(reinterpret_cast<char*> (&det), \
              sizeof(float));
            fin.read(reinterpret_cast<char*> (mean), \
              sizeof(float) * vector_dim);
            fin.read(reinterpret_cast<char*> (var), \
              sizeof(float) * vector_dim);
            new_cluster -> set_state_mixture_weight(j, k, w);
            new_cluster -> set_state_mixture_det(j, k, det);
            new_cluster -> set_state_mixture_mean(j, k, mean);
            new_cluster -> set_state_mixture_var(j, k, var);
         }
      }
      if (new_cluster -> get_member_num() > threshold) {
         clusters.push_back(new_cluster);
      }
      else {
         data_num -= new_cluster -> get_member_num();
         delete new_cluster;
      }
   }
   for (int i = 0; i < clusters.size(); ++i) {
      clusters[i] -> set_member_num(0);
   }
   Cluster::counter = clusters.size();
   fin.close();
   return true;

}

bool Manager::load_in_model_id(const string& fn_mixture_id) {
   ifstream fin(fn_mixture_id.c_str());
   for (int i = 0; i < clusters.size(); ++i) {
      int c_id;
      fin >> c_id;
      clusters[i] -> set_cluster_id(c_id);
      Cluster::aval_id = c_id + 1;
   }
   fin.close();
   return true;
}

Cluster* Manager::find_cluster(const int c_id) {
   for (int i = 0; i < clusters.size(); ++i) {
      if (clusters[i] -> get_cluster_id() == c_id) {
         return clusters[i];
      }
   }
   return NULL;
}

bool Manager::load_in_data(const string& fnbound_list, const int g_size) {
   group_size = g_size;
   ifstream fbound_list(fnbound_list.c_str(), ifstream::in);
   if (!fbound_list.is_open()) {
      return false;
   }
   cout << "file opened" << endl;
   int input_counter = 0;
   while (fbound_list.good()) {
      string fn_index;
      string fn_data;
      fbound_list >> fn_index;
      fbound_list >> fn_data;
      if (fn_index != "" && fn_data != "") {
         ++input_counter;
         ifstream findex(fn_index.c_str(), ifstream::in);
         ifstream fdata(fn_data.c_str(), ifstream::binary);
         string basename = get_basename(fn_data);
         vector<Bound*> a_seg;
         if (!findex.is_open()) {
            return false;
         }
         if (!fdata.is_open()) {
            return false;
         }
         int start = 0;
         int end = 0;
         int cluster_label;
         cout << "Loading " << fn_data << "..." << endl;
         int total_frame_num;
         findex >> total_frame_num;
         while (end != total_frame_num - 1) {
            findex >> start;
            findex >> end;
            findex >> cluster_label; 
            int frame_num = end - start + 1;
            float** frame_data = new float*[frame_num];
            for (int i = 0; i < frame_num; ++i) {
               frame_data[i] = new float[s_dim];
               fdata.read(reinterpret_cast<char*>(frame_data[i]), \
                    sizeof(float) * s_dim);
            }
            bool utt_end = false;
            if (end == total_frame_num - 1) {
               utt_end = true;
            }
            Bound* new_bound = new Bound(start, end, s_dim, utt_end);
            new_bound -> set_data(frame_data);
            new_bound -> set_index(Bound::index_counter);
            new_bound -> set_start_frame_index(Bound::total_frames);
            ++Bound::index_counter;
            Bound::total_frames += end - start + 1;
            for(int i = 0; i < frame_num; ++i) {
               delete[] frame_data[i];
            }
            delete[] frame_data;
            if (frame_num) {
               bounds.push_back(new_bound);
               a_seg.push_back(new_bound);
               // Sampler::sample_boundary(Bound*) is for sampling from prior
               bool phn_end = false;
               if (cluster_label != -1 || utt_end) {
                  phn_end = true;
               }
               new_bound -> set_phn_end(phn_end);
               if (phn_end) {
                  Segment* new_segment = new Segment(basename, a_seg);
                  ++Segment::counter;
                  segments.push_back(new_segment);
                  Cluster* new_c = find_cluster(cluster_label);
                  sampler.sample_more_than_cluster(*new_segment, clusters, new_c);
                  vector<Bound*>::iterator iter_members = a_seg.begin();
                  for(; iter_members != a_seg.end(); ++iter_members) {
                     (*iter_members) -> set_parent(new_segment);
                  }
                  new_segment -> change_hash_status(false);
                  cout << Segment::counter << " segments..." << endl;
                  cout << Cluster::counter << " clusters..." << endl;
                  a_seg.clear();
               }
            }
            else {
               delete new_bound;
               Bound::index_counter--;
            }
         }
         if (a_seg.size()) {
             cout << "Not cleaned" << endl;
             Segment* new_segment = new Segment(basename, a_seg);
             ++Segment::counter;
             segments.push_back(new_segment);
             Cluster* new_c = sampler.sample_just_cluster(*new_segment, clusters);
             sampler.sample_more_than_cluster(*new_segment, clusters, new_c);
             vector<Bound*>::iterator iter_members = a_seg.begin();
             for(; iter_members != a_seg.end(); ++iter_members) {
                (*iter_members) -> set_parent(new_segment);
             }
             cout << Segment::counter << " segments..." << endl;
             cout << Cluster::counter << " clusters..." << endl;
             a_seg.clear();
         }
         vector<Bound*>::iterator to_last = bounds.end();
         to_last--;
         (*to_last) -> set_utt_end(true);
         (*to_last) -> set_phn_end(true);
         findex.close();
         fdata.close();
      }
      if (!(input_counter % group_size) && fn_index != "" && fn_data != "") {
         batch_groups.push_back(bounds.size());
         cout << input_counter << " and " << bounds.size() << endl;
      }
   }
   if (input_counter % group_size) {
      batch_groups.push_back(bounds.size());
      cout << input_counter << " and " << bounds.size() << endl;
   }
   fbound_list.close();
   load_data_to_matrix(); 
   return true;

}

bool Manager::load_snapshot(const string& fn_snapshot, \
  const string& fn_mixture_id, const string& fn_data, \
  const int g_size) {
   if (!load_in_model(fn_snapshot, 0)) {
      cout << "Cannot load in snapshot" << endl;
      return false;
   }
   if (!load_in_model_id(fn_mixture_id)) {
      cout << "Cannot load in model id" << endl;
      return false;
   }
   if (!load_in_data(fn_data, g_size)) {
      cout << "Cannot load in data file." << endl;
      return false;
   }
   for (int c = 0; c < clusters.size(); ++c) {
      cout << "print out" << endl;
      cout << clusters[c] -> get_member_num() << endl;
      cout << clusters[c] -> get_cluster_id() << endl;
   }
   return true;
}

/*
bool Manager::load_snapshot(const string& fn_snapshot) {
   ifstream fsnapshot(fn_snapshot.c_str(), ios::binary);
   int aval_id;
   fsnapshot.read(reinterpret_cast<char*> (&aval_id), sizeof(int));
   Cluster::aval_id = aval_id;
   int cluster_counter;
   fsnapshot.read(reinterpret_cast<char*> (&cluster_counter), sizeof(int));
   Cluster::counter = cluster_counter;
   int aval_data;
   fsnapshot.read(reinterpret_cast<char*> (&aval_data), sizeof(int));
   Segment::counter = aval_data;
   int cluster_num;
   fsnapshot.read(reinterpret_cast<char*> (&cluster_num), sizeof(int));
   for (int i_cluster = 0; i_cluster < cluster_num; ++i_cluster) {
      int id;
      fsnapshot.read(reinterpret_cast<char*> (&id), sizeof(int));
      int state_num;
      fsnapshot.read(reinterpret_cast<char*> (&state_num), sizeof(int));
      int mixture_num;
      fsnapshot.read(reinterpret_cast<char*> (&mixture_num), sizeof(int));
      int vector_dim;
      fsnapshot.read(reinterpret_cast<char*> (&vector_dim), sizeof(int));

      Cluster* new_cluster = new Cluster(state_num, mixture_num, vector_dim);
      new_cluster -> set_cluster_id(id);

      float trans[state_num];
      fsnapshot.read(reinterpret_cast<char*> (trans), \
        sizeof(float) * state_num);
      new_cluster -> update_trans(trans); 
      for (int i_state = 0; i_state < state_num; ++i_state) {
         cout << "loading GMM " << i_state << endl;
         Gmm new_gmm(mixture_num, vector_dim);
         if (!load_gmm_from_file(fsnapshot, new_gmm, false)) {
            return false;
         }
         new_cluster -> update_emission(new_gmm, i_state);
      }
      cout << "Loading members..." << endl;
      int member_num;
      fsnapshot.read(reinterpret_cast<char*> (&member_num), sizeof(int));
      for (int i_member = 0; i_member < member_num; ++i_member) {
         int tag_length;
         fsnapshot.read(reinterpret_cast<char*> (&tag_length), sizeof(int));
         char tag_c[tag_length];
         fsnapshot.read(reinterpret_cast<char*> (tag_c), tag_length);
         string tag(tag_c);
         int start_frame;
         fsnapshot.read(reinterpret_cast<char*> (&start_frame), sizeof(int));
         int end_frame;
         fsnapshot.read(reinterpret_cast<char*> (&end_frame), sizeof(int)); 
         int cluster_id;
         fsnapshot.read(reinterpret_cast<char*> (&cluster_id), sizeof(int));
         int frame_num;
         fsnapshot.read(reinterpret_cast<char*> (&frame_num), sizeof(int));
         int hidden_states[frame_num];
         fsnapshot.read(reinterpret_cast<char*> (hidden_states), \
           sizeof(int) * frame_num);
         int mixture_id[frame_num];
         fsnapshot.read(reinterpret_cast<char*> (mixture_id), \
           sizeof(int) * frame_num);
         Segment* new_segment = new Segment(frame_num, vector_dim, \
           tag, start_frame, end_frame);
         new_segment -> set_cluster_id(cluster_id);
         new_segment -> set_hidden_states(hidden_states);
         new_segment -> set_mixture_id(mixture_id);
         float** frame_data = new float* [frame_num];
         for (int i_frame = 0; i_frame < frame_num; ++i_frame) {
            frame_data[i_frame] = new float[vector_dim];
            fsnapshot.read(reinterpret_cast<char*> (frame_data[i_frame]), \
              sizeof(float) * vector_dim);
         }
         new_segment -> set_frame_data(frame_data);
         for (int i_frame = 0; i_frame < frame_num; ++i_frame) {
            delete[] frame_data[i_frame];
         }
         delete[] frame_data;
         new_cluster -> append_member(new_segment);
         segments.push_back(new_segment);
      }
      clusters.push_back(new_cluster);
   }
   fsnapshot.close();
   return true;
}
*/
Manager::~Manager() {
   delete[] data; 
   vector<Bound*>::iterator iter_bounds;
   iter_bounds = bounds.begin();
   for (;iter_bounds != bounds.end(); ++iter_bounds) {
      delete *iter_bounds;
   }
   list<Segment*>::iterator iter_segments;
   iter_segments = segments.begin();
   for (;iter_segments != segments.end(); ++iter_segments) {
      delete *iter_segments;
   }
   segments.clear();
   vector<Cluster*>::iterator iter_clusters;
   iter_clusters = clusters.begin();
   for (; iter_clusters != clusters.end(); ++iter_clusters) {
      delete *iter_clusters;
   }
   clusters.clear();
}

