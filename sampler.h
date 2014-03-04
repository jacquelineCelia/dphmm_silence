/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./sampler.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <mkl_vsl.h>
/*
#include <boost/random/uniform_real.hpp> // for normal_distribution.
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>
*/
#include "segment.h"
#include "cluster.h"
#include "gmm.h"
#include "sample_boundary_info.h"
#include "calculator.h"
#include "storage.h"

using namespace std;
// using namespace boost;
// using namespace boost::random;

// typedef boost::mt19937 base_generator_type;

class Sampler {
   public:
      Sampler();
      static int seed;
      static float annealing;
      static int offset;
      // set up priors for the model
      void init_prior(const int, \
        const int, const int, \
        const float, const float, \
        const float, const float, \
        const float, const float, \
        const float, Gmm, const float);
      // sample the cluster for each segment
      SampleBoundInfo sample_h0_h1(Segment*, Segment*, Segment*, vector<Cluster*>&);
      void is_boundary(Segment*, Segment*, Segment*, list<Segment*>& , \
        vector<Cluster*>&, SampleBoundInfo&, vector<Bound*>::iterator);
      void is_not_boundary(Segment*, Segment*, Segment*, list<Segment*>& , \
        vector<Cluster*>&, SampleBoundInfo&, vector<Bound*>::iterator);
      void sample_more_than_cluster(Segment&, \
              vector<Cluster*>&, Cluster*);
      Cluster* sample_just_cluster(Segment&, vector<Cluster*>&, bool pick_new=false);
      Cluster* sample_cluster_from_others(vector<Cluster*>&);
      // sample the hidden state squence
      void sample_hidden_states(Segment&, Cluster*);
      // sample the mixture_id for each data frame
      void sample_mixture_id(Segment&, Cluster*);
      // sample cluster parameters
      void sample_hmm_parameters(Cluster&);
      void sample_pseudo_state_seq(Cluster*, int*, int);
      int sample_index_from_log_distribution(vector<double>);
      int sample_index_from_distribution(vector<double>);
      bool decluster(Segment*, vector<Cluster*>&);
      bool clean_cluster(Segment*, vector<Cluster*>&);
      bool sample_boundary(Bound*);
      bool sample_boundary(vector<Bound*>::iterator, \
        list<Segment*>&, vector<Cluster*>&);
      void encluster(Segment&, vector<Cluster*>&, Cluster*);
      // sample from beta distribution
      float sample_from_beta(const int*);
      // sample from unit distribution
      float sample_from_unit();
      // sample from a gaussian
      const float* sample_from_gaussian(int, const float*, \
        const float*, const float);
      // sample from a diagonal covariance 
      const float* sample_from_gamma(int, const float*, \
        const float*, const float);
      float update_gamma_rate(float, float, float, float, float);
      // const float* sample_from_gamma_for_weight(const float*);
      const float* sample_from_gamma_for_multidim(const float*, int, float*);
      void sample_trans(vector<vector<float> >&, vector<vector<float> >&);
      void set_precompute_status(Cluster*, const bool);
      void precompute(Cluster*, int, const float**);
      Cluster* sample_cluster_from_base();
      // Cluster* sample_from_hash_for_cluster(Segment*, vector<Cluster*>&);
      // get DP prior
      double get_dp_prior(Cluster*) const;
      bool hidden_state_valid_check(const int*, const int);
      ~Sampler();
   private:
      int dim; 
      int mixture_num;
      int state_num;
      float dp_alpha;
      float beta_alpha;
      float beta_beta;
      float gamma_shape;
      float gamma_weight_alpha;
      float gamma_trans_alpha;
      float norm_kappa;
      vector<double> boundary_prior;
      vector<double> boundary_prior_log;
      Gmm gmm;
      Calculator calculator;
      // base_generator_type generator;
      VSLStreamStatePtr stream; 
      Storage storage;
      int UNIT;
      int MEAN;
      int PRE;
      int EMIT;
};

#endif
