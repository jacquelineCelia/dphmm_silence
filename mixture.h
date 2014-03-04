/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./mixture.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef MIXTURE_H
#define MIXTURE_H

using namespace std;

class Mixture {
   public:
      Mixture();
      Mixture(int);
      Mixture(const Mixture&);
      const Mixture& operator= (const Mixture&);
      void init(int);
      void update_mean(const float* const);
      void update_var(const float* const);
      void update_weight(const float);
      void update_det();
      void add_sample_count_mean(const float* const, const float);
      void add_sample_count_weight(const float);
      void add_sample_count_var(const float* const, const float);
      void sub_sample_count_mean(const float* const, const float);
      void sub_sample_count_weight(const float);
      void sub_sample_count_var(const float* const, const float);
      const float* get_mean() const {return mean;}
      const float* get_var() const {return var;}
      float get_det() const {return det;}
      float get_weight() const {return weight;}
      const float* get_likelihood() const {return likelihood;}
      int get_dim() const {return dim;}
      bool get_precompute_status() const {return precompute_status;}
      double compute_mixture_likelihood(const float* const);
      double compute_mixture_likelihood(const int);
      void precompute(int, const float**);
      void set_precompute_status(const bool);
      void show_parameters() const;
      void set_weight(const float);
      void set_det(const float);
      void set_mean(const float*);
      void set_var(const float*);
      ~Mixture();
   private:
      int dim;
      float weight;
      float* mean;
      float* likelihood;
      // it is actually percision
      float* var;
      float det;
      bool precompute_status;
};

#endif
