/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./gmm.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef GMM_H
#define GMM_H

#include <vector>
#include "mixture.h"
#include "calculator.h"

using namespace std;

class Gmm {
   public:
      Gmm();
      Gmm(int, int);
      Gmm(const Gmm&);
      const Gmm& operator= (const Gmm&);
      void init(int, int);
      void set_mixture_weight(const int, const float);
      void set_mixture_det(const int, const float);
      void set_mixture_mean(const int, const float*);
      void set_mixture_var(const int, const float*);
      Mixture& get_mixture(int);
      const Mixture& copy_mixture(int) const;
      int get_mixture_num() const {return mixture_num;}
      int get_dim() const {return dim;}
      bool get_precompute_status() const {return precompute_status;}
      double compute_likelihood(const float* const); 
      double compute_likelihood(const int);
      vector<double> compute_posterior_weight(const float* const);
      vector<double> compute_posterior_weight(int);
      void precompute(int, const float**);
      void set_precompute_status(const bool);
      ~Gmm();
   private:
      int mixture_num;
      int dim; 
      vector<Mixture> mixtures;
      Calculator calculator;
      bool precompute_status;
      float* likelihoods;
};

#endif
