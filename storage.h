/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./storage.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef STORAGE_H
#define STORAGE_H

#include "mixture.h"

class Storage {
   public:
      Storage();
      void init(const int, const int, const int, const float, Mixture&); 
      Storage(const int, const int, const int, const float, Mixture&); 
      void sample_batch(const int);
      void set_emit_gamma(const float);
      float* get_random_samples(const int);
      ~Storage();
   private:
      int batch_size;
      int gamma_shape;
      float emit_gamma_shape;
      int norm_kappa;
      int dim;
      float** mean;
      float** pre;
      float* unit;
      float* emit;
      const int UNIT;
      const int MEAN;
      const int PRE;
      const int EMIT;
      int unit_index;
      int mean_index;
      int pre_index;
      int emit_index;
      Mixture mixture_base;
      VSLStreamStatePtr stream; 
};

#endif
