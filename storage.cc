/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./storage.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <mkl_vsl.h>
#include <ctime>
#include <iostream>
#include <cmath>
#include "storage.h"
#include "mixture.h"

#define BRNG VSL_BRNG_MT19937 
#define GAMMA_METHOD VSL_RNG_METHOD_GAMMA_GNORM
#define UNIFORM_METHOD VSL_RNG_METHOD_UNIFORM_STD
#define GAUSSIAN_METHOD VSL_RNG_METHOD_GAUSSIAN_ICDF

using namespace std;

Storage::Storage():UNIT(0), MEAN(1), PRE(2), EMIT(3) {
}

void Storage::init(const int s_dim, \
                const int s_gamma_shape, \
                const int s_norm_kappa, \
                const float s_emit_gamma, \
                Mixture& s_mixture) {
   batch_size = 10000000;
   dim = s_dim;
   gamma_shape = s_gamma_shape;
   norm_kappa = s_norm_kappa;
   emit_gamma_shape = s_emit_gamma;
   mean = new float* [dim];
   pre = new float* [dim];
   unit = new float [batch_size];
   emit = new float[batch_size];
   for (int i = 0; i < dim; ++i) {
      mean[i] = new float [batch_size];
      pre[i] = new float [batch_size];
   }
   mixture_base = s_mixture;
   unsigned int SEED = time(0);
   vslNewStream(&stream, BRNG,  SEED);
   unit_index = 0;
   mean_index = 0;
   pre_index = 0;
   emit_index = 0;
}

Storage::Storage(const int s_dim, \
                 const int s_gamma_shape, \
                 const int s_norm_kappa, \
                 const float s_emit_gamma, \
                 Mixture& s_mixture) \
                 :UNIT(0), MEAN(1), PRE(2), EMIT(3){
   batch_size = 10000000;
   dim = s_dim;
   gamma_shape = s_gamma_shape;
   norm_kappa = s_norm_kappa;
   emit_gamma_shape = s_emit_gamma;
   mean = new float* [dim];
   pre = new float* [dim];
   unit = new float [batch_size];
   emit = new float [batch_size];
   for (int i = 0; i < dim; ++i) {
      mean[i] = new float [batch_size];
      pre[i] = new float [batch_size];
   }
   mixture_base = s_mixture;
   unsigned int SEED = time(0);
   vslNewStream(&stream, BRNG,  SEED);
   unit_index = 0;
   mean_index = 0;
   pre_index = 0;
   emit_index = 0;
}

void Storage::set_emit_gamma(const float new_gamma) {
   emit_gamma_shape = new_gamma;
   cout << "emit gamma shape " << emit_gamma_shape << endl;
}

float* Storage::get_random_samples(const int TYPE) {
   if (TYPE == UNIT) {
      if (!(unit_index % batch_size)) {
         sample_batch(UNIT);
         unit_index = 0;
      }
      float* sample = &unit[unit_index];
      ++unit_index;
      return sample; 
   }
   else if (TYPE == EMIT) {
      if (!(emit_index % batch_size)) {
         sample_batch(EMIT);
         emit_index = 0;
      }
      float* sample = &emit[emit_index];
      ++emit_index;
      return sample; 
   }
   else if (TYPE == PRE) {
      if (!(pre_index % batch_size)) {
         sample_batch(PRE);
         pre_index = 0;
      }
      float* sample = new float [dim];
      for (int i = 0; i < dim; ++i) {
         sample[i] = pre[i][pre_index];
      }
      ++pre_index;
      return sample;
   }
   else if (TYPE == MEAN) {
      if (!(mean_index % batch_size)) {
         sample_batch(MEAN);
         mean_index = 0;
      }
      float *sample = new float [dim];
      for (int i = 0; i < dim; ++i) {
         sample[i] = mean[i][mean_index];
      }
      ++mean_index;
      return sample;
   }
   else {
         cout << "Wrong sample type access " << TYPE << endl;
         return NULL;
   }
}

void Storage::sample_batch(const int TYPE) {
   const float* gamma_rate = mixture_base.get_var(); 
   const float* prior_mean = mixture_base.get_mean();
   if (TYPE == UNIT) {
      vsRngUniform(UNIFORM_METHOD, stream, batch_size, unit, 0, 1);
   }
   else if (TYPE == EMIT) {
      vsRngGamma(GAMMA_METHOD, stream, batch_size, emit, \
        emit_gamma_shape, 0, 1);
   }
   else if (TYPE == PRE) {
     for (int i = 0; i < dim; ++i) {
        vsRngGamma(GAMMA_METHOD, stream, batch_size, pre[i], \
          gamma_shape, 0, 1 / gamma_rate[i]);
     }
   }
   else if (TYPE == MEAN) {
     for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < batch_size; ++j) {
           float std = sqrt( 1 / (pre[i][j] * norm_kappa));
           vsRngGaussian(GAUSSIAN_METHOD, \
             stream, 1, &mean[i][j], prior_mean[i], std);
        }
     }
   }
   else {
      cout << "Wrong sample type " << TYPE << endl;
   }
}

Storage::~Storage() {
   for (int i = 0; i < dim; ++i) {
      delete[] mean[i];
      delete[] pre[i];
   }
   delete[] mean;
   delete[] pre;
   delete[] unit;
   delete[] emit;
   vslDeleteStream(&stream);
}
