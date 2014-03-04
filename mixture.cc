/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./mixture.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include <mkl_cblas.h>
#include <mkl_vml.h>

#include "mixture.h"

using namespace std;

Mixture::Mixture() {
   weight = 0;
   det = 0;
   likelihood = NULL;
   mean = NULL;
   var = NULL;
   precompute_status = false;
}

const Mixture& Mixture::operator= (const Mixture& source) {
   if (this == &source) {
      return *this;
   }
   if (mean != NULL) {
      delete[] mean;
   }
   if (var != NULL) {
      delete[] var;
   }
   if (likelihood != NULL) {
      delete[] likelihood;
   }
   likelihood = NULL;
   precompute_status = false;
   det = source.get_det();
   weight = source.get_weight();
   dim = source.get_dim();
   mean = new float[dim];
   var = new float[dim];
   memcpy(mean, source.get_mean(), sizeof(float) * dim); 
   memcpy(var, source.get_var(), sizeof(float) * dim);
   return *this;
}

Mixture::Mixture(const Mixture& source) {
   weight = source.get_weight();
   det = source.get_det();
   precompute_status = false;
   dim = source.get_dim();
   mean = new float[dim];
   var = new float[dim];
   likelihood = NULL;
   memcpy(mean, source.get_mean(), sizeof(float) * dim); 
   memcpy(var, source.get_var(), sizeof(float) * dim);
}

// initialize by assigning values to weight and dim
// dynamically allocate space to mean and var
// initialize the values of mean and var to all 0
// this is convenient for the case of gmm_count 
// (couting update info from samples)
Mixture::Mixture(int vec_dim) {
   likelihood = NULL;
   precompute_status = false;
   weight = 0;
   det = 0;
   dim = vec_dim;
   mean = new float[dim];
   var = new float[dim];
   for (int i = 0; i < dim; ++i) {
      mean[i] = 0.0;
      var[i] = 0.0;
   }
}

void Mixture::init(int vec_dim) {
   likelihood = NULL;
   precompute_status = false;
   weight = 0;
   dim = vec_dim;
   mean = new float[dim];
   var = new float[dim];
   for (int i = 0; i < dim; ++i) {
      mean[i] = 0.0;
      var[i] = 0.0;
   }
}

// update the mean of this mixture from reading new_mean
void Mixture::update_mean(const float* const new_mean) {
   memcpy(mean, new_mean, sizeof(float) * dim );
}

// update the var of this mixture from reading new_var
void Mixture::update_var(const float* const new_var) {
   memcpy(var, new_var, sizeof(float) * dim );
}

// update the weight of this mixture from reading new_weight
void Mixture::update_weight(const float new_weight) {
   weight = new_weight;
}

void Mixture::update_det() {
   det = 0;
   for (int i = 0; i < dim; ++i) {
      det += log(var[i]);
   }
   det *= 0.5;
   det -= (dim / 2.0) * log(2*3.1415926);
}

// count update info from samples for mean
void Mixture::add_sample_count_mean(const float* const count, const float w) {
   for (int i = 0; i < dim; ++i) {
      mean[i] += w * count[i];
   }
}

void Mixture::sub_sample_count_mean(const float* const count, const float w) {
   for (int i = 0; i < dim; ++i) {
      mean[i] -= w * count[i];
   }
}

// count update info from samples for weight
void Mixture::add_sample_count_weight(const float w) {
   weight += w; 
}

void Mixture::sub_sample_count_weight(const float w) {
   weight -= w; 
}

void Mixture::set_precompute_status(const bool new_status) {
   precompute_status = new_status;
}

void Mixture::set_weight(const float s_w) {
   weight = s_w;
}

void Mixture::set_det(const float s_det) {
   det = s_det;
}

void Mixture::set_mean(const float* s_mean) {
   memcpy(mean, s_mean, sizeof(float) * dim);
}

void Mixture::set_var(const float* s_var) {
   memcpy(var, s_var, sizeof(float) * dim);
}

// count update info from samples for var
// need to double check this formula
void Mixture::add_sample_count_var(const float* const count, const float w) {
   for (int i = 0; i < dim; ++i) {
      var[i] += w * w * count[i] * count[i];
   }
}

void Mixture::sub_sample_count_var(const float* const count, const float w) {
   for (int i = 0; i < dim; ++i) {
      var[i] -= w * w * count[i] * count[i];
   }
}

double Mixture::compute_mixture_likelihood(const int index) {
   return likelihood[index];
}

void Mixture::precompute(int total_frame_num, \
                                            const float** data) {
   if (likelihood == NULL) {
      likelihood = new float[total_frame_num];
   }
   float constant = det + weight;
   float* copy_data = new float[total_frame_num * dim];
   float all_ones[total_frame_num];
   for(int i = 0; i < total_frame_num; ++i) {
      likelihood[i] = constant; 
      all_ones[i] = 1.0;
      memcpy((copy_data + i * dim), data[i], sizeof(float) * dim);
   }
   MKL_INT         m, n, k;
   m = total_frame_num;
   k = 1;
   n = dim;
   MKL_INT         lda, ldb, ldc;
   float           alpha, beta;
   alpha = -1.0;
   beta = 1.0;
   CBLAS_ORDER     order = CblasRowMajor;
   CBLAS_TRANSPOSE transA, transB;
   transA = CblasNoTrans; 
   transB = CblasNoTrans;
   lda = 1;
   ldb = dim;
   ldc = dim;
   cblas_sgemm(order, transA, transB, m, n, k, alpha, all_ones, lda, \
                                    mean, ldb, beta, copy_data, ldc);
   vsMul(total_frame_num * dim, copy_data, copy_data, copy_data);
   alpha = -0.5;
   MKL_INT icnx = 1, icny = 1;
   cblas_sgemv(order, transA, m, n, alpha, copy_data, n, var, icnx, \
     beta, likelihood, icny); 
   delete[] copy_data;
}

// compute p(data|mixture) 
// likelihood is computed in log.
double Mixture::compute_mixture_likelihood(const float* const data) {
   // P(x|mean, cov) = 
   // 1/((2pi)^(d/2)*det^(1/2) exp(-((x-u) sigma^-1 (x-u))/2)
   double likelihood = 0.0; 
   double exponet = 0.0;
   for(int i = 0; i < dim; ++i) {
      exponet += ((data[i] - mean[i]) * (data[i] - mean[i]) * var[i]);
   }
   exponet /= -2;
   likelihood += (det + exponet);
   return likelihood;
}

void Mixture::show_parameters() const {
   cout << "weight is " << weight << endl;
   for (int i = 0; i < dim; ++i) {
      cout << "mean " << i << " is " << mean[i] << endl;
      cout << "var " << i << " is " << var[i] << endl;
   }
}

Mixture::~Mixture() {
   if (likelihood != NULL) {
      delete[] likelihood;
   }
   delete[] mean;
   delete[] var;
}
