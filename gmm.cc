/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./gmm.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <cstring>
#include <mkl_vml.h>
#include <mkl_cblas.h>
#include "gmm.h"

using namespace std;

Gmm::Gmm() {
}

Gmm::Gmm(int mix, int vec_dim) {
   mixture_num = mix;
   dim = vec_dim;
   precompute_status = false;
   likelihoods = NULL;
   for(int i = 0; i < mixture_num; ++i) {
      Mixture new_mixture(dim);
      mixtures.push_back(new_mixture);
   }
}

const Gmm& Gmm::operator= (const Gmm& source) {
   if (this == &source) {
      return *this;
   }
   precompute_status = source.get_precompute_status();
   if (likelihoods != NULL) {
      delete[] likelihoods;
   }
   likelihoods = NULL;
   mixtures.clear();
   mixture_num = source.get_mixture_num();
   dim = source.get_dim();
   for (int i = 0; i < mixture_num; ++i) {
      Mixture new_mixture(source.copy_mixture(i));
      mixtures.push_back(new_mixture);
   }
   return *this;
}

Gmm::Gmm(const Gmm& source) {
   precompute_status = source.get_precompute_status();
   likelihoods = NULL;
   mixture_num = source.get_mixture_num();
   dim = source.get_dim();
   for (int i = 0; i < mixture_num; ++i) {
      Mixture new_mixture(source.copy_mixture(i));
      mixtures.push_back(new_mixture);
   }
}

void Gmm::init(int mix, int vec_dim) {
   precompute_status = false;
   likelihoods = NULL;
   mixture_num = mix;
   dim = vec_dim;
   for(int i = 0; i < mixture_num; ++i) {
      Mixture new_mixture(dim);
      mixtures.push_back(new_mixture);
   }
}

void Gmm::set_precompute_status(const bool new_status) {
   precompute_status = new_status;
   for(int i = 0; i < mixture_num; ++i) {
      mixtures[i].set_precompute_status(new_status);
   }
}

double Gmm::compute_likelihood(const int index) {
   return likelihoods[index];
   /*
   double likelihood = 0.0; 
   double log_arr[mixture_num];
   for (int i = 0; i < mixture_num; ++i) {
      log_arr[i] = mixtures[i].compute_mixture_likelihood(index);
   }
   vector<double> log_reg(log_arr, log_arr + mixture_num);
   likelihood = calculator.sum_logs(log_reg);
   return likelihood; */
}

// Every thing is in log.
double Gmm::compute_likelihood(const float* const data) {
   double log_arr[mixture_num];
   for (int i = 0; i < mixture_num; ++i) {
      log_arr[i] = mixtures[i].get_weight() + \
        mixtures[i].compute_mixture_likelihood(data);
   }
   if (mixture_num == 1) {
      return log_arr[0];
   }
   else {
      return calculator.sum_logs(log_arr, mixture_num);
   }
}

void Gmm::precompute(int total, const float** data) {
   for(int i = 0; i < mixture_num; ++i) {
      mixtures[i].precompute(total, data);
   }
   if (likelihoods == NULL) {
      likelihoods = new float[total];
   }
   memcpy(likelihoods, mixtures[0].get_likelihood(), sizeof(float) * total);
   if (mixture_num > 1) {
      float* t_scores = new float[total * (mixture_num - 1)];
      MKL_INT incy = mixture_num - 1;
      for (int i = 1; i < mixture_num; ++i) {
         vsUnpackI(total, mixtures[i].get_likelihood(), (t_scores + i - 1), incy); 
      }
      for (int i = 0; i < total; ++i) {
         for (int j = 0; j < mixture_num - 1; ++j) {
            if (t_scores[i * (mixture_num - 1) + j] > likelihoods[i]) {
               float t = likelihoods[i];
               likelihoods[i] = t_scores[i * (mixture_num - 1) + j];
               t_scores[i * (mixture_num - 1) + j] = t;
            }
         }
      }
      float all_ones[mixture_num - 1];
      for (int i = 0; i < mixture_num - 1; ++i) {
         all_ones[i] = 1.0;
      }
      MKL_INT m, n, k;
      m = total;
      k = 1;
      n = mixture_num - 1;
      MKL_INT lda, ldb, ldc;
      float alpha, beta;
      alpha = -1.0;
      beta = 1.0;
      CBLAS_ORDER order = CblasRowMajor;
      CBLAS_TRANSPOSE transA, transB;
      transA = CblasNoTrans;
      transB = CblasNoTrans;
      lda = 1;
      ldb = mixture_num - 1;
      ldc = mixture_num - 1;
      cblas_sgemm(order, transA, transB, m, n, k, alpha, \
        likelihoods, lda, all_ones, ldb, beta, t_scores, ldc);
      vsExp(total * (mixture_num - 1), t_scores, t_scores);
      //to sum the exponentials up
      float sum[total]; 
      MKL_INT inca = mixture_num - 1;
      vsPackI(total, t_scores, inca, sum);
      for (int i = 1; i < mixture_num - 1; ++i) {
         float mixture_likelihood[total];
         vsPackI(total, t_scores + i, inca, mixture_likelihood);
         vsAdd(total, sum, mixture_likelihood, sum);
      }
      vsLog1p(total, sum, sum);
      vsAdd(total, likelihoods, sum, likelihoods);
      delete[] t_scores;
   }
}

vector<double> Gmm::compute_posterior_weight(int index) {
   double posterior_arr[mixture_num];
   for (int i = 0; i < mixture_num; ++i) {
      posterior_arr[i] = mixtures[i].compute_mixture_likelihood(index);
   }
   vector<double> posterior_weight(posterior_arr, posterior_arr + mixture_num);
   return posterior_weight;
}

vector<double> Gmm::compute_posterior_weight(const float* const data) {
   vector<double> posterior_weight;
   for (int i = 0; i < mixture_num; ++i) {
      double likelihood = mixtures[i].compute_mixture_likelihood(data);
      double posterior_i = mixtures[i].get_weight() + likelihood;
      /*
      cout << "weight is " << mixtures[i].get_weight() << endl;
      cout << "likelihood is " << likelihood << endl;
      */
      posterior_weight.push_back(posterior_i);
   }
   return posterior_weight;
}

Mixture& Gmm::get_mixture(int index) {
   return mixtures[index];
}

const Mixture& Gmm::copy_mixture(int index) const {
   return mixtures[index];
}

void Gmm::set_mixture_weight(const int m, const float w) {
   mixtures[m].set_weight(w);
}

void Gmm::set_mixture_det(const int m, const float det) {
   mixtures[m].set_det(det);
}

void Gmm::set_mixture_mean(const int m, const float* mean) {
   mixtures[m].set_mean(mean);
}

void Gmm::set_mixture_var(const int m, const float* var) {
   mixtures[m].set_var(var);
}

Gmm::~Gmm() {
   if (likelihoods != NULL) {
      delete[] likelihoods;
   }
}
