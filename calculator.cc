/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./calculator.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>

#include "calculator.h"

Calculator::Calculator() {
}

double Calculator::sum_logs(double* log_reg, int len) {
   for (int i = 1; i < len; ++i) {
      if (log_reg[i] > log_reg[0]) {
         double t = log_reg[0];
         log_reg[0] = log_reg[i];
         log_reg[i] = t;
      }
   }
   double sum = 1.0;
   for (int i = 1; i < len; ++i) {
      sum += exp(log_reg[i] - log_reg[0]);
   }
   return (log(sum) + log_reg[0]);
}

double Calculator::sum_logs(vector<double> log_reg) {
   double marginal_max = find_log_max(log_reg);
   double marginal_sum = 0;
   vector<double>::iterator iter;
   for (iter = log_reg.begin(); iter != log_reg.end(); ++iter) {
      marginal_sum += exp((*iter) - marginal_max);
   }
   return (marginal_max + log(marginal_sum));
}

double Calculator::find_log_max(vector<double> log_reg){
   vector<double>::iterator iter = log_reg.begin();
   double max = *iter; 
   iter++;
   for (; iter != log_reg.end(); ++iter) { 
      if (*iter > max) {
         max = *iter;
      }
   }
   return max;
}

Calculator::~Calculator() {
}
