/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./calculator.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <vector>

using namespace std;

class Calculator {
   public:
      Calculator();
      double sum_logs(vector<double>);
      double sum_logs(double*, int);
      double find_log_max(vector<double>);
      ~Calculator();
};

#endif
