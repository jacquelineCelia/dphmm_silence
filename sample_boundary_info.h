/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./sample_boundary_info.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef SAMPLE_BOUNDARY_INFO_H
#define SAMPLE_BOUNDARY_INFO_H

#include "cluster.h"

class SampleBoundInfo {
   public:
      SampleBoundInfo();
      ~SampleBoundInfo() {};
      void set_boundary_decision(unsigned int ); 
      void set_c_h0(Cluster* s) {c_h0 = s;}
      void set_c_h1_l(Cluster* s) {c_h1_l = s;}
      void set_c_h1_r(Cluster* s) {c_h1_r = s;}
      Cluster* get_c_h0() {return c_h0;}
      Cluster* get_c_h1_l() {return c_h1_l;}
      Cluster* get_c_h1_r() {return c_h1_r;}
      int get_boundary_decision() {return boundary_decision;}
   private:
      Cluster* c_h0;
      Cluster* c_h1_l;
      Cluster* c_h1_r;
      int boundary_decision;
};

#endif
