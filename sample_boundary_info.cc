/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./sample_boundary_info.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include "sample_boundary_info.h"

SampleBoundInfo::SampleBoundInfo() {
   boundary_decision = 0;
}

void SampleBoundInfo::set_boundary_decision(unsigned int s) {
   boundary_decision = s;
}
