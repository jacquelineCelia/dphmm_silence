/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./storage_bk.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef STORAGE_H
#define STORAGE_H

class Storage {
   public:
      Storage();
      void init(); 
      void sample_batch(const int);
      float get_random_samples(const int);
      ~Storage();
   private:
      int batch_size;
      float* unit;
      const int UNIT;
      int unit_index;
      VSLStreamStatePtr stream; 
};

#endif
