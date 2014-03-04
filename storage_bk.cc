/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./storage_bk.cc
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

Storage::Storage() : UNIT(0){
}

void Storage::init() {
   batch_size = 100000000;
   unit = new float [batch_size];
   unsigned int SEED = time(0);
   vslNewStream(&stream, BRNG,  SEED);
   unit_index = 0;
}

float Storage::get_random_samples(const int TYPE) {
   if (TYPE == UNIT) {
      if (!(unit_index % batch_size)) {
         sample_batch(UNIT);
         unit_index = 0;
      }
      float* sample = &unit[unit_index];
      ++unit_index;
      return sample; 
   }
   else {
         cout << "Wrong sample type access " << TYPE << endl;
         return NULL;
   }
}

void Storage::sample_batch(const int TYPE) {
   if (TYPE == UNIT) {
      vsRngUniform(UNIFORM_METHOD, stream, batch_size, unit, 0, 1);
   }
   else {
      cout << "Wrong sample type " << TYPE << endl;
   }
}

Storage::~Storage() {
   /*
   for (int i = 0; i < dim; ++i) {
      delete[] mean[i];
      delete[] pre[i];
   }
   */
   // delete[] mean;
   // delete[] pre;
   delete[] unit;
   // delete[] emit;
   vslDeleteStream(&stream);
}
