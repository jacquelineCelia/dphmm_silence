/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./local_seg_main.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <cstring>
#include "local_seg.h"

using namespace std;

int main(int argc, char* argv[]) {
   string inputfile = argv[1];
   string outputfile = argv[2];

   Local_seg seg(inputfile, outputfile);
   if (!seg.load()) {
      cout << "Bad input file." << endl;
   }
   seg.self_align();
   seg.output_local_min();
   cout << "[" << argv[1] << "] is processed successfully." << endl;
   return 0;
}

