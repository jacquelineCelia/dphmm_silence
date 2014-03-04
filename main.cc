/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./main.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "manager.h"

void print_usage() {
   cout << "./gibbs -l data_list -p gmm_prior -c config_list \
     -n gibbs_iter -d result_dir -b base -sil silence_model -s snapshot_file -l cluster_id" << endl;
}

int main(int argc, char* argv[]) {
   if (argc != 19 and argc != 15) {
      print_usage();
      return 1;
   }
   string data_list = argv[2];
   string gmm_prior = argv[4];
   string config_list = argv[6];
   int gibbs_iter = atoi(argv[8]);
   string result_dir = argv[10];
   int base = atoi(argv[12]);
   string fn_sil = argv[14];
   string snapshot = "";
   string fn_cluster_id;
   if (argc == 19) {
      snapshot = argv[16];
      fn_cluster_id = argv[18];
   }
   Manager projectManager;
   if (!projectManager.load_config(config_list)) {
      cout << "Configuration file seems bad. Check " 
           << config_list << " to make sure." << endl;
      return -1;
   }
   else {
      cout << "Configuration file loaded successfully..." << endl;
   }
   if (!projectManager.load_silence_model(fn_sil)) {
       cout << "Silence model seems bad. Check "
           << fn_sil << endl;
       return -1;
   }
   else {
       cout << "Silence model loaded successfully..." << endl;
   }
   if (!projectManager.load_gmm(gmm_prior)) {
      cout << "Gmm file seems bad. Check " 
           << gmm_prior << " to make sure." << endl;
      return -1;
   }
   else {
      cout << "Gmm file loaded successfully..." << endl;
   }
   projectManager.init_sampler();
   cout << "Sampler initialized successfully..." << endl;
   if (snapshot != "") {
      cout << "Loading snapshot..." << endl;
      if (!projectManager.load_snapshot(snapshot, \
           fn_cluster_id, data_list, base)) {
         cout << "snapshot file seems bad. Check "
           << snapshot << "." << endl;
      }
      else {
         cout << "Snapshot loaded successfully..." << endl;
      }
   }
   else {
      if (!projectManager.load_bounds(data_list, base)) {
         cout << "data list file seems bad. Check " 
            << data_list << " to make sure." << endl;
         return -1;
      }
      else {
         cout << "Data loaded successfully..." << endl;
      }
   }
   projectManager.gibbs_sampling(gibbs_iter, result_dir);
   return 0;
}
