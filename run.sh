#!/bin/bash

python protein_network_analysis_updated.py trajectory_analysis_files/mdm2.pdb trajectory_analysis_files/mdm2.dcd 25 109 --cov_type=displacement_mean_dot --filtering_mode=original_ec --contact_atoms=cbeta
