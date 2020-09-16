#!/bin/bash
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
# Simple script to extract {{x},G} data from MLL, and create files with format that can be analized separately by MLL_calculate_a.py and MLL_calculate_S.py
#################################################################################
### START CUSTOMIZABLE DATA ###
data_file_name='data_x_G'		# prefix of data files taht will be created
t1=200								# number of t1 points
t0=15 								# number of t0 points (ignored)
Nspf=100								# number of datasets
adv=(10 20 30 40 50 60 70 80 90 100)		# array with values of adventurousness values
#### END CUSTOMIZABLE DATA ####
N=$((${t0}+${t1}))
# Loop for all datasets
for l in `seq 0 $((Nspf-1))`
do
   counter=0
	# Loop for all adventurousness values
   for i in `seq ${#adv[@]}`
   do 
		a=${adv[counter]}
    	grep "timestep " log_grid_l_${l}.log | awk 'NR>'"${N}"'*'"${counter}"'&&NR<='"${N}"'*'"${counter}"'+'"${N}"' {$1=$2=""; print $0}'  | sed -e 's/^[[:space:]]*//' > ${data_file_name}_${a}_${l}.dat
      counter=$((counter+1))
   done
done
