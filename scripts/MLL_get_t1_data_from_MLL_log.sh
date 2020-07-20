#!/bin/bash
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
# Simple script to extract {{x},G} t1 data from MLL, and create files with general format that can be analized separately
### START CUSTOMIZABLE DATA ###
data_file_name='data_x_G'		# prefix of data files taht will be created
t1=300								# number of t1 points
t0=30									# number of t0 points (ignored)
Nspf=100								# number of landscapes
adv=(10 20 30 40 50 60 70 80 90 100)		# array with values of adventurousness values
#### END CUSTOMIZABLE DATA ####
# Loop for all landscapes
for l in `seq 0 $((Nspf-1))`
do
   counter=0
	# loop for all adventurousness values
   for i in `seq ${#adv[@]}`
   do 
		a=${adv[counter]}
      #echo ${counter} ${i} ${l} ${a}
    	grep "timestep " log_grid_l_${l}.log | awk '$2>='"${t0}"'{print $0}' | awk 'NR>'"${t1}"'*'"${counter}"'&&NR<='"${t1}"'*'"${counter}"'+'"${t1}"' {print $3,$4,$5}' > ${data_file_name}_${a}_${l}.dat
      counter=$((counter+1))
   done
done
