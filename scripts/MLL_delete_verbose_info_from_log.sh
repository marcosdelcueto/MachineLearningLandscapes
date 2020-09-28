#!/bin/bash
first_log=0
last_log=99
fine_format=true
ML='GPR'												# IMPORTANT: update to ML method, so verbose lines are identified properly
adven=( 10 20 30 40 50 60 70 80 90 100 )  # Only used for fine_format=true. Which adventurousness are used in each file
# If fine_format=true: update labels to new nomenclature. Also remove ML verbose info
# If fine_format=false: it will simply remove the verbose results of ML convergence cycles

for i in `seq ${first_log} ${last_log}`
do 
	# Remove unnecessary ML verbose
	sed -e '/## Start: "'${ML}'" function/,+7d' log_grid_l_${i}.log > provi0_${i}

	if [[ "${fine_format}" = true ]]; then
      counter=0
	   for a in "${adven[@]}"
		do
	      # Update heading at N0
	      sed -e '0,/## Start: "explore_landscape" function/ s/## Start: "explore_landscape" function/##################################/' provi0_${i} > provi1_${i}
	      sed -e '0,/^############# $/ s/^############# $/## Start initial random exploration/' provi1_${i} > provi0_${i}
	      sed -e '0,/Start explorer '${counter}'/ s/Start explorer '${counter}'/# New Adventurousness: '${a}'.00/' provi0_${i} > provi1_${i}
	      sed -e '0,/Adventurousness: '${a}'.000000/ s/Adventurousness: '${a}'.000000/# Number of grid points per dimension: 101/' provi1_${i} > provi0_${i}
	      sed -e '0,/^############# $/ s/^############# $/##################################/' provi0_${i} > provi1_${i}
	      sed '0,/Number of points per dimension: 101/{/Number of points per dimension: 101/d;}' provi1_${i} > provi0_${i}
	      sed '0,/Testing w: '${counter}'/{/Testing w: '${counter}'/d;}' provi0_${i} > provi1_${i}
	      sed '0,/## Start random biased exploration/{/## Start random biased exploration/d;}' provi1_${i} > provi0_${i}
      
         # Update heading at N1
	      sed -e '0,/## Start: "explore_landscape" function/ s/## Start: "explore_landscape" function/##############################/' provi0_${i} > provi1_${i}
	      sed -e '0,/^############# $/ s/^############# $/## Start a-weighted exploration/' provi1_${i} > provi0_${i}
	      sed -e '0,/Start explorer '${counter}'/ s/Start explorer '${counter}'/##############################/' provi0_${i} > provi1_${i}
	      sed '0,/Adventurousness: '${a}'.000000/{/Adventurousness: '${a}'.000000/d;}' provi1_${i} > provi0_${i}
	      sed '0,/^############# $/{/^############# $/d;}' provi0_${i} > provi1_${i}
	      sed '0,/Number of points per dimension: 101/{/Number of points per dimension: 101/d;}' provi1_${i} > provi0_${i}
	      sed '0,/Testing w: '${counter}'/{/Testing w: '${counter}'/d;}' provi0_${i} > provi1_${i}
	      mv provi1_${i} provi0_${i}
			counter=$((counter+1))
		done
   fi
	mv provi0_${i} log_grid_l_${i}.log
done
