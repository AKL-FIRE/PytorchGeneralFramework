#! /bin/bash

PARTITION=("Pose1" "Pose2" "Pose3" "Test")
FLAG="NO"
read -p "Please enter the task name:" TASKNAME
read -p "Please enter the number of gpus:" NUMGPU
echo `seninfo` > search_temp.txt
if [ ${NUMGPU} == "8" ]
then
	for part in ${PARTITION[*]}
	do
		echo "TESTING ${part}..."
		grep -E --color -o "BJ-IDC1-10-10-30-[0-9]{0,3} 0/8 idle ${part}" search_temp.txt
		if [ $? -eq 0 ]
			then echo "Find free machine in ${part}, choose it!"
			FLAG="YES"
			source /mnt/lustre/share/miniconda3/envsetup.sh
			srun -p ${part} --job-name=${TASKNAME} -n1 --gres=gpu:8 --ntasks-per-node=8 python -u main.py
			echo "Your job is done!"
			rm search_temp.txt
			break
		fi
	done
	if [ ${FLAG} == "NO" ]
		then echo "There is no available machine to use."
	fi
else
	MAX_NUM=`expr 8 - ${NUMGPU}`
	for part in ${PARTITION[*]}
	do
		echo "TESTING ${part}..."
		grep -E --color -o "BJ-IDC1-10-10-30-[0-9]{0,3} [0-${MAX_NUM}]/8 mix ${part}|BJ-IDC1-10-10-30-[0-9]{0,3} 0/8 idle ${part}" search_temp.txt
		if [ $? -eq 0 ]
			then echo "Find free machine in ${part}, choose it!"
			FLAG="YES"
			source /mnt/lustre/share/miniconda3/envsetup.sh
			srun -p ${part} --job-name=${TASKNAME} -n1 --gres=gpu:${NUMGPU} --ntasks-per-node=${NUMGPU} python -u main.py
			echo "Your job is done!"
			rm search_temp.txt
			break
		fi
	done
	if [ ${FLAG} == "NO" ]
		then echo "There is no available machine to use."
	fi
fi