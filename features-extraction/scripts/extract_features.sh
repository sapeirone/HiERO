#!/bin/bash -x

#SBATCH -p boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o logs/job_%A_%a.out
#SBATCH -e logs/job_%A_%a.err

# set -ex

module load python
module load profile/deeplrn

source ../.env/bin/activate

script=$1; shift
root=$1; shift
output_dir=$1; shift

mkdir -p $output_dir

echo "Executing script $script"
echo "Reading input videos from $root and outputting to $output_dir"
echo "SLURM_ARRAY_TASK_ID: [$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT]"

videos=($root/*.mp4)
echo "There are ${#videos[@]} in the input directory"

subset_size=$((${#videos[@]}/$SLURM_ARRAY_TASK_COUNT))
echo "There are ${subset_size} in each subset."

# Calculate the start and end indices for the current subset
start_index=$((($SLURM_ARRAY_TASK_ID) * $subset_size))
end_index=$(($start_index + $subset_size - 1))

echo task id: $SLURM_ARRAY_TASK_ID

# Loop through the videos in the current subset
for ((i = $start_index; i <= $end_index && i < ${#videos[@]}; i++)); do

	input_file=${videos[$i]}
	video_id=$(basename $input_file)
	video_id=${video_id%.*}

	if [ ! -f $output_dir/$video_id.pt ]; then
		echo $input_file is missing...
		python $script path=$root/$(basename $input_file) out_dir=$output_dir $@
	fi
done
