
#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -J snapshot_close_repeat_check
#SBATCH -o ./logs/snapshot_close_repeat_check.%j.out
#SBATCH -p cosma8-serial
#SBATCH -A dp004
#SBATCH -t 02:00:00
########################

# This example was written for use on COSMA:
# CHANGE ALL PATHS AS NEEDED

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# virtual environment made with venv_scripts/make_cosma_env.sh
source "/cosma/apps/dp004/${USER}/swiftlet_env/bin/activate"

# use the cores given to the job for the numba threads
export SWIFTLET_MAX_THREADS=${SLURM_CPUS_PER_TASK}


# the simulations to check
box_res="L1000N1800"
sim_name="HYDRO_FIDUCIAL"

# redshift range to check 
zmin=0.001
zmax=0.5


max_snapshot_difference=1 #allowed max snapshot difference between cell copies

orientation_seed=3 #orientation seed to check for each table

tables=("default" "cube_quarter_spacing" "sphere_quarter_spacing") # check each of these installed tables

planned_beam_vector=(1.0 1.0 1.0) # direction of the planned beam to overlay onto the map


# iterate through each orientation table

for table_name in "${tables[@]}"; do
    read -r table_file <<< "${table_name}"
    output_dir=./example_outputs/close_repeat_check/${box_res}/${sim_name}/${table_file}
    mkdir -p "${output_dir}"
    python3 -u ./examples/snapshot_close_repeat_map.py \
        --box_res="${box_res}" \
        --sim_name="${sim_name}" \
        --zmin=${zmin} \
        --zmax=${zmax} \
        --orientation_seed=${orientation_seed} \
        --output_dir="${output_dir}" \
        --max_snapshot_difference=${max_snapshot_difference} \
        --orientation_table="${table_file}" \
        --angle_deg=1.0 \
        --planned_beam_vector "${planned_beam_vector[@]}" \
        --planned_ang_radius_deg=15 
    
done
