#!/bin/bash -l
#
#SBATCH --nodes=4
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -J number_density_seeds
#SBATCH -o ./logs/number_density_seeds.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
########################

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# virtual environment made with venv_scripts/make_cosma_env.sh
venv_name="/cosma/apps/dp004/${USER}/swiftlet_env"
source "${venv_name}/bin/activate"

# one numba thread per MPI rank, so the ranks don't compete for the cores
export SWIFTLET_MAX_THREADS=1

# orientation seeds 0 to nr_seeds - 1, with the box tiles locked into cubes around the observer
nr_seeds=10
orientation_lock="cube"
nr_bins=25
# the redshift range of both simulations
zmin=0.001
zmax=0.15
output_dir=./example_outputs/number_density_seeds
mkdir -p "${output_dir}"

# FLAMINGO L1_m9
mpirun -- python3 -m mpi4py ./examples/snapshot_number_density_comparison.py \
    --box_res="L1000N1800" --sim_name="HYDRO_FIDUCIAL" --zmin=${zmin} --zmax=${zmax} \
    --nr_seeds=${nr_seeds} --orientation_lock="${orientation_lock}" --nr_bins=${nr_bins} --output_dir="${output_dir}"

# COLIBRE L400_m7
mpirun -- python3 -m mpi4py ./examples/snapshot_number_density_comparison.py \
    --box_res="L0400N3008" --sim_name="Thermal" --zmin=${zmin} --zmax=${zmax} \
    --nr_seeds=${nr_seeds} --orientation_lock="${orientation_lock}" --nr_bins=${nr_bins} --output_dir="${output_dir}"

# the two side by side, the files named with the redshifts to 3 decimal places
z_label=$(printf "z_%.3f_%.3f" ${zmin} ${zmax})
python3 ./examples/snapshot_number_density_comparison.py --plot_files \
    "${output_dir}/dark_matter_number_density_seeds.L1000N1800.HYDRO_FIDUCIAL.${z_label}.txt" \
    "${output_dir}/dark_matter_number_density_seeds.L0400N3008.Thermal.${z_label}.txt"

