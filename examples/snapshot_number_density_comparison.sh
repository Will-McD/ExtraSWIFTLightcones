#!/bin/bash -l
#
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16
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
nr_bins=12
# the redshift range of both simulations

output_dir=./example_outputs/number_density_seeds
mkdir -p "${output_dir}"


colibre_zmin=0.001
colibre_zmax=0.08 # volume of lightcone is approx 2.5 times volume of 400 Mpc snapshot box
colibre_nr_shells=15 # split into 15 smaller shells

flamingo_zmin=0.001
flamingo_zmax=0.16 # volume of light cone is approx 1.25 to the volume of 1000 Mpc snapshot box
flamingo_nr_shells=1 # can be done in a single large shell


# COLIBRE L400_m7
mpirun -- python3 -m mpi4py ./examples/snapshot_number_density_comparison.py \
    --box_res="L0400N3008" --sim_name="Thermal" --zmin=${colibre_zmin} --zmax=${colibre_zmax} --nr_shells=${colibre_nr_shells} \
    --nr_seeds=${nr_seeds} --orientation_lock="${orientation_lock}" --nr_bins=${nr_bins} --output_dir="${output_dir}"


# FLAMINGO L1_m9
mpirun -- python3 -m mpi4py ./examples/snapshot_number_density_comparison.py \
    --box_res="L1000N1800" --sim_name="HYDRO_FIDUCIAL" --zmin=${flamingo_zmin} --zmax=${flamingo_zmax} --nr_shells=${flamingo_nr_shells} \
    --nr_seeds=${nr_seeds} --orientation_lock="${orientation_lock}" --nr_bins=${nr_bins} --output_dir="${output_dir}"

# the two side by side, the files named with the redshifts to 3 decimal places
flamingo_z_label=$(printf "z_%.3f_%.3f" ${flamingo_zmin} ${flamingo_zmax})
colibre_z_label=$(printf "z_%.3f_%.3f" ${colibre_zmin} ${colibre_zmax})

python3 ./examples/snapshot_number_density_comparison.py --plot_files \
    "${output_dir}/dark_matter_number_density_seeds.L1000N1800.HYDRO_FIDUCIAL.${flamingo_z_label}.txt" \
    "${output_dir}/dark_matter_number_density_seeds.L0400N3008.Thermal.${colibre_z_label}.txt"

