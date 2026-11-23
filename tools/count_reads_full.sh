#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH -J count_reads_all_locks
#SBATCH -o ./logs/count_reads_all_locks.%j.out
#SBATCH -p cosma8-serial
#SBATCH -A dp004
#SBATCH -t 12:00:00

# SLURM job: count what placing particles reads (tools/count_reads.py) for each orientation lock, first in serial,
# then on MPI ranks, then compare reading whole files with reading in batches (max_particles_per_read): peak memory
# against time. One markdown report per run goes into an output directory, with a summary of them all.
#
# CHANGE THE #SBATCH LINES AND THE DEFAULTS BELOW AS NEEDED. 
#
#The ./logs directory must exist before submitting.
#
# Usage (options after the script are passed to it by sbatch):
#     mkdir -p logs && sbatch count_reads_all_locks.slurm [options]
#
# Options:
#     -r DIR      the SWIFTLET checkout to measure (default: the directory the job was submitted from)
#     -o DIR      output directory for the reports, relative to the checkout (default: ./read_counts_<job id>)
#     -n RANKS    number of MPI ranks (default: the job's tasks, SLURM_NTASKS); 0 skips the MPI runs
#     -l LOCKS    the locks, quoted (default: "none cube sphere")
#     -s SOURCE   where the snapshots come from, quoted (default: "--box_res L1000N1800 --sim_name HYDRO_FIDUCIAL";
#                 "--fake" for the fake simulation of the tests)
#     -a ARGS     the other count_reads.py arguments, quoted (default: "--ptype PartType1 --beam 1 2 3 2.0
#                 --beam 0 0 1 10.0 --all_sky --z 0.1 0.15 --memory")
#     -b BATCHES  batch sizes to compare with whole files, quoted (default: "1000000 10000000"); "" skips the
#                 comparison, which is also skipped on code without max_particles_per_read
#     -e ACTIVATE the virtual environment's activate script (default:
#                 /cosma/apps/dp004/$USER/swiftlet_env/bin/activate, from venv_scripts/make_cosma_env.sh); "" for none
#     -p PYTHON   Python to run (default: python3)
#     -m MPIEXEC  MPI launcher (default: mpirun)
#     -h          show this help
#
# Examples:
#     cd swiftlet_after  && mkdir -p logs && sbatch count_reads_all_locks.slurm -o ../after_flamingo
#     sbatch --ntasks=32 count_reads_all_locks.slurm -o ../after_flamingo_32 -b "10000000"   # 32 ranks, one batch size
#     sbatch -t 01:00:00 count_reads_all_locks.slurm -s "--fake" -b "500"                    # the fake simulation
#
# Reports: <DIR>/serial_<lock>.md and <DIR>/mpi<RANKS>_<lock>.md; for the batches, <DIR>/batches_<serial or
# mpi<RANKS>>_<lock>_memory.md (peak memory, measured with tracemalloc) and ..._time.md (times, measured without it,
# as tracemalloc slows a run down). Each run's output is in <DIR>/logs/, and <DIR>/summary.md has every table and the
# comparison of whole files against batches. The serial runs use one core of the job, the MPI runs all its tasks.

set -euo pipefail

checkout="/cosma8/data/do012/dc-mcdo1/ExtraSWIFTLightcones"
out_dir="./read_counts_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
ranks="${SLURM_NTASKS:-3}"
locks="none cube sphere"
source_args="--box_res L1000N1800 --sim_name HYDRO_FIDUCIAL"
#args="--ptype PartType1 --beam 1 2 3 2.0 --beam 0 0 1 10.0 --all_sky --z 0.05 0.1 --memory"
args="--ptype PartType1 --beam 1 2 3 2.5 --beam 1 2 3 10.0 --beam 0 0 1 30.0 --z 0.05 0.15 --memory"
batches="none 1000000 10000000"
digits=3
#activate="/cosma/apps/dp004/${USER:-$(id -un)}/swiftlet_env/bin/activate"
activate="/cosma/apps/do012/${USER:-$(id -un)}/swiftlet_env/bin/activate"
python="python3"
mpiexec="mpirun"

# the help: this file's comment lines (SLURM runs a copy of the script, which has them too)
usage() { sed -n '/^# SLURM job/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

#while getopts "o:n:l:s:a:p:m:b:d:h" option; do
while getopts "r:o:n:l:s:a:b:e:p:m:d:h" option; do

    case "$option" in
        r) checkout="$OPTARG" ;;
        o) out_dir="$OPTARG" ;;
        n) ranks="$OPTARG" ;;
        l) locks="$OPTARG" ;;
        s) source_args="$OPTARG" ;;
        a) args="$OPTARG" ;;
        p) python="$OPTARG" ;;
        m) mpiexec="$OPTARG" ;;
        b) batches="$OPTARG" ;;
        d) digits="$OPTARG" ;;
        e) activate="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

# the environment: COSMA's modules (as the examples' job scripts), the virtual environment, one thread per task
if command -v module > /dev/null 2>&1; then
    set +u  # module scripts read unset variables
    module purge
    module load gnu_comp/14.1.0 openmpi/5.0.3
    module load python/3.12.4
    set -u
fi
if [ -n "$activate" ]; then
    if [ ! -f "$activate" ]; then
        echo "error: no virtual environment at $activate (give -e, or -e \"\" to use the current Python)" >&2
        exit 1
    fi
    set +u
    # shellcheck disable=SC1090
    source "$activate"
    set -u
fi
export SWIFTLET_MAX_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$checkout"
if [ ! -f tools/count_reads.py ]; then
    echo "error: $checkout is not a SWIFTLET checkout with tools/count_reads.py (submit from one, or give -r)" >&2
    exit 1
fi
echo "job ${SLURM_JOB_ID:-(not under SLURM)} on $(hostname), $ranks MPI ranks, $(date)"
if [ "$ranks" -gt 0 ] && ! command -v "${mpiexec%% *}" > /dev/null; then
    echo "error: $mpiexec not found (give -m, or -n 0 to skip the MPI runs)" >&2
    exit 1
fi

mkdir -p "$out_dir/logs"
version="$(git rev-parse --short HEAD 2> /dev/null || echo unknown)"
dirty="$(git status --porcelain 2> /dev/null | grep -c . || true)"
echo "SWIFTLET at $version ($dirty changed files) in $(pwd)"
echo "reports in $out_dir"

failed=0

# run <label> <report name> <command...>: one count_reads.py run, its output kept in the log
run() {
    local label="$1" name="$2"
    shift 2
    local start=$SECONDS
    echo -n "$label ... "
    # shellcheck disable=SC2086  # the arguments are split on purpose
    if "$@" --output "$out_dir/$name.md" > "$out_dir/logs/$name.log" 2>&1; then
        echo "done ($((SECONDS - start)) s)"
    else
        echo "FAILED, see $out_dir/logs/$name.log"
        failed=$((failed + 1))
    fi
}

# 1) each lock in serial
for lock in $locks; do
    # shellcheck disable=SC2086
    run "serial, lock $lock" "serial_$lock" "$python" tools/count_reads.py $source_args $args --lock "$lock"
done

# 2) each lock on MPI ranks
if [ "$ranks" -gt 0 ]; then
    for lock in $locks; do
        # shellcheck disable=SC2086
        run "$ranks MPI ranks, lock $lock" "mpi${ranks}_$lock" \
            $mpiexec -n "$ranks" "$python" -m mpi4py tools/count_reads.py $source_args $args --lock "$lock"
    done
fi

# 3) whole files against batches, for each lock, in serial and on MPI: the peak memory (with --memory) and the time
#    (without, as tracemalloc slows the run), each run placing every lightcone whole and in each batch size
args_time="${args//--memory/}"
args_memory="$args_time --memory"
supports_batches() {
    "$python" - << 'PY'
import inspect
import sys

sys.path.insert(0, ".")
from swiftlet.snapshot_lightcone import SnapshotLightcone

sys.exit(0 if "max_particles_per_read" in inspect.signature(SnapshotLightcone.place_snapshot_particles_in_shell).parameters else 1)
PY
}
batch_modes=()
if [ -z "${batches// /}" ]; then
    echo "batches: skipped (-b \"\")"
elif ! supports_batches; then
    echo "batches: skipped, this SWIFTLET has no max_particles_per_read (code from before batches)"
else
    batch_modes=(serial)
    [ "$ranks" -gt 0 ] && batch_modes+=("mpi$ranks")
    for mode in "${batch_modes[@]}"; do
        runner=("$python")
        [ "$mode" != serial ] && runner=($mpiexec -n "$ranks" "$python" -m mpi4py)
        for lock in $locks; do
            for measure in memory time; do
                extra="$args_time"
                [ "$measure" = memory ] && extra="$args_memory"
                # shellcheck disable=SC2086
                run "batches ($measure), $mode, lock $lock" "batches_${mode}_${lock}_$measure" \
                    "${runner[@]}" tools/count_reads.py $source_args $extra --lock "$lock" \
                    --max_particles_per_read none $batches
            done
        done
    done
fi

# one summary of every run: the rows of each report's main table
summary="$out_dir/summary.md"
{
    echo "# Read counts: SWIFTLET $version"
    echo
    echo "$source_args $args"
    reports=()
    for lock in $locks; do reports+=("$out_dir/serial_$lock.md"); done
    if [ "$ranks" -gt 0 ]; then
        for lock in $locks; do reports+=("$out_dir/mpi${ranks}_$lock.md"); done
    fi
    for report in "${reports[@]}"; do
        [ -f "$report" ] || continue
        echo
        echo "## $(basename "$report" .md)"
        echo
        # the main table, with the numbers of particles in scientific notation
        "$python" - "$report" "$digits" << 'PY'
import sys
from pathlib import Path

# the columns that count particles; the others (files, opens, requests, ...) stay as they are
PARTICLES = {"to read", "to place", "rows read", "distinct rows", "rows read again", "particles kept"}
report, digits = Path(sys.argv[1]), int(sys.argv[2])
lines = report.read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith("| lightcone |"))
header = [cell.strip() for cell in lines[start].strip("|").split("|")]
print("| " + " | ".join(name.replace("peak memory [MB]", "peak memory [GB]") for name in header) + " |")
print(lines[start + 1])
for line in lines[start + 2 :]:
    if not line.startswith("|"):
        break
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    cells = [
        f"{float(cell):.{digits}e}" if name in PARTICLES and cell.replace(".", "", 1).isdigit() else cell
        for name, cell in zip(header, cells)
    ]
    # the peak memory, from MB (as the reports give it) to GB
    cells = [
        f"{float(cell) / 1024:.3g}" if name == "peak memory [MB]" and cell else cell
        for name, cell in zip(header, cells)
    ]
    print("| " + " | ".join(cells) + " |")
PY
    done
    if [ "${#batch_modes[@]}" -gt 0 ]; then
        echo
        echo "## Whole files against batches: peak memory and time"
        echo
        echo "Peak memory from the runs with --memory (tracemalloc; on MPI the largest rank's), time from the same"
        echo "runs without it (on MPI the slowest rank's). Changes are against whole files of the same lightcone."
        echo
        "$python" - "$out_dir" "${batch_modes[*]}" "$locks" << 'PY'
import sys
from pathlib import Path


def rows_of(path):
    """The main table of a count_reads.py report: one dict per row, keyed by the header."""
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| lightcone |"))
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(dict(zip(header, (cell.strip() for cell in line.strip("|").split("|")))))
    return rows


def change(value, whole):
    """The change of a value against whole files, in per cent."""
    return "" if not whole else f"{100 * (value - whole) / whole:+.0f}%"


out_dir, modes, locks = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3].split()
print("| run | lock | lightcone | batch | peak memory [GB] | peak change | time [s] | time change | requests |")
print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
for mode in modes:
    for lock in locks:
        memory_file = out_dir / f"batches_{mode}_{lock}_memory.md"
        time_file = out_dir / f"batches_{mode}_{lock}_time.md"
        if not (memory_file.exists() and time_file.exists()):
            continue
        times = {(row["lightcone"], row["batch"]): row for row in rows_of(time_file)}
        whole = {}
        for row in rows_of(memory_file):
            key = (row["lightcone"], row["batch"])
            peak, seconds = float(row["peak memory [MB]"]) / 1024 , float(times[key]["time [s]"])
            if row["batch"] == "whole files":
                whole[row["lightcone"]] = (peak, seconds)
            whole_peak, whole_seconds = whole.get(row["lightcone"], (None, None))
            print(
                f"| {mode} | {lock} | {row['lightcone']} | {row['batch']} | {peak:.3g} | {change(peak, whole_peak)} "
                f"| {seconds:.3f} | {change(seconds, whole_seconds)} | {times[key]['requests']} |"
            )
PY
    fi
} > "$summary"
echo "summary in $summary"

if [ "$failed" -gt 0 ]; then
    echo "$failed run(s) failed" >&2
    exit 1
fi
