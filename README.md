# SWIFTLET: SWIFT Lightcone Extension Tools

Supplementary tools for the post-processing and visualisation of [SWIFT](https://swift.strw.leidenuniv.nl/docs/index.html) lightcones. This module includes tools to generate new lightcones from the SWIFT snapshots, and, project, plot and post process lightcones. It builds on [LightconeIO](https://lightconeio.readthedocs.io/en/latest/#) and [SWIFTsimIO](https://swiftsimio.readthedocs.io/en/latest/), which it requires.

Further information about the FLAMINGO lightcones: https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html

---

<details>
<summary>

## Installation

</summary>

Clone the repository, then either build a new virtual environment with everything needed, or pip install into an environment you already have. 
Both download the FLAMINGO lightcone shell redshift files into the environment and export their paths whenever it is activated.

<details>
<summary><b>Install module and requirements to a new or existing environment</b></summary>

### New virtual environment

---

From anywhere, make a virtual environment with SWIFTLET (in editable mode), the dependencies of the package, [examples](./examples) and tests, and the shell redshift files:

```
git clone https://github.com/Will-McD/ExtraSWIFTLightcones.git
bash ExtraSWIFTLightcones/venv_scripts/make_venv.sh /path/to/environment
source /path/to/environment/bin/activate
```

If no path is given, the environment is made in `ExtraSWIFTLightcones/swiftlet_env`.

If you have acess to COSMA, see [Use with COSMA](#use-with-cosma) to build the virtual environment.

### Existing environment

---

With your environment (venv or conda) activated, install the module and its requirements from the repositories directory before then downloading the shell redshift .txt files:  

```
cd ExtraSWIFTLightcones
pip install -e .              # the module and its requirements
pip install ".[all]"          # the module, its requirements + the requirements of the examples and tests
swiftlet-configure
```

To specifically install the additional requirements for the examples and tests:

```
pip install ".[examples]"     # the requirements of the examples
pip install ".[test]"         # the requirements of the examples
```

The shell redshift .txt files, give the redshift bounds of [each shell within FLAMINGO's lightcones](https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html). 
`swiftlet-configure` downloads the shell redshift .txt files to `<environment>/share/swiftlet/redshifts` and adds `L1_REDSHIFTS_FILENAME` and `L2P8_REDSHIFTS_FILENAME` to the environment's activate script (`bin/activate` for a venv, `etc/conda/activate.d` for conda). 
If `swiftlet-configure` isn't run, the lightcone shell redshifts are downloaded the first time they are needed instead.

These shell redshifts are required for mapping between FLAMINGO's HEALPix maps, constructed in concentric redshift shells, and the corresponding halo lightcones which are constructed per snapshot or for identifying which HEALPix maps exist within a given redshift range. 

The [COLIBRE](https://colibre.strw.leidenuniv.nl/index.html) simulation suite does not have lightcones and as such doesn't have shell redshift files. 

---

</details>

<a id="use-with-cosma"></a>
<details>
<summary><b>Use with COSMA</b></summary>

The following step is only applicable for those with access to [COSMA](https://cosma.readthedocs.io/en/latest/index.html).
On COSMA, build the virtual environment with the pre-built wheels of mpi4py and h5py for COSMA's MPI and parallel HDF5 instead:

```
bash ExtraSWIFTLightcones/venv_scripts/make_cosma_env.sh /path/to/environment
```
</details>



<details>
<summary><b>MPI support</b></summary>

MPI support is not required to generate new lightcones from snapshots with the `SnapshotLightcone` sub classes. However, it is necessary for the more efficient parallel methods of `SnapshotLightcone`, the `BeamProjection` class and for generating binary masks of haloes (`mask_haloes.py`).

MPI support requires mpi4py and an MPI enabled build of h5py. 
The mpi4py package installed from PyPI needs an MPI library when it is run. If your system has none, install one into the environment with `pip install mpich` (or `pip install openmpi`). `swiftlet-configure` doesn't need MPI.


<details>
<summary><b>MPI support with COSMA</b></summary>

The following step is only applicable for those with access to [COSMA](https://cosma.readthedocs.io/en/latest/index.html). See the above information under [Use with COSMA](#use-with-cosma) on building a virtual environment on COSMA. `ExtraSWIFTLightcones/venv_scripts/make_cosma_env.sh` used to build the COSMA environment is already configured use the pre-existing h5py and mpi4py wheels on COSMA.  

</details>

See the instructions for installing LightconeIO with MPI support: [LightconeIO with MPI support](https://lightconeio.readthedocs.io/en/latest/installation.html#with-mpi-support)


</details>

<details>
<summary><b>Number of threads</b></summary>

The threaded parts of SWIFTLET, numba's parallel functions (used in re-orienting the snapshots, and `BeamProjection`) and the healpy alm functions (`map2alm`, `rotate_alm` and `alm2map`), use at most  threads.  
Set to 30 by default, and `SWIFTLET_MAX_THREADS` can never more than the cores the process is allowed to run on (e.g. the cores given to a job or an MPI rank).

`swiftlet-configure` adds `SWIFTLET_MAX_THREADS` to the environment's activate script. To change it:

```
swiftlet-configure --max_threads 16          # saved in the environment, re-activate it to use
export SWIFTLET_MAX_THREADS=16               # for one shell or job script
```

otherwise, within a python script: 
```python
swiftlet.set_max_threads(16)
```

As an example, when building a new environment to set the limit as 16 instead of 30:
```
SWIFTLET_MAX_THREADS=16 bash ExtraSWIFTLightcones/venv_scripts/make_venv.sh /path/to/environment
```

To have no fixed maximum, so that each job (or MPI rank) uses all the cores it is allowed to run on, use `all` instead of a number, e.g. `swiftlet-configure --max_threads all`. 
This option is best used when the cores given to a job set the number of threads, e.g. `--cpus-per-task` in Slurm. 

</details>

<details>
<summary><a name="additional-data"></a><b>Additional Data</b></summary>

To add the shell redshift files to another existing environment:

```
bash ExtraSWIFTLightcones/venv_scripts/shell_redshifts.sh /path/to/environment
```

Additionally, to further speed up the rotation of HEALPix maps, download the HEALPix pixel weights and add their paths to your virtual environment. 

```
cd ./ExtraSWIFTLightcones
bash venv_scripts/healpix_pixel_weights.sh
```

</details>

</details>

---

<details>
<summary>

## Generating Lightcones from SWIFT Snapshots

</summary>

<details>
<summary><b>Overview</b></summary>

The `SnapshotLightcone` classes build new lightcones from SWIFT snapshots by tiling periodic copies of the snapshot box around an observer, who sits at the centre of the first box. We refer to these periodic replicas as tiles. 
As the observer's past lightcone propagates through the lattice of tiles, the lightcone is filled with particles (and/or haloes) from the snapshot closest in redshift, as chosen by the comoving distance from the observer.
The snapshot particles (and haloes) placed in the lightcone (`place_snapshot_particles_in_shell`) are returned with the same structure, for each particle type, as given by `lightcone_io.ParticleLightcone` class objects. The returned snapshot-lightcone particles always have the following properties: 
- `Coordinates`         the lightcone-frame x,y,z coordinates of the particles
- `ExpansionFactors`    the particles scale factor ($`\frac{1}{z+1}`$) at their comoving distance from the observer
- `SnapshotNumber`      the number of the snapshot the particle belongs to.

The reading and placing of particles into the lightcone is done first at the cell-level of the SWIFT snapshots. The cells from a given snapshot that overlap with the lightcone's footprint (including a $`1/2`$ cell length buffer) are flagged, then only the particles within the flagged cells are read from the corresponding SWIFT snapshot .hdf5 files. 
Note that passing an MPI communicator (`comm`) reads the snapshot files in parallel.

Alternativley, and to limit peak memory use, the files containing particles within the lightcone's footprint can also be gathered first in a seperate step and then read individually, one at a time, to place particles in the lightcone (`gather_files` and `place_file_in_shell`).  
Moreover, after building (or just applying a first pass at the cell level with `gather_files`), points can be mapped between the snapshot and the lightcone (`Snapshot2Lightcone` and `Lightcone2Snapshot`).

A diagram of the complete workflow of `SnapshotLightcone` and its subclasses is shown in [docs/snapshot_lightcone_flow.md](./docs/snapshot_lightcone_workflow.md). 


Haloes from SOAP catalogues, can be used to both construct halo lightcones and additionally be placed in the same lightcone as the particles. 
In both cases the haloes are treated the same as particles (i.e. point masses) in how they are mapped between the snapshots and lightcone coordinate systems. The halo lightcones are therefore built exactly as the particle lightcones and we include halo equivalents of the aforementioned patrticle methods: 

- `gather_halo_catalogues` (the halo equivalent of `gather_files`): identify all halo catalogues required to fill the lightcones redshift range and build the lightcone without reading any haloes from the catalogues, i.e. what tiles each snapshot's haloes will be placed into. The box and cell metadata are read from one snapshot file. Unlike the particles, the SOAP catalogues are not sorted by cells, so unlike for particles no cells are selected.
- `place_halos_in_shell` (the halo equivalent of `place_file_in_shell`): reads the SOAP catalogues, places every halo into the tiles corresponding to the snapshot (exactly as particles are moved), and only keeps haloes that are within the lightcone. If the lightcone was not pre-built using a dry run (`gather_halo_catalogues`) or from populating it with particles, it runs `gather_halo_catalogues` with the same arguments. The reading of the SOAP catalogues is split into a serial and MPI (parallel) approach as done with the particles. `soap_chunk_size` sets the number of rows in the catalogue read at a time, so only one chunk is in memory at once. With MPI (`comm`), the catalogues are spread over the MPI ranks (as particle files are), and `redistribute_halos` shares the haloes placed evenly between the ranks.

The returned halo lightcones store only the necessary information given by the following properties (same naming scheme as LightconeIO): 
- `Lightcone/HaloCentre`            the lightcone-frame x,y,z coordinates of the haloes
- `Lightcone/SnapshotNumber`        the number of the snapshot the haloes belongs to.
- `InputHalos/HaloCatalogueIndex`   the index of the halo in the original halo finder catalogue that is unique to its snapshot (first halo has index=0).
- `InputHalos/SOAPIndex`            the index of the halo in its snapshots SOAP catalogue. Note, that this is specific to the catalogues used to build the halo lightcone and will be different for full and downsampled SOAP catalogues.
- `Lightcone/ExpansionFactor`       the particles scale factor ($`\frac{1}{z+1}`$) at their comoving distance from the observer

With `Lightcone/SnapshotNumber` and `InputHalos/SOAPIndex` the haloes in the lightcone can all be mapped to the haloes correponsding properties stored within the SOAP catalogue. 

</details>

<details>
<summary><b>Snapshot-to-lightcone methods</b></summary>

There are two subclasses of snapshot lightcones

- `SnapshotBeam` builds a beam about a line of sight (`beam_vector`, a unit vector), with an angular radius up to 60 degrees. Box tiles are added along the line of sight as the lightcone extends past each tile, and transverse to the line of sight once the beam surpasses the width of a tile.
- `SnapshotAllSky` builds a lightcone over the full sky, adding shells of box tiles around the observer's box as the lightcone extends: a single box, then a 3×3×3 cube of boxes, then 5×5×5, and so on.

The box tiles sit on the lattice of whole box sidelengths, so they fill the lightcone exactly once in any direction.

<details>
<summary><b>Place snapshot particles into a past lightcone</b></summary>

For example, to place the gas and dark matter particles of an all-sky shell, reading the snapshot files in parallel with MPI:

```python
from mpi4py import MPI
from swiftlet import SnapshotAllSky

snap_all_sky = SnapshotAllSky(
    simulation_suite="flamingo",
    boxsize_resolution="L1000N1800",
    simulation_name="HYDRO_FIDUCIAL",
    orientation_seed=0,          # reproduces the same box orientations
    orientation_lock="cube",     # None, "cube" or "sphere", see below
)
shell_particles = snap_all_sky.place_snapshot_particles_in_shell(
    lightcone_redshift_range=(0.05, 0.1),
    property_names=["Masses", "Temperatures"],
    particle_types=["PartType0", "PartType1"],
    comm=MPI.COMM_WORLD,         # leave out to read in serial
)

gas = shell_particles["PartType0"]
gas["Coordinates"]       # lightcone positions, with the observer at the origin
gas["ExpansionFactors"]  # expansion factor at each particle's distance from the observer
gas["SnapshotNumber"]    # snapshot each particle was taken from
```

`Coordinates` and `ParticleIDs` (plus `SmoothingLengths` for particles other than dark matter) are always read. To limit memory use in serial, the files can be placed one at a time instead:

```python
numb_files, _ = snap_all_sky.gather_files("PartType1", shell_z=(0.05, 0.1))
for file_number in range(numb_files):
    file_particles = snap_all_sky.place_file_in_shell(file_number, "PartType1", ["Masses"])  # {} if none are kept
```

`SnapshotBeam` is used in the same way, with a `beam_vector` when it is created and the beam's angular radius (`ang_radius_deg`) passed to each of these methods.

</details>

<details>
<summary><b>Place SOAP haloes into a past lightcone</b></summary>

As an example, to place the haloes from the SOAP catalogues in a beam of radius 5 degrees, sharing the catalogues between the MPI ranks:

```python
import numpy as np
from mpi4py import MPI
from swiftlet import SnapshotBeam
from swiftlet.soap_reader import read_soap_selected_rows, comoving

soap_format = "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"

snap_beam = SnapshotBeam(
    simulation_suite="flamingo",
    boxsize_resolution="L1000N1800",
    simulation_name="HYDRO_FIDUCIAL",
    beam_vector=(1., 1., 1.),
    orientation_seed=0,
)

# the dry run (optional): the SOAP catalogue and box tiles of each snapshot, no haloes are read
catalogues = snap_beam.gather_halo_catalogues(soap_format, shell_z=(0.01, 0.1), ang_radius_deg=5.)

halos = snap_beam.place_halos_in_shell(
    soap_format,
    lightcone_redshift_range=(0.01, 0.1),
    ang_radius_deg=5.,
    soap_chunk_size=2**22,       # rows of a catalogue read at a time
)

halos["Lightcone/HaloCentre"]           # lightcone positions, with the observer at the origin
halos["Lightcone/ExpansionFactor"]      # expansion factor at each halo's distance from the observer
halos["Lightcone/SnapshotNumber"]       # snapshot each halo was taken from
halos["InputHalos/SOAPIndex"]           # index of each halo in its snapshot's SOAP catalogue
halos.metadata                          # the settings that made the lightcone
```

All other properties for the returned haloes can then be read from the SOAP catalogues using each halo's snapshot and index values, e.g. for $`M_{200\mathrm{c}}`$:

```python
snap_nr = halos["Lightcone/SnapshotNumber"].value.astype(int)
soap_index = halos["InputHalos/SOAPIndex"].value.astype(int)
m200c = np.zeros(len(snap_nr))
for snap in np.unique(snap_nr):
    in_snap = snap_nr == snap
    rows, locaiton = np.unique(soap_index[in_snap], return_inverse=True)  # each row read once, in order
    soap = read_soap_selected_rows(soap_format.format(snap_nr=snap), ["SO/200_crit/TotalMass"], rows)
    m200c[in_snap] = comoving(soap["SO/200_crit/TotalMass"]).to_value("Msun")[locaiton]
```

`SnapshotAllSky` is used in the same way, without `beam_vector` or `ang_radius_deg`.

</details>

<a id="flamingo-and-colibre-simulations"></a>
<details>
<summary><b>FLAMINGO and COLIBRE simulations</b></summary>

For both FLAMINGO and COLIBRE simulaiton suites, to track a naming conventions, paths to the snapshots and SOAP catalogues and, to map a simulations snapshot number to redshift we have implemented a Snapshot Class.
Accessing a simulation's snapshot, identifying how a simulation is named and computing the redshift ranges of each snapshot, is done through a `Simulation` object. This way each simulations key information is consistent within the same suite. 

Moreover, lightcones can be built by passing a `Simulation` object instead of manually definging the individul paths for snapshots, SOAP catalogues and snapshot number to redshift conversions as shown below: 

```python
from swiftlet import Simulation, SnapshotAllSky, SnapshotBeam

flamingo = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")    # the suite is found from the label
colibre = Simulation.from_label("L0200N1504", "Thermal")
colibre = Simulation.from_path("/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal")

Snap_AllSky = SnapshotAllSky(colibre, orientation_lock="cube") # build full-sky lightcone object for colibre
Snap_Beam = SnapshotBeam(flamingo, beam_vector=[0., 0., 1.])  # build a beam lightcone object for flamingo               

colibre.snapshot_file(127)              # .../snapshots/colibre_0127/colibre_0127.0.hdf5
colibre.snapshot_redshift_range(127)    # redshifts the snapshot fills in a lightcone
```

Otherwise, `SnapshotAllSky("L1000N1800", "HYDRO_FIDUCIAL")` still works and makes a `Simulation` object witin `SnapshotAllSky` itself, with `simulation_suite_name="flamingo"` or `"colibre"` for a label that isn't recognised, and `simulation_base_dir_format` for a simulation somewhere else. 

The table below outlines the recognised simualtion suites, names and expected file structures for both COLIBRE and FLAMINGO. 

| | FLAMINGO | COLIBRE |
|---|---|---|
| Labels | `L1000N0900`, `L1000N1800`, `L1000N3600`, `L2800N5040` | box size zero padded, e.g. `L0200N1504`, or e.g. `L200m6` |
| Directory | `/cosma8/data/dp004/flamingo/Runs/<box_res>/<sim_name>` | `/cosma8/data/dp004/colibre/Runs/<box_res>/<sim_name>` |
| Snapshots | `snapshots/flamingo_XXXX/flamingo_XXXX.N.hdf5` | `snapshots/colibre_XXXX/colibre_XXXX.N.hdf5` |
| Snapshot redshifts | `flamingo_L1.txt` (`L1000N0900`, `L1000N1800`), `flamingo_L2p8.txt` (`L1000N3600`, `L2800N5040`) | `colibre.txt` |

To use simulations kept somewhere else, set the environmental variables `SWIFTLET_FLAMINGO_DIR` or `SWIFTLET_COLIBRE_DIR` to the directory holding them, each in `<box_res>/<sim_name>`.

The method for converting between snapshot numbers and redshifts is stored in predetermined tables. By default the snapshot redshifts are text files of comma separated values in `swiftlet/data/output_redshifts`, one snapshot per line:

```
# snapshot_number, redshift
0, 15.0
1, 10.38
...
```

`read_output_redshifts` reads them. You can provide your own, e.g. for a run with a different output list: `dataclasses.replace(COLIBRE, output_redshift_tables={"*": "/path/to/my_outputs.txt"})` gives a suite using it.

</details>

</details>

<details>
<summary><b>Orientation of the snapshot box tiles</b></summary>

To avoid exact copies of the same structure along a line of sight, each box tile is re-oriented at the level of the SWIFT cells. In order:

1. a periodic shift of the cells (and all the particles in them) along each axis, of 0, ±4, ±8 or ±16 cells, giving the box a 'new' structure,
2. a reflection (or mirroring) about each axis,
3. a rotation about each axis by a multiple of 90 degrees.

The total number of unique orientations or snapshot box structures is $`N_{\mathrm{shifts}}^3 \times 48`$, where $`N_{\mathrm{shifts}}`$ is the number of unique periodic shifts. 
For a SWIFT simulation with a cosmological volume subdivided into $`32\times32\times32`$ cells, there are 16464 possible unique tiles, all of which are used with orientation tables made by `write_orientation_tables`. The default tables use 79 of them. 

Which shift, reflection and rotation a tile gets is set by its position and the random number seed `orientation_seed`, so a lightcone can be reproduced exactly.
[`show_snapshot_tile_reorientation.py`](./examples/show_snapshot_tile_reorientation.py) builds a diagram of how the cells of a box are moved by each step:

```
python3 examples/show_snapshot_tile_reorientation.py
```

Each tile reads one row of the orientation tables (quarter turn, reflection and periodic shift). The default tables have 79 rows, so the tiles take at most 79 orientations. 
To use more orientations, write a file of orientation tables and pass it as `orientation_table`:

```python
from swiftlet import SnapshotAllSky, write_orientation_tables

write_orientation_tables("orientation_tables.hdf5", seed=0)   # every distinct orientation, 48 x 7^3 = 16464 rows
snap_all_sky = SnapshotAllSky("L1000N1800", "HYDRO_FIDUCIAL", orientation_seed=0, orientation_lock="cube",
                    orientation_table="orientation_tables.hdf5")
```

The generated tables hold every combination of the 48 quarter turns and reflections with the periodic cell shifts (by default 0, ±4, ±8 or ±16 cells along each axis), with the first row leaving the box unchanged. `seed` shuffles the rows, `n_rows` keeps only the first rows, and `cell_shift_values` sets the shifts, e.g. for 64 cells per axis. 
Any tables can be written with `write_orientation_tables(filename, tables={...})`. Without `orientation_table`, the default tables are used.

The orientations and their settings that made the lightcone are stored in `.metadata` of the output lightcones. `.metadata` stores the `orientation_seed`, `orientation_lock`, the orientation table (`"default"` or the file, and a fingerprint of its contents), the simulation, the beam, the redshift range and the package version. With the same .metadata and snapshots the same lightcone can be reproduced exactly. `write_lightcone_metadata` writes the metadata as attributes.  
The example, [`snapshot_smoothed_map.py`](./examples/snapshot_smoothed_map.py) does with its output map .hdf5 file, and `read_lightcone_metadata` reads them back.


The `orientation_lock` parameter of `SnapshotLightcone` (and subclasses) sets which tiles share an orientation:

- `orientation_lock=None` (or `"none"`, the default): every box tile has its own unique orientation, so discontinuities occur at every face of every tile in the lightcone. This creates the most discontinuities. 
- `orientation_lock="cube"`: every box in the same cube shell of boxes around the observer's box shares one orientation. Shell $`n`$ holds the tiles with $`n=\max(|n_x|, |n_y|, |n_z|)`$, where $`(n_x, n_y, n_z)`$ is the position of the box in box sidelengths. The orientation only changes across tile faces, a single tile is always completley within a single layer, and computationally the cost is the same as without a lock. However, the boundary between shells is not at one distance or redshift: it ranges from $`(n+½)L`$ along an axis to $`(n+½)\sqrt{3}L`$ along a box diagonal.
- `orientation_lock="sphere"`: everything in the same spherical shell, $`(n−½)L ≤ r < (n+½)L`$, shares the same orientation, so each shell covers its own redshift range. A tile crossing a sphere is used twice, once with each shell's orientation, and each copy keeps only the particles on its own side. This is more expensive, as those tiles are read twice, and the same region of a snapshot can appear in both shells, at different positions and orientations.

Within a locked shell, the tiles are periodically continuous, so structures continue across the faces between them. With the `"cube"` or `"sphere"` lock and the same `orientation_seed`, a beam from `SnapshotBeam` is exactly the same as the matching patch of the `SnapshotAllSky` lightcone, whichever way it points. Without a lock, this is not the case.

The diagram made by [`show_snapshot_orientation_lock.py`](./examples/show_snapshot_orientation_lock.py) shows which box tiles share an orientations for each lock, for an all-sky lightcone and for beams on and off the box axes:

```
python3 examples/show_snapshot_orientation_lock.py
```

</details>

<details>
<summary><b>Writing output .hdf5 files</b></summary>

<details>
<summary><b>HEALPix maps</b></summary>

See `examples/snapshot_smoothed_map.py` for an example of how construct all-sky HEALPIx maps from the snapshots. 

</details>

<details>
<summary><b>Particle lightcones (coming soon)</b></summary>

A method to directly write new particle lightcone .hdf5 files (as available with FLAMINGO) with a `SnapshotLightcone` function is yet to be implemented. The current best practice to write the outputs given by the `SnapshotLightcone` subclasses described above to a .hdf5 file using the example file structure below before indexing these new particle lightcones with [`lightcone_io/index_particles.py`](https://github.com/jchelly/LightconeIO/blob/master/lightcone_io/index_particles.py). These indexed particle lightcone files are now readable with `lightcone_io`.

Example FLAMINGO particle lightcone file structure:

```
L1000N1800/HYDRO_FIDUCIAL/lightcones/
│
├── lightcone0/                             # observer 0's lightcone
│   │
│   ├── lightcone0_0000.hdf5                # particle lightcone, file 0 of N
│   ├── lightcone0_0001.hdf5                # particle lightcone, file 1 of N
│   │   │
│   │   ├── PartType0/                      # gas particles
│   │   ├── PartType0/                      # gas particles
│   │   │   ├── Coordinates
│   │   │   ├── ExpansionFactors
│   │   │   ├── SnapshotNumber
│   │   │   ├── Masses
│   │   │  └── ...
│   │   ├── PartType1/                      # dark matter particles
│   │   │   └── ...
│   │   ├── PartType4/                      # stars
│   │   ├── PartType5/                      # black holes
│   │   │
│   │   ├── ...
│   │
│   ├── ...
│
├── lightcone1/                             # observer 1 (different orientation/position)
├── ...

```

</details>

<details>
<summary><b>Halo lightcones</b></summary>

The haloes placed from one snapshot can be written to a `.hdf5` file in the format of the `lightcone_io` (FLAMINGO) halo lightcones, with `write_halo_lightcone_file` of [`swiftlet.snapshot_lightcone`](./swiftlet/snapshot_lightcone.py). The haloes are sorted by HEALPix pixel (the `Index` group) with the `Lightcone` and `InputHalos` datasets and the units used by LightconeIO, and any SOAP properties given are written alongside, so `lightcone_io.halo_reader.HaloLightconeFile` can read the file and look up the rest from SOAP. 
For example, as given in [`colibre_halo_lightcone_full_sky.py`](./examples/colibre_halo_lightcone_full_sky.py):

```python
from lightcone_io.halo_reader import HaloLightconeFile
from swiftlet.soap_reader import write_halo_lightcone_file  

snap = int(snap_nr.max())  # the lowest redshift snapshot
keep = snap_nr == snap
snapshot_halos = {name: values[keep] for name, values in halos.items()}
soap_filename = soap_format.format(snap_nr=snap)

# SOAP properties to write with the haloes, one value per halo
rows, where = np.unique(soap_index[keep], return_inverse=True)
is_central = read_soap_selected_rows(soap_filename, ["InputHalos/IsCentral"], rows)["InputHalos/IsCentral"][where]

halo_filename = f"lightcone_halos_{snap:04d}.hdf5"
write_halo_lightcone_file(halo_filename, snapshot_halos, soap_filename, {"InputHalos/IsCentral": is_central.value}, halos.metadata)

# read it back with lightcone_io, the haloes in a region of the sky, and M200c from SOAP
halo_file = HaloLightconeFile(halo_filename, soap_filename=soap_filename)
haloes = halo_file.read_halos_in_radius(
                            np.array([1., 1., 1.]) / np.sqrt(3.), np.radians(5.),
                            ["Lightcone/HaloCentre", "Lightcone/Redshift", "InputHalos/IsCentral", "SO/200_crit/TotalMass"]
                            )
```

</details>

</details>

---

<details>
<summary>

## Projections

</summary>

<details>
<summary><b>Project particles in a slice</b></summary>

`BeamProjection` makes projections of a slice through a beam of a lightcone using [`swiftsimio.visualisation.projection`](https://swiftsimio.readthedocs.io/en/latest/visualisation/projection.html) submodule backends. The particles can come from the FLAMINGO particle lightcones (a path, or `lightcone_io` particle data) or from a snapshot using `SnapshotBeam` and `SnapshotAllSky`. 

As shown in the example below, to make a projection:
- **define the projected slice:** to define a slice through the lightcone requires: 1) a beam within the lightcone (i.e. pass a `vector`, `radius` and `redshift range` that defines the beam) 2) the thickness of the slice through the beam (`slice_thickness`), and 3) either a cosmology object of the simulation corresponding to the intended lightcone data or a path to a snapshot of the same simulation. Note that, the projected beam and the corresponding particles are rotated to lie along the x-axis, so the slice is always through the middle of the beam: along the z-axis.
- **add particles to the slice:** add the lightcones particle data (or equivalent dictionary object of `unyt` arrays) and the property names you want to project too the slice via `BeamProjection.place_particles_in_slice()`
- **project selected properties:** for a given particle type in the slice, create a projection of the selected properties in the assigned units using `BeamProjection.project_properties()`. The slice is projected with SWIFTsimIO, through a mock snapshot that takes its metadata (units, box size, cosmology) from a real snapshot of the simulation. Gas uses its own smoothing lengths, and smoothing lengths are generated for dark matter.

```python
from swiftlet import BeamProjection

BP = BeamProjection(
    vector=(1, 0, 0),                           # direction of the beam
    angular_diameter=10,                        # [deg]
    redshift_range=(0.01, 0.1),`
    slice_thickness=10,                         # [Mpc]
    store_snapshot_filename=snapshot_filename,  # snapshot used for the metadata and cosmology
)

# place the gas and dark matter particles of the beam in the slice
BP.place_particles_in_slice(
    gas_particle_data, ["Coordinates", "Masses", "SmoothingLengths", "ExpansionFactors", "Temperatures"],
    dm_particle_data=dm_particle_data, dm_property_names=["Coordinates", "Masses", "ExpansionFactors"],
)

# surface densities, and a mass weighted mean
gas_surface_density = BP.project_properties(["Masses"], ptype="Gas", assign_units=["Msun"])[0]
dm_surface_density = BP.project_properties(["Masses"], ptype="DM", assign_units=["Msun"])[0]
total_surface_density = gas_surface_density + dm_surface_density
gas_temperature = BP.project_properties(["Temperatures"], ptype="Gas", weight="Masses")[0]
```

`project_properties` returns one image per property (`unyt` arrays):

- **Default:** the surface density of each property.
- **With `weight`:** the weighted mean, sum(q w) / sum(w), and 0 where there is no weight.
- **Other options:** `resolution` sets the number of pixels along each axis, and `periodic` (default True) treats the box as periodic. `parallel` uses SWIFTsimIO's parallel projection backend. Other properties can be added to the slice with `add_property_to_slice`.

</details>

<details>
<summary><b>Plotting beams</b></summary>

`BeamProjection.split_beam_plot` draws the beam as a wedge in comoving distance and angle, with redshift, comoving distance and angle axes. The beam can be split into several wedges, each showing a different particle type and/or property:

```python
images = [image.to_value("Msun/Mpc**2") for image in (gas_surface_density, total_surface_density, dm_surface_density)]
fig, ax, wedge_images = BP.split_beam_plot(
    3,
    [[image, (vmin, vmax)] for image in images],   # each wedge's image and colour range
    ["magma", "cubehelix", "viridis"],             # individually set each wedges colour map
    titles=["Gas", "Gas+DM", "DM"],                # the title placed in each wedge
    norms=["log", "log", "log"],                   # or "linear", or any matplotlib Normalize, "log" by default
    overlay_grid=(False, False, False),            # grid lines at the redshift, distance and angle ticks
    filename="beam.png",                           # save image to this file
)
```

It either draws a new figure or adds each wedge to an existing axes, then returns the figure, the axes and the image of each wedge. Other keyword arguments style the wedges, ticks, labels and grid lines (see `BeamPlot.add_wedge` and `BeamPlot.add_beam_axes`). 
The plotting is done by `BeamPlot`, which can also be used on its own with a cosmology, angular diameter, redshift range and the extent of the images.

See [`lightcone_beam_projection.py`](./examples/lightcone_beam_projection.py) for beams of the FLAMINGO particle lightcones, and [`snapshot_beam_projection.py`](./examples/snapshot_beam_projection.py) for beams built from the snapshots.

</details>

</details>

---

<details>
<summary>

## Examples

</summary>

<details>
<summary><b>Overview of example scripts</b></summary>

The [examples](./examples) are named after the data they start from: `snapshot_` examples build new lightcones from the snapshots, `lightcone_` and `map_` examples use the existing FLAMINGO lightcone particles, maps and haloes, and `show_` examples draw diagrams that need no data.

| Example | What it does |
|---|---|
| `snapshot_beam_projection.py` | builds a `SnapshotBeam` of gas particles for each orientation lock and projects a slice through it, to compare the locks |
| `snapshot_beam_particle_placement.py` | places snapshot particles in beams pointing in different directions, reading the files in chunks, and plots them against the geometry of the beam |
| `snapshot_beam_halo_placement.py` | places haloes from the SOAP catalogues in a `SnapshotBeam` |
| `snapshot_smoothed_map.py` / `.sh` | makes an all-sky HEALPix map of the smoothed gas mass from the snapshots, in parallel with MPI, with a job script |
| `lightcone_beam_projection.py` | projects slices through a beam of FLAMINGO lightcone particles: the gas, dark matter and total surface density, and the mass weighted gas temperature |
| `lightcone_halo_M500c_binary_masks.py` / `.sh` | makes binary HEALPix masks of the haloes in the lightcone within their M500c radius, with a job script |
| `integrated_lightcone_map.py` | integrates the rotated FLAMINGO HEALPix shell maps over a redshift range |
| `map_zoom_on_halo.py` | reads only the pixels of a HEALPix map around a halo and images them with a gnomonic projection |
| `snapshot_beam_zoom_on_haloes.py / .sh` | The same as above but using `SnapshotBeam` to build lightcone beams centred on a halo |
| `show_snapshot_tile_reorientation.py` | shows how the cells of a snapshot box are shifted, reflected and rotated to make a new box tile |
| `show_snapshot_orientation_lock.py` | shows which box tiles share an orientation for each `orientation_lock`, for all-sky and on and off axis beams |

</details>

</details>

---

<details>
<summary>

## Additional utility functions:

</summary>

<details>
<summary><b>Snapshots, redshifts and units</b></summary>

- `simulation_suites`: everything that differs between FLAMINGO and COLIBRE, see [FLAMINGO and COLIBRE simulations](#flamingo-and-colibre-simulations).
  - **`SimulationSuite`** (`FLAMINGO`, `COLIBRE`): the suite's box size and resolution labels, directory, snapshot file names and output redshift files (`swiftlet/data/output_redshifts`, read with `read_output_redshifts`). `suite_for_label` finds the suite of a label, and `output_redshifts(box_res)` gives its output redshifts when only those are needed.
  - **`Simulation`**: one simulation. `output_redshifts`, `snapshot_redshift`, `snapshot_number` (the snapshot at a redshift), `snapshot_redshift_range`, `snapshots_in_range` and `z0_snapshot` use its suite's output redshifts.
- `snapshot_redshifts`: funcitons to return a simulation's output redshifts, with nothing specific to the simulation suite. Each snapshot covers the redshifts between the midpoints to its neighbours, the first and last ending at their own redshifts (`redshift_edges`, `snapshot_redshift_range`). `snapshots_in_range` gives every snapshot whose range overlaps a redshift range. Edges within `range_tol` (1e-9) of the range count as touching it, so floating point noise doesn't decide which snapshots touch the lightcones redshift range. 
- `snapshot_cosmology`: `SnapshotCosmology`, the cosmology of a FLAMINGO or COLIBRE simulation read from one of its snapshots (`SnapshotCosmology.from_simulation(sim)`, or `from_path` with a snapshot file or directory). `.COSMO` is the astropy w0waCDM of the snapshot's Cosmology group, including its neutrinos (`N_eff`, `N_ur`, and the masses `M_nu_eV` of the massive species with their degeneracies `deg_nu`, mapped to astropy as swiftsimio does), with H0 converted from the code's own time unit using the snapshot's unit registry (`.unit_registry`). It also gives `comoving_distance`, `redshift_with_time_offset`, `shell_comoving_radii` and `lightcone_volume`. The lightcones, `BeamProjection` and `BeamPlot` use it, and wherever they take a cosmology they take a `SnapshotCosmology`, an astropy cosmology or the path of a snapshot (`as_astropy_cosmology`).
- `flamingo_lightcones`: FLAMINGO's own lightcone products. `flamingo_shell_redshift_file` finds the shell redshift files, downloading them if needed. `flamingo_box_resolution` reads the box and resolution label (e.g. `L1000N1800`) from a path.
- `soap_reader`: reading SOAP halo catalogues, each reader applies the properties the units `lightcone_io`'s `SOAPCatalogue` gives them. `read_soap_rows` reads a range of rows, `read_soap_in_chunks` the whole catalogue a chunk at a time, `read_soap_selected_rows` only the given rows, and `select_soap_rows` the rows where one property passes a test (e.g. a mass range), reading that property in chunks and the others only for the rows kept. `soap_catalogue_length` gives the number of haloes without reading any, and `comoving` drops the expansion factor from a property's units (e.g. `a*snap_length` to comoving lengths). It also holds the patch letting virgodc (<= 1.0.5) read SOAP units with numpy >= 2.4. `place_halos_in_shell` and the halo zoom examples use it.
- `snapshot_units`:
  - **Applying units:** `apply_expected_units` gives values the expected units, if they have none. `drop_a_from_comoving_property` gives comoving lengths in snapshot units without the scale factor.

</details>

<details>
<summary><b>HEALPix maps (<code>healpix_map_utils</code>)</b></summary>

- `get_related_ipix` gives the parent (lower resolution) or child (higher resolution) pixels of HEALPix pixels. `get_common_maps` lists the maps found in all of a set of files.
- `write_rotated_lightcone_chunks` integrates the FLAMINGO HEALPix shell maps along the line of sight, as in [`integrated_lightcone_map.py`](./examples/integrated_lightcone_map.py). The shells are rotated by their own angles (`theta_arr_deg`, `phi_arr_deg`), so it sums each group of shells sharing the same angles into a chunk, rotates the chunk, and writes the sum of all chunks. It can also save each chunk (`save_chunks`), work at a different `rotate_nside` or `output_nside`, and convert units before rotating.
- `rotate_map` and `rotate_map_fast` rotate a map in spherical harmonic space. The fast version uses the HEALPix pixel weights (see [Additional Data](#additional-data)), and `write_rotated_lightcone_chunks` uses it when they are available for that nside.
- `sum_maps` writes a new file with the sum of the same maps across several files. `map_names=["common"]` uses every map found in all the files.

</details>

</details>

---

<details>
<summary>

## Tests

</summary>

<details>
<summary><b>How to run the tests, and what they check</b></summary>

The [tests](./tests) need no simulation data: they write small fake snapshots and SOAP catalogues to a temporary directory each time they run. 

Install the package with pytest (`pip install -e ".[test]"`, or `".[all]"`), then to run all tests from the top of the repository:

```
pytest tests
```

To run only a part of a given test:

```
pytest tests/test_snapshot_lightcone_placement.py                      # one file
pytest tests/test_snapshot_lightcone_placement.py -k number_density    # the tests whose names match
pytest tests/test_snapshot_lightcone_placement.py -k "mpi or lattice"
pytest tests/test_snapshot_lightcone_placement.py::test_beam_is_part_of_all_sky
```

Note that the tests use at most 8 threads , where ever `SWIFTLET_MAX_THREADS` would be used and can be changed in `tests/conftest.py`.

Add `-v` to list every test case, `-x` to stop at the first failure, `--durations=10` to show the slowest tests and `-p no:warnings` to hide the deprecation warnings of unyt and healpy. The snapshot lightcone tests take about 10 minutes; the first run is slower while numba compiles the re-orientation code.


| Test file | What it checks |
|---|---|
| `test_config.py` | downloading the shell redshift files, adding their paths and the maximum number of threads to the environment's activate script and finding them when needed |
| `test_threads.py` | the maximum number of threads (`SWIFTLET_MAX_THREADS`), and that numba and healpy alm parallel funcations abide by it |
| `test_init.py` | the public classes and functions are imported from the package when first used, and importing the package doesn't load the modules that need MPI |
| `test_orientation_tables.py` | the default orientation tables, generating tables with every distinct orientation, and writing, reading and checking table files |
| `test_snapshot_lightcone_placement.py` | where `SnapshotBeam` and `SnapshotAllSky` place particles and haloes |
| `test_snapshot_cosmology.py` | `SnapshotCosmology` read from fake FLAMINGO and COLIBRE snapshots: H0 from the code units, the cosmological parameters, the neutrinos (checked against swiftsimio), distances, volumes and shells, finding the snapshot files, and the lightcones taking any cosmology |
| `test_snapshot_redshifts.py` | the redshift range each snapshot covers, and `snapshots_in_range` giving every snapshot overlapping a redshift range and the tolerance, for the FLAMINGO and COLIBRE output redshifts |
| `test_simulation_suites.py` | each simulation's paths and snapshot redshifts come from its suite, FLAMINGO or COLIBRE. Make a `Simulation` object from its label or path, finding the suite of a label, the directories (including `SWIFTLET_FLAMINGO_DIR` and `SWIFTLET_COLIBRE_DIR`), the output redshift files, and the lightcone classes taking a `Simulation` or a label and name |
| `test_soap_reader.py` | the SOAP readers of `swiftlet.soap_reader` give the same rows and units whether reading a in chunks or selecting rows by a property, and correctly drops the expansion factor |


Note the lightcone placement tests are very extensive and can take a while (order of 30 minutes) to complete. Below is a list of the various tests within `test_snapshot_lightcone_placement.py`: 

- with a `"cube"` or `"sphere"` lock, a beam is exactly the part of the all-sky lightcone inside its cone;
- particles on a uniform lattice are placed exactly once and the reconstructed lattice in the lightcone has no gaps or overlaps between them;
- consecutive shells together give the same particles as one shell over the whole range;
- placing the particles file by file (`gather_files` and `place_file_in_shell`) or in parallel with MPI gives exactly the particles of placing them in serial;
- the number of particles placed matches the snapshots' number density times the volume they fill;
- re-orienting points on cell faces keeps them on cell faces.
- an all-sky lightcone out to half a box length places every snapshot particle within it at its box position less half a box length, as the observer's box tile is not re-oriented;
- haloes placed from SOAP catalogues end up exactly where particles at the same positions in the snapshots are placed, including haloes on cell faces;
- the haloes placed are the same, in the same order, whether the SOAP catalogues are read whole or in chunks, and with MPI (on 1, 2 and 3 ranks, also with `redistribute_halos`) the same as in serial;
- the halo dry run (`gather_halo_catalogues`) reads no haloes and the cell metadata of one snapshot only, and `place_halos_in_shell` reuses it, reading only the catalogues it found; with no snapshot in the shell no halo is placed;
- `SnapshotBeam` needs an angular radius and `SnapshotAllSky` takes none, so arguments given by position in a beam's order are caught;
- without the `"sphere"` lock, no particle or halo is placed twice in the same box tile, and across the lightcone none are placed more often than the number of tiles its snapshot fills.
- the shortest distance between particles placed into the lightcone from the same tile is the same as in the snapshot, so each tile is the box moved as a whole;
- the `ExpansionFactors` of the particles and haloes placed are 1 / (1 + z) at their comoving distance, with z found independently;
- non-essential properties read with the particles from a snapshot (test cases uses `Masses`, different for every snapshot and ID) are carried through unchanged with each particle;
- every tile's orientation is an exact signed permutation of the axes with a periodic shift of 0, ±4, ±8 or ±16 cells, face neighbouring tiles differ without a lock, tiles of one layer share an orientation with the `"cube"` or `"sphere"` lock, and the same `orientation_seed` gives the same orientations;
- a lightcone made with the default tables written to a file is the same as one made with the hardcoded default a table;
- the .metadata of the constructed lightcone (`orientation_seed`, `orientation_lock`, orientation table, beam and redshift range), is correctly written to and read back from .hdf5 files dataset attributes;

To see what the lattice test checks, add `--figures-dir` with a directory to save figures to. `test_lattice_over_many_tiles` then saves one figure per case, showing one layer of the lattice placed in the lightcone with the expected lattice points, any missing or repeated points, the box tile edges and the snapshot hand-overs, and a close-up of a corner where tiles meet. The figure is drawn before the checks, so a failing case shows where it goes wrong:

```
pytest tests/test_snapshot_lightcone_placement.py -k lattice_over_many_tiles --figures-dir test_figures
```


The MPI tests, `test_mpi_matches_serial` and `test_mpi_haloes_match_serial`, runs [`mpi_place_particles.py`](./tests/mpi_place_particles.py) and [`mpi_place_halos.py`](./tests/mpi_place_halos.py) with `mpiexec -n 2` and `-n 3`. They use the `mpiexec` next to the Python executable (where the `mpich` wheel installs it) if there is one, otherwise the one on the `PATH`, and are skipped if there is none or mpi4py can't be imported. The `mpiexec` must belong to the MPI library mpi4py was built with. On a cluster, run them on a compute node (e.g. in an interactive `salloc` or `srun` session), as login nodes may not allow `mpiexec`.

</details>

</details>

---

<details>
<summary>

## Use of AI

</summary>

We acknowledge the assistance of AI (Claude Code, Anthropic 2026) in writing parts of this module, which included  most of the test suite (`tests/`), the documentation (the doc-strings and this `README.md` file), configuration code (`swiftlet/threads.py`, `swiftlet/config.py` and `swiftlet/__init__.py`) and the diagram examples (`examples/show_snapshot_tile_reorientation.py` and `examples/show_snapshot_orientation_lock.py`). 
All use of AI was supervised and any contribution to this module was first vetted by humans.

</details>

