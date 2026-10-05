# ExtraSWIFTLightcones

Supplementary tools for the post-processing and visualisation of [SWIFT](https://swift.strw.leidenuniv.nl/docs/index.html) lightcones. This module includes tools to generate new lightcones from the SWIFT snapshots, and, project, plot and post process lightcones. It builds on [LightconeIO](https://lightconeio.readthedocs.io/en/latest/#) and [SWIFTsimIO](https://swiftsimio.readthedocs.io/en/latest/), which it requires.

Further information about the FLAMINGO lightcones: https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html

## Installation
Clone the repository, then either build a new virtual environment with everything needed, or pip install into an environment you already have. 
Both download the FLAMINGO lightcone shell redshift files into the environment and export their paths whenever it is activated.

### New virtual environment

From anywhere, make a virtual environment with ExtraSWIFTLightcones (in editable mode), the dependencies of the package, [examples](./examples) and tests, and the shell redshift files:

```
git clone https://github.com/Will-McD/ExtraSWIFTLightcones.git
bash ExtraSWIFTLightcones/venv_scripts/make_venv.sh /path/to/environment
source /path/to/environment/bin/activate
```

If no path is given, the environment is made in `ExtraSWIFTLightcones/extra_swift_lightcones_env`.

On COSMA, build the virtual environment with the pre-built wheels of mpi4py and h5py for COSMA's MPI and parallel HDF5 instead:

```
bash ExtraSWIFTLightcones/venv_scripts/make_cosma_env.sh /path/to/environment
```
### Existing environment

With your environment (venv or conda) activated, install the module and its requirements from the repositories directory before then downloading the shell redshift .txt files:  

```
cd ExtraSWIFTLightcones
pip install -e .              # the module and its requirements
pip install ".[all]"          # the module, its requirements + the requirements of the examples and tests
extra_swift_lightcones-configure
```

To specifically install the additional requirements for the examples and tests:

```
pip install ".[examples]"     # the requirements of the examples
pip install ".[test]"         # the requirements of the examples
```

The shell redshift .txt files, give the redshift bounds of [each shell within FLAMINGO's lightcones](https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html). 
`extra_swift_lightcones-configure` downloads the shell redshift .txt files to `<environment>/share/extra_swift_lightcones/redshifts` and adds `L1_REDSHIFTS_FILENAME` and `L2P8_REDSHIFTS_FILENAME` to the environment's activate script (`bin/activate` for a venv, `etc/conda/activate.d` for conda). 
If `extra_swift_lightcones-configure` isn't run, the lightcone shell redshifts are downloaded the first time they are needed instead.

These shell redshifts are required for mapping between FLAMINGO's HEALPix maps, constructed in concentric redshift shells, and the corresponding halo lightcones which are constructed per snapshot or for identifying which HEALPix maps exist within a given redshift range. 

The [COLIBRE](https://colibre.strw.leidenuniv.nl/index.html) simulation suite does not have lightcones and as such doesn't have shell redshift files. 



### MPI support

MPI support is not required to generate new lightcones from snapshots with the `SnapshotLightcone` sub classes. However, it is necessary for the more efficient parallel methods of `SnapshotLightcone`, the `BeamProjection` class and for generating binary masks of haloes (`mask_haloes.py`).

MPI support requires mpi4py and an MPI enabled build of h5py. 
The mpi4py package installed from PyPI needs an MPI library at run time. If your system has none (e.g. on a laptop), install one into the environment with `pip install mpich` (or `pip install openmpi`). `extra_swift_lightcones-configure` doesn't need MPI.

See the instructions for installing LightconeIO with MPI support: [LightconeIO with MPI support](https://lightconeio.readthedocs.io/en/latest/installation.html#with-mpi-support)

### Additional Data 

To add the shell redshift files to another existing environment:

```
bash ExtraSWIFTLightcones/venv_scripts/shell_redshifts.sh /path/to/environment
```

Additionally, to further speed up the rotation of HEALPix maps, download the HEALPix pixel weights and add their paths to your virtual environment. 

```
cd ./ExtraSWIFTLightcones
bash venv_scripts/healpix_pixel_weights.sh
```


## Generating Lightcones from SWIFT Snapshots

The `SnapshotLightcone` classes build new lightcones from SWIFT snapshots by tiling periodic copies of the snapshot box around an observer, who sits at the centre of the first box. We refer to these periodic replicas as tiles. 
As the observer's past lightcone propagates through the lattice of tiles, the lightcone is filled with particles (and/or haloes) from the snapshot closest in redshift, as chosen by the comoving distance from the observer.
The snapshot particles (and haloes) placed in the lightcone (`place_snapshot_particles_in_shell`) are returned with the same structure, for each particle type, as given by `lightcone_io.ParticleLightcone` class objects. The returned snapshot-lightcone particles always have the following properties: 
- `Coordinates`         the lightcone-frame x,y,z coordinates of the particles
- `ExpansionFactors`    the particles scale factor ($`\frac{1}{z+1}`$) at their comoving distance from the observer
- `SnapshotNumber`      the number of the snapshot the particle belongs to.

The reading and placing of particles into the lightcone is done first at the cell-level of the SWIFT snapshots. The cells from a given snapshot that overlap with the lightcone's footprint (including a $`1/2`$ cell length buffer) are flagged, then only the particles within the flagged cells are read from the corresponding SWIFT snapshot .hdf5 files. 
Note that passing an MPI communicator (`comm`) reads the snapshot files in parallel.
To limit memory use, the files containing particles within the lightcone's footprint can also be gathered and read from one at a time to place particles in the lightcone (`gather_files` and `place_file_in_shell`).  
Haloes from SOAP catalogues can be placed in the same lightcone (`place_halos_in_shell`), and points can be mapped between the snapshot and the lightcone (`Snapshot2Lightcone` and `Lightcone2Snapshot`).

### Snapshot-to-lightcone methods

There are two subclasses of snapshot lightcones

- `SnapshotBeam` builds a beam about a line of sight (`beam_vector`, a unit vector), with an angular radius up to 60 degrees. Box tiles are added along the line of sight as the lightcone extends past each tile, and transverse to the line of sight once the beam surpasses the width of a tile.
- `SnapshotAllSky` builds a lightcone over the full sky, adding shells of box tiles around the observer's box as the lightcone extends: a single box, then a 3×3×3 cube of boxes, then 5×5×5, and so on.

The box tiles sit on the lattice of whole box sidelengths, so they fill the lightcone exactly once in any direction.

#### Place snapshot particles into a past lightcone

For example, to place the gas and dark matter particles of an all-sky shell, reading the snapshot files in parallel with MPI:

```python
from mpi4py import MPI
from extra_swift_lightcones import SnapshotAllSky

snap_all_sky = SnapshotAllSky(
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


### Orientation of the snapshot box tiles

To avoid exact copies of the same structure along a line of sight, each box tile is re-oriented at the level of the SWIFT cells. In order:

1. a periodic shift of the cells (and all the particles in them) along each axis, of 0, ±4, ±8 or ±16 cells, giving the box a 'new' structure,
2. a reflection (or mirroring) about each axis,
3. a rotation about each axis by a multiple of 90 degrees.

The total number of unique orientations or snapshot box structures is $`N_{\mathrm{shifts}}^3 \times 48`$, where $`N_{\mathrm{shifts}}`$ is the number of unique periodic shifts. 
For a SWIFT simulation with a cosmological volume subdivided into $`32\times32\times32`$ cells, there are 16464 possible unique tiles. 

Which shift, reflection and rotation a tile gets is set by its position and the random number seed `orientation_seed`, so a lightcone can be reproduced exactly.
[`show_snapshot_box_reorientation.py`](./examples/show_snapshot_box_reorientation.py) builds a diagram of how the cells of a box are moved by each step:

```
python3 examples/show_snapshot_box_reorientation.py
```

The `orientation_lock` parameter of `SnapshotLightcone` (and subclasses) sets which tiles share an orientation:

- `orientation_lock=None` (or `"none"`, the default): every box tile has its own unique orientation, so discontinuities occur at every face of every tile in the lightcone. This creates the most discontinuities. 
- `orientation_lock="cube"`: every box in the same cube shell of boxes around the observer's box shares one orientation. Shell $`n`$ holds the tiles with $`n=\max(|n_x|, |n_y|, |n_z|)`$, where $`(n_x, n_y, n_z)`$ is the position of the box in box sidelengths. The orientation only changes across tile faces, a single tile is always completley within a single layer, and computationally the cost is the same as without a lock. However, the boundary between shells is not at one distance or redshift: it ranges from $`(n+½)L`$ along an axis to $`(n+½)\sqrt{3}L`$ along a box diagonal.
- `orientation_lock="sphere"`: everything in the same spherical shell, $`(n−½)L ≤ r < (n+½)L`$, shares the same orientation, so each shell covers its own redshift range. A tile crossing a sphere is used twice, once with each shell's orientation, and each copy keeps only the particles on its own side. This is more expensive, as those tiles are read twice, and the same region of a snapshot can appear in both shells, at different positions and orientations.

Within a locked shell, the tiles are periodically continuous, so structures continue across the faces between them. With the `"cube"` or `"sphere"` lock and the same `orientation_seed`, a beam from `SnapshotBeam` is exactly the same as the matching patch of the `SnapshotAllSky` lightcone, whichever way it points. Without a lock, this is not the case.

The diagram made by [`show_snapshot_orientation_lock.py`](./examples/show_snapshot_orientation_lock.py) shows which box tiles share an orientations for each lock, for an all-sky lightcone and for beams on and off the box axes:

```
python3 examples/show_snapshot_orientation_lock.py
```
 
### Writing .hdf5 files

#### HEALPix maps
See `examples/snapshot_smoothed_map.py` for an example of how construct all-sky HEALPIx maps from the snapshots. 

#### Particle lightcones (**coming soon**)

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

## Projections

`BeamProjection` makes projections of a slice through a beam of a lightcone using [`swiftsimio.visualisation.projection`](https://swiftsimio.readthedocs.io/en/latest/visualisation/projection.html) submodule backends. The particles can come from the FLAMINGO particle lightcones (a path, or `lightcone_io` particle data) or from a `SnapshotBeam`. The beam is rotated to lie along the x-axis, and the particles in its redshift range and in a slice `slice_thickness` thick through the middle of the beam (along z) are kept. The slice is projected with SWIFTsimIO, through a mock snapshot that takes its metadata (units, box size, cosmology) from a real snapshot of the simulation. Gas uses its own smoothing lengths, and smoothing lengths are generated for dark matter.

```python
from extra_swift_lightcones import BeamProjection

BP = BeamProjection(
    vector=(1, 0, 0),                           # direction of the beam
    angular_diameter=10,                        # [deg]
    redshift_range=(0.01, 0.1),
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

### Plotting beams

`split_beam_plot` draws the beam as a wedge in comoving distance and angle, with redshift, comoving distance and angle axes. The beam can be split into several wedges along the angle, each showing a different image, e.g. gas, total and dark matter:

```python
images = [image.to_value("Msun/Mpc**2") for image in (gas_surface_density, total_surface_density, dm_surface_density)]
fig, ax, wedge_images = BP.split_beam_plot(
    3,
    [[image, (vmin, vmax)] for image in images],   # each wedge's image and colour range
    ["magma", "cubehelix", "viridis"],
    titles=["Gas", "Gas+DM", "DM"],
    norms=["log", "log", "log"],           # or "linear", or any matplotlib Normalize, "log" by default
    overlay_grid=(False, False, False),    # grid lines at the redshift, distance and angle ticks
    filename="beam.png",
)
```

It draws onto a new figure or onto an existing axes and returns the figure, the axes and the image of each wedge. Other keyword arguments style the wedges, ticks, labels and grid lines (see `BeamPlot.add_wedge` and `BeamPlot.add_beam_axes`). 
The plotting is done by `BeamPlot`, which can also be used on its own with a cosmology, angular diameter, redshift range and the extent of the images.

See [`lightcone_beam_projection.py`](./examples/lightcone_beam_projection.py) for beams of the FLAMINGO particle lightcones, and [`snapshot_beam_projection.py`](./examples/snapshot_beam_projection.py) for beams built from the snapshots.



## Examples

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
| `show_snapshot_box_reorientation.py` | shows how the cells of a snapshot box are shifted, reflected and rotated to make a new box tile |
| `show_snapshot_orientation_lock.py` | shows which box tiles share an orientation for each `orientation_lock`, for all-sky and on and off axis beams |


## Additional utility functions: 

### Snapshots, redshifts and units

- `swift_snapshot_redshift_conversion`:
  - **Snapshot redshifts:** `snapshot_number_redshifts` gives the redshift of FLAMINGO (and COLIBRE) snapshots, or the snapshot at a redshift. `snapshot_redshift_range` gives the redshift range each snapshot covers, and `snapshot_number_in_range` the snapshots whose ranges overlap a redshift range.
  - **Other:** `flamingo_shell_redshift_file` finds the shell redshift files, downloading them if needed. `flamingo_box_resolution` reads the box and resolution label (e.g. `L1000N1800`) from a path.
- `snapshot_units`:
  - **Applying units:** `apply_expected_units` gives values the expected units, if they have none. `drop_a_from_comoving_property` gives comoving lengths in snapshot units without the scale factor.
  

### HEALPix maps (`healpix_map_utils`)

- `get_related_ipix` gives the parent (lower resolution) or child (higher resolution) pixels of HEALPix pixels. `get_common_maps` lists the maps found in all of a set of files.
- `write_rotated_lightcone_chunks` integrates the FLAMINGO HEALPix shell maps along the line of sight, as in [`integrated_lightcone_map.py`](./examples/integrated_lightcone_map.py). The shells are rotated by their own angles (`theta_arr_deg`, `phi_arr_deg`), so it sums each group of shells sharing the same angles into a chunk, rotates the chunk, and writes the sum of all chunks. It can also save each chunk (`save_chunks`), work at a different `rotate_nside` or `output_nside`, and convert units before rotating.
- `rotate_map` and `rotate_map_fast` rotate a map in spherical harmonic space. The fast version uses the HEALPix pixel weights (see [Additional Data](#additional-data)), and `write_rotated_lightcone_chunks` uses it when they are available for that nside.
- `sum_maps` writes a new file with the sum of the same maps across several files. `map_names=["common"]` uses every map found in all the files.


## Tests

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

Add `-v` to list every test case, `-x` to stop at the first failure, `--durations=10` to show the slowest tests and `-p no:warnings` to hide the deprecation warnings of unyt and healpy. The snapshot lightcone tests take about 10 minutes; the first run is slower while numba compiles the re-orientation code.


| Test file | What it checks |
|---|---|
| `test_config.py` | downloading the shell redshift files, adding their paths to the environment's activate script and finding them when needed |
| `test_init.py` | the public classes and functions are imported from the package when first used, and importing the package doesn't load the modules that need MPI |
| `test_snapshot_lightcone_placement.py` | where `SnapshotBeam` and `SnapshotAllSky` place particles and haloes, see below |

`test_snapshot_lightcone_placement.py` checks, for all-sky and beams with each `orientation_lock`, that:

- with a `"cube"` or `"sphere"` lock, a beam is exactly the part of the all-sky lightcone inside its cone;
- particles on a uniform lattice are placed exactly once and the reconstructed lattice in the lightcone has no gaps or overlaps between them;
- consecutive shells together give the same particles as one shell over the whole range;
- placing the particles file by file (`gather_files` and `place_file_in_shell`) or in parallel with MPI gives exactly the particles of placing them in serial;
- the number of particles placed matches the snapshots' number density times the volume they fill;
- re-orienting points on cell faces keeps them on cell faces.
- an all-sky lightcone out to half a box length places every snapshot particle within it at its box position less half a box length, as the observer's box tile is not re-oriented;
- haloes placed from SOAP catalogues end up exactly where particles at the same positions in the snapshots are placed, including haloes on cell faces;

The MPI test, `test_mpi_matches_serial`, runs [`mpi_place_particles.py`](./tests/mpi_place_particles.py) with `mpiexec -n 2` and `-n 3`. It uses the `mpiexec` next to the Python executable (where the `mpich` wheel installs it) if there is one, otherwise the one on the `PATH`, and is skipped if there is none or mpi4py can't be imported. The `mpiexec` must belong to the MPI library mpi4py was built with. On a cluster, run it on a compute node (e.g. in an interactive `salloc` or `srun` session), as login nodes may not allow `mpiexec`.
