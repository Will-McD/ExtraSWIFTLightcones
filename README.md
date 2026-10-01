# ExtraSWIFTLightcones

This module provides additional tools for the post-processing and visualisation of [SWIFT](https://swift.strw.leidenuniv.nl/docs/index.html) lightcones. Specifically, this module provides the tools to generate new lightcones from the swift snapshots or  to either [LightconeIO]([https://github.com/jchelly/LightconeIO](https://lightconeio.readthedocs.io/en/latest/#)) and [SWIFTsimIO](https://swiftsimio.readthedocs.io/en/latest/), therefore it requires the both the aforementioned packages 

Further information about the FLAMINGO lightcones: https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html

## Installation
Clone the repository, then either build a new virtual environment with everything needed, or pip install into an environment you already have. 
Both download the FLAMINGO lightcone shell redshift files into the environment and export their paths whenever it is activated.

### New virtual environment

From anywhere, make a virtual environment with ExtraSWIFTLightcones (in editable mode), the dependencies of the package, [examples](./examples) and tests, and the shell redshift files

```
git clone https://github.com/Will-McD/ExtraSWIFTLightcones.git
bash ExtraSWIFTLightcones/venv_scripts/make_venv.sh /path/to/environment
source /path/to/environment/bin/activate
```

If no path is given the environment is made in `ExtraSWIFTLightcones/extra_swift_lightcones_env`.

On COSMA, build the virtual environment with the pre-built wheels of mpi4py and h5py for COSMA's MPI and parallel HDF5 instead 

```
bash ExtraSWIFTLightcones/venv_scripts/make_cosma_env.sh /path/to/environment
```
### Existing environment

With your environment (venv or conda) activated, install the package and its requirements, then download the shell redshift files 

```
cd ExtraSWIFTLightcones
pip install .                 # the package and its requirements
pip install ".[examples]"     # also the requirements of the examples
pip install ".[all]"          # also the requirements of the examples and tests
extra_swift_lightcones-configure
```

Use `pip install -e .` for an editable install while developing.

`extra_swift_lightcones-configure` downloads the shell redshift files to `<environment>/share/extra_swift_lightcones/redshifts` and adds `L1_REDSHIFTS_FILENAME` and `L2P8_REDSHIFTS_FILENAME` to the environment's activate script (`bin/activate` for a venv, `etc/conda/activate.d` for conda). Re-activate the environment to set them. Running it again is safe, use `--force` to download the files again, `--dest_dir` to download them somewhere else and `--no_activate` to leave the activate script unchanged. 

pip itself can't run a step after installing a package, so if `extra_swift_lightcones-configure` isn't run the files are downloaded the first time they are needed instead. Compute nodes often have no internet access, so run it on a login node before submitting jobs.

### MPI support

To install mpi4py with MPI support requires mpi4py and an MPI enabled build of h5py. This is necessary for the full use of the SnapshotLightcone class, BeamProjection class and generating binary masks of haloes (mask_halos.py). 

The mpi4py package installed from PyPI needs an MPI library at run time. If your system has none (e.g. on a laptop), install one into the environment with `pip install mpich` (or `pip install openmpi`). `extra_swift_lightcones-configure` doesn't need MPI.

See instructions for installing lightconeIO with MPI support: [LightconeIO with MPI support](https://lightconeio.readthedocs.io/en/latest/installation.html#with-mpi-support)

### Additional Data 

To add the shell redshift files to another existing environment 

```
bash ExtraSWIFTLightcones/venv_scripts/shell_redshifts.sh /path/to/environment
```

Additionally to further speed up the rotation of HEALPix maps, download the HEALPix pixel weights and add their paths to your virtual environment. 

```
cd ./ExtraSWIFTLightcones
bash venv_scripts/healpix_pixel_weights.sh
```


## Generating Lightcones from SWIFT Snapshots

The `SnapshotLightcone` class is used to create particle lightcones from SWIFT snapshots. The output data is structured the same as given by `lightcone_io.ParticleLightcone` class objects. 
The `SnapshotLightcone` builds new unique lightcones by slicing together snapshots boxes. An observer is positioned with the snapshot box and the snapshot particle data used to populate the lightcone is determined purley via the comoving distance from the observer. 

These snapshot lightcones can be built in two different ways given by the `SnapshotBeam` class and `SnapshotAllSky` class. The former is used to produce a beam with an angular radius of 60 degrees, whislt the latter produces a lightcone for the full celestial sphere. 


`SnapshotBeam` constructs a lightcone by continously adding snapshot boxes along the beams line of sight, where for every time the observers past lightcone surpases one snapshot sidelength, new snapshot boxes are added along the line of sight to extened the lightcone. 

`SnapshotAllSky` constructs a lightcone by continously adding additional layers or shells of snapshot boxes about the observer every time the observers past lightcone surpases one snapshot sidelength. e.g. starting from one snapshot box the next layer will be a 3x3 cube, followed by a 5x5 cube and so on. 

### Orientation and structure of additional snapshot layers.
In both methods for each additional layer of snapshots added we re-orientate (or restructure) the snapshot at the SWIFT cell level to avoid exact replications of the same structure along a given line of sight. This restructuring is done by: 
    1. Applying a random periodic shift, along each axis, to the snapshot cells (and all particles within each cell), essentially giving the snapshot a 'new' structure. The shift is the size of either 0, 4, 8 or 16 cells. 
    2. Randomly mirroring the new snapshots coordinates about each axis
    3. Randomly rotating the snapshot by -90, 0, 90 or 180 degrees about each axis

We control how these re-oritentations are applied per snapshot side length with the `orientation_lock` parameters when initialising each method.  
`orientation_lock='none'`:      Every new snapshot box added has its own unique orientation, this creates discontinuties across all faces of each snapshot box added into the lightcone. Computationally the fastest but creates the most discontinuties, it is the best method for small lightcone beams or all-sky lightcones. 

`orientation_lock='sphere'`:    Apply the same orientation in sphereical layers ($`n`$), where each layer is computed as $`(n−½)L ≤ r < (n+½)L`$. A box crossing a spherical layer is used twice, once with each layer's orientation, and each copy keeps only the particles on its own side. The advantage is that each layer sits within it's own redshift range, but is computationally more expensive and can create instances of the same objects within a given snapshot appearing twice.  

`orientation_lock='cube'`:      Apply the same orientation in cubic layers ($`n`$), where each layer is computed as $`max(|r_x|, |r_y|, |r_z|) = n`$. Hence every layer is along a snapshots box face and no boxes are ever split across layers like in the sphereical model. However the boundary between lays is no longer defined by a set radius or redshift, instead the layers vary between 
$`(n+½)L`$ to $`(n+½)·1.73L`$ as the line of sight moves the mid point to the corners of a box. 

Within each flavor of `orientation_lock` the random selection of periodic shifts, mirroring and rotations is can be controlled by random number seed, `orientation_seed`. For the same lock and seed, the full-sky lightcone given by `SnapshotAllSky` can be reproduced by different pointings of the beam from `SnapshotBeam`. 

## Projections

## Additional tools
