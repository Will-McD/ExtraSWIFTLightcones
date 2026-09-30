# ExtraSWIFTLightcones

This module is designed to provide additional tools for the post-processing and the visualization of [SWIFT](https://swift.strw.leidenuniv.nl/docs/index.html) lightcones. Specifically, this module provides the tools to generate new lightcones from the swift snapshots or  to either [LightconeIO]([https://github.com/jchelly/LightconeIO](https://lightconeio.readthedocs.io/en/latest/#)) and [SWIFTsimIO](https://swiftsimio.readthedocs.io/en/latest/), therefore it requires the both the aforementioned packages 

Further information about the FLAMINGO lightcones: https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html

## Installation

### MPI support and Examples

To install mpi4py with MPI support requires mpi4py and an MPI enabled build of h5py. This is necessary for the full use of the SnapshotLightcone class, BeamProjection class and generating binary masks of haloes (mask_halos.py). 

See instructions for installing lightconeIO with MPI support: [LightconeIO with MPI support](https://lightconeio.readthedocs.io/en/latest/installation.html#with-mpi-support)


### Virtual environment and Examples

To install LightconeIO, SWIFTsimIO and be able to utilize all of the [examples](./examples) build the ExtraSWIFTLightcones virtual environment. 
From the ExtraSWIFTLightcones directory build the examples virtual environment
```
bash venv_scripts/make_venv.sh
```

If using COSMA build a ExtraSWIFTLightcones virtual environment by using the pre-existing wheels for mpi4py and h5py. 
From the ExtraSWIFTLightcones directory install a COSMA based virtual environment
```
bash venv_scripts/make_cosma_venv.sh
```

### Additional Data 

The FLAMINGO lightcone shell redshifts are added to [./data/redshifts](./data/redshifts) on setup. To download the shell redshifts text files again and add the paths to a specific virtual environment replace the first line of [./venv_scripts/shell_redshifts.sh](./venv_scripts/shell_redshifts.sh) with the path to the environment they will be added too. 

```
venv_name="/path/to/environment"
```

then to run add the files to the virtual environment

```
cd ./ExtraSWIFTLightcones
bash venv_scripts/shell_redshifts.sh
```

Additionally to further speed up the rotation of HEALPix maps, download the HEALPix pixel weights and add their paths to your virtual environment. 

```
cd ./ExtraSWIFTLightcones
bash venv_scripts/healpix_pixel_weights.sh
```


## Generating Lightcones from SWIFT Snapshots

## Projections

## Additional tools
