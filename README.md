# SWIFTLET: SWIFT Lightcone Extension Tools

[![Build and test](https://github.com/Will-McD/ExtraSWIFTLightcones/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/Will-McD/ExtraSWIFTLightcones/actions/workflows/build_and_test.yml)
[![Documentation Status](https://readthedocs.org/projects/extraswiftlightcones/badge/?version=latest)](https://extraswiftlightcones.readthedocs.io/en/latest/)
[![Version](https://img.shields.io/github/v/release/Will-McD/ExtraSWIFTLightcones?include_prereleases&sort=semver&label=version)](https://github.com/Will-McD/ExtraSWIFTLightcones/releases)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](https://github.com/Will-McD/ExtraSWIFTLightcones/blob/main/.github/workflows/build_and_test.yml)
[![License](https://img.shields.io/github/license/Will-McD/ExtraSWIFTLightcones)](./PublicLicense)

SWIFTLET builds custom lightcones from the snapshots and SOAP halo catalogues of [SWIFT](https://swift.strw.leidenuniv.nl/docs/index.html) simulations (FLAMINGO and COLIBRE). It works in tandem with [LightconeIO](https://lightconeio.readthedocs.io/en/latest/).

**📖 Documentation: <https://extraswiftlightcones.readthedocs.io>**

- **Lightcones from snapshots**: all-sky lightcones or pencil beams of particles, filled from the snapshot closest in redshift at each distance, reading only the SWIFT cells inside the lightcone.
- **Halo lightcones** from SOAP catalogues, written in the format of LightconeIO's halo lightcones.
- **Control over repeated structures**: re-oriented copies of the snapshot box, orientation locks and seeds, and checks for close repeats before anything is read.
- **Projections, HEALPix maps and halo masks** of snapshot lightcones and of FLAMINGO's lightcones.


## Installation

```bash
git clone https://github.com/Will-McD/ExtraSWIFTLightcones.git
bash ExtraSWIFTLightcones/venv_scripts/make_venv.sh /path/to/environment   # a new environment with everything
source /path/to/environment/bin/activate
```

Or, into an environment you already have:

```bash
cd ExtraSWIFTLightcones
pip install -e ".[all]"      # the package and the requirements of the examples and tests
swiftlet-configure           # downloads the FLAMINGO shell redshift files and sets the environment variables
```

On COSMA, use `venv_scripts/make_cosma_env.sh` instead, which uses COSMA's MPI builds of mpi4py and h5py. See the [installation guide](https://extraswiftlightcones.readthedocs.io/en/latest/installation.html) for MPI support, the number of threads and the optional data files.


## Quick example

A pencil beam of gas particles, 2.5° in radius, out to redshift 0.3, from FLAMINGO L1_m10:

```python
from swiftlet import Simulation, SnapshotBeam

sim = Simulation.from_label("L1000N0900", "HYDRO_FIDUCIAL")
beam = SnapshotBeam(simulation_object=sim, beam_vector=(0, 0, 1))

particles = beam.place_snapshot_particles_in_shell(
    lightcone_redshift_range=(0.05, 0.3),
    property_names=["Masses", "Temperatures"],
    ang_radius_deg=2.5,
    particle_types=["PartType0"],
)
gas = particles["PartType0"]    # Coordinates, ExpansionFactors, SnapshotNumber, Masses, Temperatures, ...
```

The [quick start](https://extraswiftlightcones.readthedocs.io/en/latest/quickstart.html) continues with halo lightcones and projections.


## Documentation

| | |
|---|---|
| [Quick start](https://extraswiftlightcones.readthedocs.io/en/latest/quickstart.html) | build a beam of particles and haloes, and then project it |
| [User guide](https://extraswiftlightcones.readthedocs.io/en/latest/user_guide/index.html) | how lightcones are built, tile orientations and locks, repeated structures, simulations and suites, projections |
| [Examples](https://extraswiftlightcones.readthedocs.io/en/latest/examples.html) | the scripts in [`examples/`](./examples) |
| [API reference](https://extraswiftlightcones.readthedocs.io/en/latest/api/index.html) | every public class and function |
| [Tests](https://extraswiftlightcones.readthedocs.io/en/latest/tests/index.html) | what the test suite checks, and the developer tools |
| [Glossary](https://extraswiftlightcones.readthedocs.io/en/latest/glossary.html) | the terms used in the docs: tiles, layers, cells, shells, close repeats and more |

<details>
<summary>

## Tests

</summary>



The tests need no simulation data: they write small fake snapshots and SOAP catalogues each time they run.

```bash
pip install -e ".[test]"
python -m pytest tests
```

See [Tests](https://extraswiftlightcones.readthedocs.io/en/latest/tests/test_summary.html) for what each test file checks and how to run the MPI tests.

</details>

<details>
<summary>

## Contributing

</summary>

Contributions are welcome as pull requests; please read [CONTRIBUTING.md](./CONTRIBUTING.md). The documentation is in [`docs/`](./docs) and is built by Read the Docs from this repository.

</details>

<details>
<summary>
    
## Use of AI

</summary>

We acknowledge the assistance of AI (Claude Code, Anthropic 2026) in writing (and formatting) the documentation (the docstrings, the pages in `docs/` and this `README.md`), the test suite (`tests/`), the general set-up scripts (`swiftlet/threads.py`, `swiftlet/config.py` and `swiftlet/__init__.py`), the diagram examples (`examples/diagrams/`) and the placement examples (`examples/snapshot_beam_halo_placement.py`, `examples/snapshot_beam_particle_placement.py` and `examples/snapshot_number_density_comparison.py`).
All use of AI was supervised and any contribution to this module was first vetted by humans.

</details>

<details>
<summary>

## License

</summary>

GNU General Public License v3, see [PublicLicense](./PublicLicense).

</details>

