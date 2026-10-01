#!/bin/env python
import numpy as np
import unyt
import collections
import lightcone_io.particle_reader as pr
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone
from .property_to_field_names import property_to_field, field_to_property
from .snapshot_units import apply_expected_units
from . import beam_plotting
from .beam_plotting import BeamPlot, round_up_10
import swiftsimio as sw
from swiftsimio.objects import cosmo_array


# particle types that can be placed in the slice
PARTICLE_TYPES = ("Gas", "DM")

# properties always read into the slice
SLICE_REQUIRED_PROPERTIES = {
    "Gas": ["Coordinates", "ExpansionFactors", "SmoothingLengths"],
    "DM": ["Coordinates", "ExpansionFactors"],
}

# properties always added to the mock snapshot, DM smoothing lengths are generated rather than read
MOCK_REQUIRED_PROPERTIES = {
    "Gas": ["Coordinates", "SmoothingLengths"],
    "DM": ["Coordinates", "Masses", "SmoothingLengths"],
}
GENERATED_MOCK_PROPERTIES = {
    "Gas": [],
    "DM": ["SmoothingLengths"],
}

# mock snapshot attribute holding each particle type
MOCK_PARTICLE_TYPE = {
    "Gas": "lightcone_gas",
    "DM": "lightcone_dm",
}

# fixed units of properties in the mock snapshot
MOCK_UNITS = {
    "Coordinates": "Mpc",
    "SmoothingLengths": "Mpc",
    "Masses": "Msun",
}

def check_ptype(ptype):
    """
    Raise ValueError if the particle type is not "Gas" or "DM".

    :param  ptype:  particle type
    :type   ptype:  str
    """
    if ptype not in PARTICLE_TYPES:
        raise ValueError(f"unknown ptype {ptype!r} not added yet, must be one of {PARTICLE_TYPES}")


# beam projection functions 
class BeamProjection:
    """
    Class for making simple projects of past lightcone beams. 
    """
    def __init__(self, vector, angular_diameter, redshift_range, cosmology=None, slice_thickness=None, store_snapshot_filename=None):
        """
        Define the beam and the slice through it to project.

        :param  vector:             direction vector as an array of 3 floats
        :type   vector:             numpy.ndarray
        :param  angular_diameter:   angular diameter in degrees
        :type   angular_diameter:   float
        :param  redshift_range:     redshift range to read
        :type   redshift_range:     sequence of two floats [z_min, z_max]
        :param  cosmology:          cosmology object of the simulation, or a path to a snapshot to read it from
        :type   cosmology:          astropy cosmology object or str
        :param  slice_thickness:    thickness of the slice [Mpc], length along the z-axis, of particles
        :type   slice_thickness:    int, float or unyt.unyt_quantity (units of Mpc or equivalent)
        """
        self._beam_vec = tuple(vector)
        self._diameter_deg = angular_diameter
        self._radius = np.deg2rad(angular_diameter/2)
        self._redshift_range = tuple(redshift_range)
        
        self.slice_thickness = None if slice_thickness is None else apply_expected_units(slice_thickness, unyt.Mpc) # size on z-axis
        
        # try apply cosmology 
        if isinstance(cosmology, str):
            cosmology = Snapshot_Cosmology_For_Lightcone(cosmology).COSMO

        self.cosmology = cosmology
       

        self._make_empty_flags()

        if store_snapshot_filename is not None:
            self._snapshot_filename=store_snapshot_filename

    def _make_empty_flags(self,):
        """
        Reset the slice properties, plot boundaries and snapshot filename.
        """
        # slice properties 
        self.particle_data = {ptype: None for ptype in PARTICLE_TYPES}    # particle data in the slice
        self.in_slice_boolean = {ptype: None for ptype in PARTICLE_TYPES} # track particles being added to the mask
        self._properties_added = {ptype: None for ptype in PARTICLE_TYPES} # track names of particle properties added to the slice

        # boundaries of the plot in coordinate space
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        self._zmin = None
        self._zmax = None
        
        self._projection_region = None
        self.axes_extent = None

        # snapshot filename used for mock data and cosmology
        self._snapshot_filename = None
    
    @property
    def gas_particle_data(self):
        """
        Gas particle data in the slice.
        """
        return self.particle_data["Gas"]

    @gas_particle_data.setter
    def gas_particle_data(self, value):
        self.particle_data["Gas"] = value

    @property
    def dm_particle_data(self):
        """
        DM particle data in the slice.
        """
        return self.particle_data["DM"]

    @dm_particle_data.setter
    def dm_particle_data(self, value):
        self.particle_data["DM"] = value
    
    def update_lc_particles(self, particle_data, particle_properties, slice_thickness, ptype):
        """
        Rotate the particles and keep only those in the slice.

        Returns a dict of the particle properties in the slice.

        :param  particle_data:          particle data read from the lightcone
        :type   particle_data:          dict
        :param  particle_properties:    properties of the particles to keep
        :type   particle_properties:    list
        :param  slice_thickness:        thickness of the slice [Mpc], length along the z-axis, of particles
        :type   slice_thickness:        unyt.unyt_quantity (units of Mpc or equivalent)
        :param  ptype:                  particle type, "Gas" or "DM"
        :type   ptype:                  str
        """

        # rotate the particles coordinates to produce consistent plots
        rotated_coords = self.rotate_lc_coordinates(particle_data["Coordinates"][:])

        # make a mask that is True when particle is in the selected slice
        in_slice = self.identify_particles_in_slice(rotated_coords, 1./particle_data["ExpansionFactors"].value - 1., slice_thickness)
        self.in_slice_boolean[ptype] = in_slice

        # update with 'in slice' mask         
        slice_particle_data = {"Coordinates": rotated_coords[in_slice]}
        
        # keep particles and properties that are in the slice
        for k in particle_data.keys():
            if k != "Coordinates" and k in particle_properties:
                slice_particle_data[k] = particle_data[k][in_slice]
        
        return slice_particle_data

    def rotate_lc_coordinates(self, coordinates):

        """
        Returns particle coordinates rotated so they are aligned with the vector (1,0,0).
        
        :param  coordinates:    particle coordinates, shape (N, 3)
        :type   coordinates:    unyt.unyt_array
        """
        if self._beam_vec==(1,0,0):
            return coordinates
        rot_matrix = self.rotation_matrix_from_vectors(v_to=np.array([1., 0., 0.]))
        return coordinates @ rot_matrix.T

    def identify_particles_in_slice(self, coordinates, redshift, slice_thickness=10*unyt.Mpc):
        """
        Define simple mask for all particles within slice, in terms of redshift and coordinates.
        
        Returns a boolean array, True if particle is in slice. 
        
        :param  coordinates:        rotated particle coordinates, shape (N, 3)
        :type   coordinates:        unyt.unyt_array
        :param  redshift:           redshift of each particle
        :type   redshift:           np.ndarray
        :param  slice_thickness:    thickness of the slice [Mpc], length along the z-axis, of particles
        :type   slice_thickness:    int, float or unyt.unyt_quantity (units of Mpc or equivalent)
        """

        dz = apply_expected_units(slice_thickness, unyt.Mpc).to_value("Mpc")
        z_coords = coordinates[:,-1].to_value("Mpc")
        slice_midpoint = 0.5 * (np.max(z_coords) + np.min(z_coords))

        coords_boolean = (z_coords >= slice_midpoint - dz/2) & (z_coords <= slice_midpoint + dz/2)
        redshift_boolean = (redshift >= self._redshift_range[0]) & (redshift <= self._redshift_range[-1])

        return redshift_boolean & coords_boolean

    def add_property_to_slice(self, dset_name, dset, ptype):
        """
        Add additional properties to the slice particle data.

        :param  dset_name:  name of the property
        :type   dset_name:  str
        :param  dset:       values of the property for all particles read, before selecting the slice
        :type   dset:       unyt.unyt_array or np.ndarray
        :param  ptype:      particle type, "Gas" or "DM"
        :type   ptype:      str
        """
        
        #check ptype is ok.
        check_ptype(ptype)
        
        if self.in_slice_boolean[ptype] is None or self.particle_data[ptype] is None:
            print(f"No {ptype} particles in slice, cannot add {dset_name}")
        
        if len(dset) != len(self.in_slice_boolean[ptype]):
            print("incorrect size of dataset")
            return
        if dset_name in self.particle_data[ptype]:
            print(f"replacing {dset_name} in existing data")
        self.particle_data[ptype][dset_name] = dset[self.in_slice_boolean[ptype]]
        self._properties_added[ptype].append(dset_name)

    def place_particles_in_slice(self, gas_particle_data,  gas_property_names, slice_thickness=None, dm_particle_data=None, dm_property_names=None, xy_buffer=10.):
        """
        Add particles in the slice to the lightcone

        :param  gas_particle_data:  gas particle data or a path to the particle lightcone
        :type   gas_particle_data:  lightcone_io.IndexedLightconeParticleType for 'Gas' or a str
        :param  gas_property_names: properties of gas particles to add to the slice. If None then essential properties are assumed
                                        ["Coordinates", "ExpansionFactors","SmoothingLengths"]
        :type   gas_property_names: list
        :param  slice_thickness:    thickness of the slice [Mpc], length along the z-axis, of particles
        :type   slice_thickness:    int, float or unyt.unyt_quantity (units of Mpc or equivalent)
        :param  dm_particle_data:   dark matter particle data or a path to the particle lightcone that contains dark matter particles.
        :type   dm_particle_data:   lightcone_io.IndexedLightconeParticleType for 'DM' or a str
        :param  dm_property_names:  properties of dm particles to add to the slice. If dm_particle_data != None and dm_property_names is None, 
                                        all properties are loaded by default ["Coordinates","ExpansionFactors"]
        :type   dm_property_names:  list
        :param  xy_buffer:          distance [Mpc] between the min and max position of a particle and the boundary of the slice.
        :type   xy_buffer:          float or int
        """ 

        # update units for the size of the slice 
        if slice_thickness is None:
            if self.slice_thickness is None:
                raise ValueError("slice_thickness must be given here or when creating the BeamProjection")
        
            slice_thickness = self.slice_thickness
        slice_thickness = apply_expected_units(slice_thickness, unyt.Mpc)

        for ptype, data, property_names in (("Gas", gas_particle_data, gas_property_names), ("DM", dm_particle_data, dm_property_names)):
            if data is None:
                self.particle_data[ptype] = None
                continue
            # add required properties where needed
            slice_properties = list(property_names) if property_names is not None else []
            for required_prop in SLICE_REQUIRED_PROPERTIES[ptype]:
                if required_prop not in slice_properties:
                    slice_properties.append(required_prop)
            self._properties_added[ptype] = slice_properties

            if isinstance(data, str):
                data = self._read_lightcone_particles(data, ptype, slice_properties)
            else:
                for prop in slice_properties: # sanity check properties against provided data
                    if prop not in data:
                        raise ValueError(f"{prop} not in {ptype} particle data")

            self.particle_data[ptype] = self.update_lc_particles(data, slice_properties, slice_thickness, ptype)

            # set limits of the beam from the gas, or from the DM if they are not already set
            if ptype == "Gas" or self.axes_extent is None:
                self._lightcone_limits(self.particle_data[ptype]["Coordinates"], axis_boundaries_buffer=xy_buffer)
        
    def _read_lightcone_particles(self, lightcone_filename, ptype, property_names):
        """
        Read the particles in the beam from a particle lightcone.

        Returns a dict of the particle properties.

        :param  lightcone_filename: path to the particle lightcone
        :type   lightcone_filename: str
        :param  ptype:              particle type, "Gas" or "DM"
        :type   ptype:              str
        :param  property_names:     properties to read
        :type   property_names:     list
        """
        print(f"\nLoading {ptype} particle data ....")
        lightcone = pr.IndexedLightcone(lightcone_filename)
        particle_data = lightcone[ptype].read(
            property_names=property_names,
            redshift_range=self._redshift_range,
            vector=self._beam_vec, radius=self._radius
        )
        print("\nread in {n_all} {ptype} particles".format(n_all=len(particle_data["ExpansionFactors"]), ptype=ptype))
        return particle_data

    def _lightcone_limits(self, coordinates, axis_boundaries_buffer=10):
        self._xmin = np.amin(coordinates[:,0])
        self._xmax = np.amax(coordinates[:,0])
        self._ymin = np.amin(coordinates[:,1])
        self._ymax = np.amax(coordinates[:,1])
        self._zmin = np.amin(coordinates[:,2])
        self._zmax = np.amax(coordinates[:,2])
        
        # ensure that the projected region  == the extent of the projected image

        self._projection_region = [
            self._xmin.to_value("Mpc")-axis_boundaries_buffer,
            self._xmax.to_value("Mpc")+axis_boundaries_buffer,
            self._ymin.to_value("Mpc")-axis_boundaries_buffer,
            self._ymax.to_value("Mpc")+axis_boundaries_buffer,
        ]
        self.axes_extent = list(self._projection_region)
        self.axis_buffer=axis_boundaries_buffer

    def project_properties(self, project_particle_properties, snapshot_filename=None, resolution=1024, assign_units=None, ptype="Gas", periodic=True, parallel=False, weight=None):
        
        """
        Project the selected properties for a given particle type. Without a weight these projections are surface densities. 

        Returns a list of 2D histograms, one per property, for a given particle type. 
        

        :param  project_particle_properties:    properties to project
        :type   project_particle_properties:    list
        :param  snapshot_filename:              path to a snapshot, used for its metadata and cosmology
        :type   snapshot_filename:              str
        :param  resolution:                     number of pixels along each axis of the projection
        :type   resolution:                     int
        :param  assign_units:                   units of each property. If None, use the units of the slice data
        :type   assign_units:                   list
        :param  ptype:                          particle type, "Gas" or "DM"
        :type   ptype:                          str
        :param  weight:                         property to weight by, e.g. "Masses" for mass weighted projections. 
                                                    If None, project surface densities
        :type   weight:                         str
        :param  periodic:                       If True, treat the snapshot box as periodic when projecting. Periodic copies 
                                                    of the particles appear if the projected region is larger than the box
        :type   periodic:                       boolean
        :param  parallel:                       If True, use parallel backend of swiftsimio with projections. 
        :type   parallel:                       boolean
        """

        # check ptype is ok. 
        check_ptype(ptype)
        
        if self.cosmology is None:
            self.cosmology=Snapshot_Cosmology_For_Lightcone(snapshot_filename).COSMO
            
            # update snapshot name stored if used for cosmology
            self._snapshot_filename = snapshot_filename
        
        # update snapshot filename if a new one is passed 
        if snapshot_filename is not None:
            if snapshot_filename != self._snapshot_filename:
                self._snapshot_filename = snapshot_filename

        if weight is None:
            return self._project(list(project_particle_properties), self._snapshot_filename, resolution, assign_units, ptype, periodic, parallel)

        particle_data = self.particle_data[ptype]
        if particle_data is None or weight not in particle_data:
            raise ValueError(f"{weight} values not in slice")

        # add each property multiplied by the weight to the slice, removed again after projecting
        weighted_names = [f"{prop}Times{weight}" for prop in project_particle_properties]
        for prop, weighted_name in zip(project_particle_properties, weighted_names):
            if prop not in particle_data:
                raise ValueError(f"{prop} values not in slice")
            if weighted_name in particle_data:
                raise ValueError(f"{weighted_name} is already in the slice, it is needed to weight {prop} by {weight}")
        for prop, weighted_name in zip(project_particle_properties, weighted_names):
            particle_data[weighted_name] = particle_data[prop] * particle_data[weight]
        properties_added = self._properties_added[ptype]
        self._properties_added[ptype] = properties_added + weighted_names
        try:
            *weighted_projections, weight_projection = self._project(weighted_names + [weight], self._snapshot_filename, resolution, None, ptype, periodic,parallel)
        finally:
            for weighted_name in weighted_names:
                del particle_data[weighted_name]
            self._properties_added[ptype] = properties_added

        # weighted mean, sum(q*w)/sum(w), in the units of q
        no_weight = weight_projection.value == 0
        projection_outputs = []
        for idx, (prop, weighted_projection) in enumerate(zip(project_particle_properties, weighted_projections)):
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = weighted_projection / weight_projection
            mean[no_weight] = 0
            units = particle_data[prop].units if assign_units is None else assign_units[idx]
            projection_outputs.append(mean.to(units))

        return projection_outputs

    def _project(self, particle_properties, snapshot_filename, resolution, assign_units, ptype, periodic, parallel):
        """
        Project the surface density of each property.

        Returns a list of 2D histograms, one per property.

        :param  particle_properties:    properties to project
        :type   particle_properties:    list
        :param  snapshot_filename:      path to a snapshot, used for its metadata
        :type   snapshot_filename:      str
        :param  resolution:             number of pixels along each axis of the projection
        :type   resolution:             int
        :param  assign_units:           units of each property. If None, use the units of the slice data
        :type   assign_units:           list
        :param  ptype:                  particle type, "Gas" or "DM"
        :type   ptype:                  str
        :param  periodic:               If True, treat the snapshot box as periodic when projecting
        :type   periodic:               boolean
        """
        # export to snapshot datatype to project with swiftsimio
        
        mock_snap_data, preferred_units = self.make_mock_snapshot(list(particle_properties), self._snapshot_filename, assign_units=assign_units, ptype=ptype)

        mock_particles = getattr(mock_snap_data, MOCK_PARTICLE_TYPE[ptype])
        
        #create region to project over
        plot_region= cosmo_array(
                self._projection_region, # if problems with alignment of frame and projection add a buffer to either increase or decrease this region
                unyt.Mpc,
                comoving=True,
                scale_factor=1.,
                scale_exponent=1,  # distances scale as a**1, so the scale exponent is 1
            )
        
        projection_outputs=[]
        for prop in particle_properties:
            field_name = property_to_field.get(prop, prop) # allow for custom properties 
            print(f"Projecting:\t{prop} [{field_name}]")
            proj = sw.visualisation.projection.project_pixel_grid(
                mock_particles,
                resolution=resolution,
                project=field_name,
                parallel=parallel,
                periodic=periodic,
                region=plot_region,
                )
            projection_outputs.append(proj)

        return projection_outputs

    def _add_lightcone_particles_to_snap(self, snap, particle_properties, assign_units, ptype):
        """
        Add the gas particles in the slice to a snapshot template as mock snapshot data, snap.lightcone_gas

        Returns a tuple, (snapshot, preferred units of each property).

        :param  snap:                   snapshot data
        :type   snap:                   swiftsimio.reader.SWIFTDataset
        :param  particle_properties:    properties to add. Note, required properties are appended to this list
        :type   particle_properties:    list
        :param  assign_units:           units of each property. If None, use the units of the slice data
        :type   assign_units:           list
        :param  ptype:                  particle type, "Gas" or "DM"
        :type   ptype:                  str
        """

        mock_name = MOCK_PARTICLE_TYPE[ptype]
        
        if hasattr(snap, mock_name):
            raise ValueError(f"Snapshot already has mock snapshot {ptype} particles!!!")
        
        particle_data = self.particle_data[ptype]
        
        # define mock fields
        mock_fields=["metadata"] + [property_to_field.get(prop, prop) for prop in particle_properties]

        # add required fields
        for required_prop in MOCK_REQUIRED_PROPERTIES[ptype]:
            if required_prop not in particle_properties and property_to_field[required_prop] not in mock_fields:
                mock_fields.append(property_to_field[required_prop])
                particle_properties.append(required_prop)

        # determine the preferred units of each property 
        preferred_units={}
        for idx, prop in enumerate(particle_properties):

            if prop in MOCK_UNITS:
                preferred_units[prop] = MOCK_UNITS[prop]
            elif assign_units is None:
                preferred_units[prop] = str(particle_data[prop].units.expr)
            else:
                preferred_units[prop] = assign_units[idx]
        
        mock_values = {"metadata": snap.metadata}
        for field_name in mock_fields[1:]:
            prop_name = field_to_property.get(field_name, field_name)
            if prop_name in GENERATED_MOCK_PROPERTIES[ptype]:
                mock_values[field_name] = self._generate_smoothing_lengths(particle_data["Coordinates"], snap.metadata.boxsize)
            else:
                 mock_values[field_name] = self._lc2cosmoarray(particle_data, prop_name, preferred_units[prop_name])
        
        MockTotal = collections.namedtuple("MockTotal", mock_fields)
        setattr(snap, mock_name, MockTotal(**mock_values))
        return snap, preferred_units


    def add_lc_gas_to_snap(self, snap, particle_properties, assign_units):
        """
        Add the gas particles in the slice to a snapshot as mock snapshot data, snap.lightcone_gas.

        Returns a tuple of (snapshot, preferred units of each property).

        :param  snap:                   snapshot data
        :type   snap:                   swiftsimio.reader.SWIFTDataset
        :param  particle_properties:    properties to add
        :type   particle_properties:    list
        :param  assign_units:           units of each property. If None, use the units of the slice data
        :type   assign_units:           list
        """
        return self._add_lightcone_particles_to_snap(snap, particle_properties, assign_units, "Gas")

    def add_lc_dm_to_snap(self, snap, particle_properties, assign_units):
        """
        Add dark matter particles in the slice to a snapshot template as mock snapshot data, snap.lightcone_dm
        Smoothing lengths are generated for the DM particles.

        Returns a tuple of (snapshot, preferred units of each property).

        :param  snap:                   snapshot data
        :type   snap:                   swiftsimio.reader.SWIFTDataset
        :param  particle_properties:    properties to add
        :type   particle_properties:    list
        :param  assign_units:           units of each property. If None, use the units of the slice data
        :type   assign_units:           list
        """
        return self._add_lightcone_particles_to_snap(snap, particle_properties, assign_units, "DM")


    @staticmethod
    def _generate_smoothing_lengths(coordinates, snapshot_boxsize):
        """
        Generate smoothing lengths for particles that do not have them. The particles are shifted so
        they are inside a box, which is the snapshot box unless the particles extend beyond it.

        Returns the smoothing lengths as a cosmo_array.
        :param  coordinates:        particle coordinates, shape (N, 3)
        :type   coordinates:        unyt.unyt_array
        :param  snapshot_boxsize:   side lengths of the snapshot box
        :type   snaps
        """

        # 1) shift position of particles so they are inside the snapshot box 
        # 2) generate smoothing lengths

        shifted_coords=unyt.unyt_array(np.zeros_like(coordinates), units=unyt.Mpc)
        ax_sidelengths=unyt.unyt_array(np.zeros(3, dtype=float), units=unyt.Mpc)
        for i in range(3):
            ax_coords = coordinates[:, i].to_value("Mpc")
            ax_min = np.min(ax_coords)
            if ax_min < 0: # shift particles on axes to only have positive values
                shifted_coords[:, i] += ax_coords - ax_min
            else:
                shifted_coords[:, i] += ax_coords

            if np.max(shifted_coords[:, i].to_value("Mpc")) < snapshot_boxsize[i].to_value("Mpc"):
                ax_sidelengths[i] = snapshot_boxsize[i].to_value("Mpc")
            else:
                ax_sidelengths[i] = round_up_10(np.max(shifted_coords[:, i].to_value("Mpc"))+0.1) # new box sidelengths go from 0-> max part location in lc

        def comoving_lengths(values):
            return cosmo_array(values, unyt.Mpc, comoving=True, scale_factor=1., scale_exponent=1)

        return sw.visualisation.generate_smoothing_lengths(
           comoving_lengths(shifted_coords.to_value("Mpc")),
           comoving_lengths(ax_sidelengths.to_value("Mpc")),
           kernel_gamma=1.8,
           neighbours=32,
           speedup_fac=2,
           dimension=3,
        )


    def make_mock_snapshot(self, particle_properties, snapshot_filename, assign_units=None, ptype="Gas"):
        """
        Load a small region of a snapshot and add the slice particles to it, so they can be projected with swiftsimio.

        Returns a tuple (snapshot, preferred units of each property)

        :param  particle_properties:    properties to add
        :type   particle_properties:    list
        :param  snapshot_filename:      path to the snapshot
        :type   snapshot_filename:      str
        :param  assign_units:           units of each property. If None, use the units of the slice data
        :type   assign_units:           list
        :param  ptype:                  particle type, "Gas" or "DM"
        :type   ptype:                  str
        """
        
        # check ptype is ok. 
        check_ptype(ptype)

        # sanity check properties
        for prop in particle_properties:
            if prop not in self._properties_added[ptype] and prop not in GENERATED_MOCK_PROPERTIES[ptype]:
                raise ValueError(f"{prop} values not in slice")

        # load a spatially constrained subset of snapshot particles to then replace with lightcone data
        mask = sw.mask(snapshot_filename)
        # The full metadata object is available from within the mask
        boxsize = mask.metadata.boxsize
        load_region = [[1 * b/b.value, 2 * b/b.value] for b in boxsize]
        # Spatially constrain and load the snapshot
        mask.constrain_spatial(load_region)
        snap = sw.load(snapshot_filename, mask=mask)
        
        return self._add_lightcone_particles_to_snap(snap, particle_properties, assign_units, ptype)
    

    @staticmethod
    def _lc2cosmoarray(particle_data, property_name, property_units):
        """
        Convert a lightcone property to a comoving swiftsimio cosmo_array.

        :param  particle_data:  particle data
        :type   particle_data:  dict
        :param  property_name:  name of the property
        :type   property_name:  str
        :param  property_units: units to convert the property to
        :type   property_units: str
        """
        return cosmo_array(
            particle_data[property_name].to_value(property_units),
            particle_data[property_name].to(property_units).units,
            comoving=True, # assume comoving for all lightcone properties 
            scale_factor=1., scale_exponent=1 # assume defined at z=0 and has no additional scale factor effects
            )

    def rotation_matrix_from_vectors(self, v_to=np.array([1., 0., 0.])):
        """
        Returns a rotation matrix which rotates the beam vector onto v_to.

        :param  v_to:   vector to rotate onto
        :type   v_to:   numpy.ndarray
        """
        v_from=np.array([self._beam_vec[0], self._beam_vec[1], self._beam_vec[2]])
        a = np.asarray(v_from, dtype=float)
        b = np.asarray(v_to, dtype=float)

        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        v = np.cross(a, b)
        c = np.dot(a, b)

        # vectors are parallel
        if np.isclose(c, 1):
            return np.eye(3)

        # vectors are anti-parallel
        if np.isclose(c, -1):
            # choose any axis perpendicular to a
            axis = np.cross(a, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-10:
                axis = np.cross(a, [0, 1, 0])
            axis /= np.linalg.norm(axis)

            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])

            return np.eye(3) + 2 * K @ K

        s = np.linalg.norm(v)

        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

        return np.eye(3) + K + K @ K * ((1 - c) / s**2)

    def beam_plot(self):
        """
        Plotter for this beam, using the current cosmology, angular diameter, redshift range and axes extent.
        """
        return BeamPlot(self.cosmology, self._diameter_deg, self._redshift_range, self.axes_extent)


    def split_beam_plot(self, numb_wedges, projection_data, colour_maps,
            axs=None, filename=None,
            angular_diameter=None, cosmology=None, redshift_range=None, 
            axes_extent=None, update_badcol=True, figsize=(7,7), titles=None, norms=None, **kwargs):
        
        """
        Create plot of the whole beam, split into seperate wedges. 

        Returns list of each projected wedge.  

        :param  numb_wedges:        number of wedges or different projections to show in the beam
        :type   numb_wedges:        int
        :param  projection_data:    nested list containing different segments of the beam the 2D array to plot and
                                        a tuple with the min and max values shown in the img [2D array, (min, max)]
        :type   projection_data:    list
        :param  colour_maps:        list of colour maps for each segment of the beam
        :type   colour_maps:        list
        :param  axs:                Default=None, If None create a new axes for the plot, otherwise plot onto the axes given
        :type   axs:                matplotlib.axes._axes.Axes
        :param  filename:           None, if given, write plot to this file
        :type   filename:           str
        :param  angular_diameter:   total angular diameter of beam in degrees
        :type   angular_diameter:   float
        :param  cosmology:          the simulation's cosmology model
        :type   cosmology:          astropy cosmology object, astropy.cosmology.flrw.w0wacdm.w0waCDM
        :param  redshift_range:     maximum and minimum redshift of the beam shown
        :type   redshift_range:     list, np.ndarray or tuple
        :param  axes_extent:        Extent of the plot's axes, in coordinate space.
                                        If None, then use the extent defined by the coordinates of the particles added to the slice.
        :type   axes_extent:        list
        :param  update_badcol:      If true, modify all colour maps so that the minimum, nan and None values are set to black
        :type   update_badcol:      boolean
        :param  figsize:            size of the figure, used if axs is None
        :type   figsize:            tuple
        :param  titles:             title of each wedge
        :type   titles:             list
        :param  norms:              colour normalisation of each wedge: "log", "linear" or a matplotlib.colors.Normalize. 
                                        If None, all are "log"
        :type   norms:              list
        :param  kwargs:             All additional arguments to be passed onto the add_wedge and add_beam_axes functions.
        :type   kwargs:             dict
        """

        # update slice information and call predefined values where needed
        if redshift_range is not None:
            self._redshift_range = tuple(redshift_range)
        elif self._redshift_range is None:
            raise ValueError("Redshift range is not defined")
        
        # check plots extent definition
        if axes_extent is not None:
            self.axes_extent = axes_extent
        elif self.axes_extent is None:
            raise ValueError("Axes range is not defined")

        # check diameter definition
        if angular_diameter is not None:
            self._diameter_deg = angular_diameter
        elif self._diameter_deg is None:
            raise ValueError("angular diameter is not defined")

        #check cosmology object is present
        if cosmology is not None:
            self.cosmology = cosmology
        elif self.cosmology is None:
            if self._snapshot_filename is None:
                raise ValueError("Cosmology is not defined")
            self.cosmology = Snapshot_Cosmology_For_Lightcone(self._snapshot_filename).COSMO

        # return plot 
        return self.beam_plot().split_beam_plot(numb_wedges, projection_data, colour_maps,axs=axs, filename=filename, update_badcol=update_badcol, figsize=figsize, titles=titles, norms=norms, **kwargs)

    def add_wedge(self, ax, *args, **kwargs):

        """
        Add a wedge onto the plot, drawn with the stored axes extent.
        """
        return self.beam_plot().add_wedge(ax, *args, **kwargs)

    def add_beam_axes(self, ax, *args, **kwargs):
        """
        Add axes, labels and ticks to the split beam plot, using the stored cosmology. 
        """
        return self.beam_plot().add_beam_axes(ax, *args, **kwargs)
   
    # plotting helpers kept as methods for existing code
    filter_kwargs = staticmethod(beam_plotting.filter_kwargs)
    arc_xy = staticmethod(beam_plotting.arc_xy)
    minor_tick_spacer = staticmethod(beam_plotting.minor_tick_spacer)
    labels_loc = staticmethod(beam_plotting.labels_loc)
    offset_point_from_tick = staticmethod(beam_plotting.offset_point_from_tick)
    points_to_data = staticmethod(beam_plotting.points_to_data)
