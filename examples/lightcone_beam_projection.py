#!/bin/env python
import numpy as np
import unyt
import lightcone_io.particle_reader as pr
from swiftlet.lightcone_projections import BeamProjection
import cmasher as cmr # noqa: F401 registers the "cmr.*" colour maps used by name
from pathlib import Path

"""
Create 2 different projections of slices through a beam from an observers past lightcone. 

Read in gas and dark matter particles from a lightcone of the fiducial L1_m9 FLAMINGO simulation (i.e. L1000N1800/HYDRO_FIDUCIAL). 
Create a split beam projection for:
    1) The surface mass density of gas, dark matter and total mass within the beam. 
    2) The mass weighted and redshift weighted gas temperature within the lightcone. 
        This demonstrates the  mass weighting function of BeamProjection class. 
"""

def example_fig1(BP, particle_lightcone, gas_particle_data, gas_property_names, dm_particle_data):
    # project a slice through a beam to show the surface mass density within the lightcone

    # add particle data to the slice 
    BP.place_particles_in_slice(gas_particle_data, gas_property_names, dm_particle_data=dm_particle_data, dm_property_names=dm_property_names)

    #project the gas and dm slices 
    gas_projections = BP.project_properties(["Masses"], ptype="Gas", assign_units=["Msun"], resolution=1024)
    cdm_projections = BP.project_properties(["Masses"], ptype="DM", assign_units=["Msun"], resolution=1024)

    # projections come out in the same order they are entered
    gas_surface_density=gas_projections[0].to_value("Msun/Mpc**2")
    cdm_surface_density=cdm_projections[0].to_value("Msun/Mpc**2")
    total_surface_density=gas_surface_density+cdm_surface_density 


    # create split beam plot of the total different surface density projections 

    # define the min and max values for each projection 
    pix_percentile = np.percentile(total_surface_density[total_surface_density>0], [10,99.5])
    pix_min, pix_max = pix_percentile[0], pix_percentile[1]
    projection_data=[
        [gas_surface_density, (pix_min*0.2, pix_max)],
        [total_surface_density, (pix_min, pix_max)],
        [cdm_surface_density, (pix_min*0.8, pix_max)]
    ]

    numb_wedges=len(projection_data)
    colour_maps=[
        "magma",
        "cubehelix",
        "cmr.eclipse"
    ]
    output_filename = "./example_outputs/beam_projection/split_beam_surface_density_example.png"
    wedge_imgs = BP.split_beam_plot(
        numb_wedges, projection_data, colour_maps, filename=output_filename, 
        minor_tick_kwargs={"color":"k", "lw":0.6}, tick_label_kwargs={"rotation":"auto"}, overlay_grid=(True, False, False),
        titles=["Gas", "Gas+DM", "DM"]
        )

    del dm_particle_data
    print(f"Example image 1 saved as: {output_filename}")


def example_fig2(BP, particle_lightcone, gas_particle_data, gas_property_names):
    # Repeat example 1 but now with custom datasets to show mass and redshift dependance
    
    # add particle data to the slice 
    BP.place_particles_in_slice(gas_particle_data, gas_property_names)

    #project the gas and dm slices 
    gas_projections = BP.project_properties(["Masses"], ptype="Gas", assign_units=["Msun"], resolution=1024)

    # projections come out in the same order they are entered
    gas_surface_density=gas_projections[0].to_value("Msun/Mpc**2")

    # mass weighted temperature
    mass_weighted_temp = gas_particle_data["Temperatures"] * gas_particle_data["Masses"]
    mass_weighted_temp.convert_to_units("K * Msun")

    # redshifted weighted temperature 
    z_weighted_temp = gas_particle_data["Temperatures"] * gas_particle_data["Masses"] / (1./gas_particle_data["ExpansionFactors"].value-1.)**2
    z_weighted_temp.convert_to_units("K*Msun")

    #add properties to slice dataset
    BP.add_property_to_slice(dset_name="MassWeightedTemp", dset=mass_weighted_temp, ptype="Gas")
    BP.add_property_to_slice(dset_name="RedshiftWeightedTemp", dset=z_weighted_temp, ptype="Gas")

    temp_projections = BP.project_properties(['Temperatures',"MassWeightedTemp", "RedshiftWeightedTemp"], ptype="Gas", assign_units=["K", "K*Msun", "K*Msun"], resolution=1024)

    # projections come out in the same order they are entered
    Temp_surface_density=temp_projections[0].to_value("K/Mpc**2")
    TempM_surface_density=temp_projections[1].to_value("K*Msun/Mpc**2")
    TempZ_surface_density=temp_projections[2].to_value("K*Msun/Mpc**2")

    TempM_surface_density[gas_surface_density>0]/=gas_surface_density[gas_surface_density>0]
    TempM_surface_density[gas_surface_density==0] = 0

    TempZ_surface_density[gas_surface_density>0]/=gas_surface_density[gas_surface_density>0]
    TempZ_surface_density[gas_surface_density==0] = 0

    # combine projections into list and set to the same limits 
    projection_data=[
        [Temp_surface_density , np.percentile(Temp_surface_density[Temp_surface_density >0], [5,99.75])],
        [TempM_surface_density, np.percentile(TempM_surface_density[Temp_surface_density>0], [5,99.75])],
        [TempZ_surface_density, np.percentile(TempZ_surface_density[Temp_surface_density>0], [5,99.75])]
    ]

    numb_wedges=len(projection_data)
    # give each the same colour map for a fair comparison 
    colour_maps=[
        "magma",
        "magma",
        "magma"
    ]
    
    output_filename = "./example_outputs/beam_projection/split_beam_temp_example.png"

    wedge_imgs = BP.split_beam_plot(
        numb_wedges, projection_data, colour_maps, filename=output_filename,
        minor_tick_kwargs={"color":"k", "lw":0.6}, tick_label_kwargs={"rotation":0}, overlay_grid=(True, False, False), major_tick_length=3.5,
        titles=["Temperature", "Mass weighted\nTemperature", r"Temperature$\,/\,$Redshift$^2$"],
        redshift_label_offset=(0,-15,0), comoving_label_offset=(0,15,0), title_kwargs={'fontsize':5}
        )
    print(f"Example image 2 saved as: {output_filename}")

if __name__ == "__main__":

    # sim base dir
    boxsize_resolution="L1000N1800"
    sim="HYDRO_FIDUCIAL"
    base_dir="/cosma8/data/dp004/flamingo/Runs/{LN}/{sim}".format(LN=boxsize_resolution, sim=sim)

    # define output directory
    output_dir="./example_outputs/lightcone_beam_projection"
    
    # ensure output directory exists
    directory_path = Path(output_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    # define snapshot filename, we will get information about the simuations cosmology from this later 
    snapshot_filename = "{base_name}/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.hdf5".format(base_name=base_dir,  snap_nr=77)
    snapshot_dir = "{base_name}/snapshots".format(base_name=base_dir)

    # Specify one file from the spatially indexed lightcone particle data
    lightcone_nr=0
    input_filename = "{base_name}/particle_lightcones/lightcone{lc_nr}_particles/lightcone{lc_nr}_0000.0.hdf5".format(base_name=base_dir, lc_nr=lightcone_nr)

    # Part of the sky to plot
    vector = (1.0, 0.0, 0.0)  # Vector pointing at a spot on the sky
    radius = np.radians(12.0) # Angular radius around that spot
    redshift_range = (0.01, 0.11) # Redshift range to plot (set to None for all redshifts in the lightcone)

    # Open the lightcone
    lightcone = pr.IndexedLightcone(input_filename)

    # load particle information for both gas and dark matter particles in the beam
    gas_property_names = ["Coordinates","Masses", "ExpansionFactors","SmoothingLengths","Temperatures"]
    dm_property_names = ["Coordinates","Masses", "ExpansionFactors"]
    
    # define a slice through the same beam  
    slice_z_width= 10 * unyt.Mpc # how thick the slice is 
    
    print(snapshot_dir)
    
    BP=BeamProjection(
        vector=vector, 
        angular_diameter=np.rad2deg(2*radius), 
        redshift_range=redshift_range, 
        slice_thickness=slice_z_width, 
        cosmology=snapshot_dir, # can pass snapshot base directory instead of cosmology object. 
        store_snapshot_filename=snapshot_filename # store for later to generate metadata and mock snapshot objects. 
        )

    gas_particle_data = lightcone["Gas"].read(
        property_names=gas_property_names,
        redshift_range=redshift_range,
        vector=vector,
        radius=radius
    )

    dm_particle_data = lightcone["DM"].read(
        property_names=("Coordinates","Masses", "ExpansionFactors"),
        redshift_range=redshift_range,
        vector=vector, 
        radius=radius
    )

    # Example 1 
    # project a slice through a beam to show the surface mass density within the lightcone
    example_fig1(BP, lightcone, gas_particle_data, gas_property_names, dm_particle_data)


    #Example 2 
    # Repeat the above example but now with custom datasets to show mass and redshift dependance
    example_fig2(BP, lightcone, gas_particle_data, gas_property_names)

