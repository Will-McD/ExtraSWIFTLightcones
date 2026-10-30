#!/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

"""
Function to import basic matplotlib settings for example scripts
"""

def plot_settings():

    # Line Properties
    plt.rcParams["lines.linewidth"]=1.5 
    # Font options
    plt.rcParams["font.size"]=8
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["legend.labelspacing"]=0.25
    plt.rcParams["legend.columnspacing"]=0.75
    plt.rcParams["legend.borderpad"]=0.3
    plt.rcParams["legend.borderaxespad"]=0.7
    plt.rcParams["legend.fontsize"]=10
    # Figure layour settings
    plt.rcParams["figure.constrained_layout.use"] =True
    plt.rcParams["figure.constrained_layout.h_pad"] =0.005
    plt.rcParams["figure.constrained_layout.w_pad"] =0.005
    plt.rcParams["figure.constrained_layout.hspace"]=0.005
    plt.rcParams["figure.constrained_layout.wspace"]=0.005
    # axes font settings
    mpl.rcParams['axes.labelsize']=10
    mpl.rcParams['figure.labelsize']=10
    mpl.rcParams['xtick.labelsize']=10
    mpl.rcParams['ytick.labelsize']=10
    # axes tick settings
    plt.rcParams["xtick.direction"] ='in'
    plt.rcParams["ytick.direction"] ='in'
    plt.rcParams["xtick.top"]=True
    plt.rcParams["ytick.right"]= True


# new colour maps for examples: 

cmap_MCG = LinearSegmentedColormap.from_list("magenta_cyan_gold", ["#8b008b", "#008c99", "#b8860b"])

