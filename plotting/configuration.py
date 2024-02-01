import matplotlib as mpl

#Axis
mpl.rcParams['errorbar.capsize']                = 2
#Axis
mpl.rcParams["axes.titlesize"              ] = "large"
mpl.rcParams[ "axes.labelsize"             ] = 13
mpl.rcParams[ "axes.linewidth"             ] = 1.
mpl.rcParams[ "axes.facecolor"             ] = "white"
# mpl.rcParams[ "axes.formatter.min_exponent"] = 3
# mpl.rcParams[ "axes.unicode_minus"         ] = False
mpl.rcParams[ "xaxis.labellocation"        ] = "right"
mpl.rcParams[ "yaxis.labellocation"        ] = "top"
mpl.rcParams[ "text.usetex"                ] = False
#Errorbars
mpl.rcParams["errorbar.capsize"            ] = 2.5
#Figure
mpl.rcParams["figure.facecolor" ] =  "white"
mpl.rcParams["figure.autolayout"] =  True
# mpl.rcParams["font.family"      ] =  "serif"
mpl.rcParams["font.family"      ] =  "Times New Roman"
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams["font.size"        ] = 13
# Lines
mpl.rcParams["lines.linewidth"      ] = 1.3
mpl.rcParams["lines.markeredgewidth"] = 1.3
# Format
mpl.rcParams["savefig.bbox"       ] = "tight"
mpl.rcParams["savefig.format"     ] = "pdf"
# Y Ticks]
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.right"        ] = True
mpl.rcParams["ytick.direction"    ] = "in"
mpl.rcParams["ytick.labelsize"    ] = 13
# X Ticks]
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.top"          ] = True
mpl.rcParams["xtick.direction"    ] = "in"
mpl.rcParams["xtick.labelsize"    ] = 13

