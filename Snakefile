import re 
import string 
import os
import hjson
import yaml
import config 

__author__ = ["Ramón Ángel Ruiz Fernández"]
__email__ = ["rruizfer@cern.ch"]

TUPLES = config.user["tuples_path"]
TUPLES_TRAINING = config.user["raw_tuples"]

include: 'IFT/Snakefile'
include: 'selection/Snakefile'
include: 'time_resolution/Snakefile'
include: 'tagging/Snakefile'
