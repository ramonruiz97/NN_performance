import re
import hjson
import numpy as np


def cuts_and(*args):
  result = list(args)
  result = [a for a in args if a]
  return '(' + ') & ('.join(result) + ')'

def trigger_cut(trigger, CUT=""):
  if trigger == "biased":
    CUT = cuts_and("hlt1b==1", CUT)
  if trigger == "unbiased":
    CUT = cuts_and("hlt1b==0", CUT)
  if trigger == "combined":
    CUT = CUT
  return CUT

