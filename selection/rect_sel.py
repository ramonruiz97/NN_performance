
__author__ = ["Ramón Ángel Ruiz Fernández"]
__email__ = ["rruizfer@cern.ch"]


import config
import uproot3 as uproot
import argparse
import os
import numpy as np
import yaml

#Loading yaml files -> To do merge all
with open("config/selection.yml") as conf:
  config = yaml.load(conf, Loader=yaml.FullLoader)

with open("selection/branches.yml") as branch:
  branches = yaml.load(branch, Loader=yaml.FullLoader)

with open("selection/binvars.yml") as binv:
  binvars = yaml.load(binv, Loader=yaml.FullLoader)

if __name__ == "__main__":
  p = argparse.ArgumentParser(description="Rectangular selection")
  p.add_argument('--input', help='Input tuple')
  p.add_argument('--year', help='Year of the tuple.')
  p.add_argument('--mode', help='Decay mode of the tuple.')
  p.add_argument('--version', help='Decay mode of the tuple.')
  p.add_argument('--variable', help='Var for doing binning')
  p.add_argument('--tree', help='Input file tree name.', default="DecayTree", required=False)
  p.add_argument('--output', help='Output tuple')
  args = vars(p.parse_args())

  mc_truth = [
  "abs(lab2_TRUEID)==431", # Ds
  "&abs(lab1_TRUEID)==211" # pi
  "&abs(lab2_MC_MOTHER_ID)!=531", # no Bs
  "&abs(lab2_MC_MOTHER_ID)!=511", # no B+
  "&abs(lab2_MC_MOTHER_ID)!=521",  # no B0
  "&abs(lab2_MC_MOTHER_ID)!=541", # no Bc
  "&abs(lab2_MC_MOTHER_ID)<1000", # no L's
  "&abs(lab2_MC_GD_MOTHER_ID)!=531", # no Bs
  "&abs(lab2_MC_GD_MOTHER_ID)!=511", # no B+
  "&abs(lab2_MC_GD_MOTHER_ID)!=521", # no B0
  "&abs(lab2_MC_GD_MOTHER_ID)!=541", # no Bc
  "&abs(lab2_MC_GD_MOTHER_ID)<1000", # no L's
  "&abs(lab3_TRUEID)==321", # K
  "&abs(lab4_TRUEID)==321", # K
  "&abs(lab5_TRUEID)==211", # pi
  "&abs(lab3_MC_MOTHER_ID)==431", # Ds
  "&abs(lab4_MC_MOTHER_ID)==431", # Ds
  "&abs(lab5_MC_MOTHER_ID)==431", # Ds
  "&abs(lab3_MC_GD_MOTHER_ID)!=531", # no Bs
  "&abs(lab4_MC_GD_MOTHER_ID)!=531",
  "&abs(lab5_MC_GD_MOTHER_ID)!=531",
  "&abs(lab3_MC_GD_MOTHER_ID)!=511", # no B+
  "&abs(lab4_MC_GD_MOTHER_ID)!=511",
  "&abs(lab5_MC_GD_MOTHER_ID)!=511",
  "&abs(lab3_MC_GD_MOTHER_ID)!=521",
  "&abs(lab4_MC_GD_MOTHER_ID)!=521",
  "&abs(lab5_MC_GD_MOTHER_ID)!=521",
  "&abs(lab3_MC_GD_MOTHER_ID)!=541",
  "&abs(lab4_MC_GD_MOTHER_ID)!=541",
  "&abs(lab5_MC_GD_MOTHER_ID)!=541",
  "&abs(lab3_MC_GD_MOTHER_ID)<1000",
  "&abs(lab4_MC_GD_MOTHER_ID)<1000",
  "&abs(lab5_MC_GD_MOTHER_ID)<1000"]

  tree = args["tree"] 
  mode = args["mode"]
  binvar = args["variable"] 
  version = args["version"]
  b = list(branches[mode].keys())

  df = uproot.open(args["input"])[tree].pandas.df(flatten=False)

  if mode=="MC_Bs2DsPi_Prompt":
    cut = " ".join(truth.replace("&", " & ") for truth in mc_truth)
    df = df.query(cut)
    print(f"Cut applied: {cut}")
    b += ["lab2_TRUEID", "lab2_MC_MOTHER_ID", "lab2_MC_GD_MOTHER_ID", "lab3_TRUEID", "lab4_TRUEID", "lab5_TRUEID", "lab3_MC_MOTHER_ID", "lab4_MC_MOTHER_ID", "lab5_MC_MOTHER_ID", "lab3_MC_GD_MOTHER_ID", "lab4_MC_GD_MOTHER_ID", "lab5_MC_GD_MOTHER_ID"]



  list_of_cuts = [v for v in config[mode].values() if v != '']


  cut = f"({') & ('.join(list_of_cuts)} )"
  print(f"Cut applied: {cut}")

  print(f"Number of events before selecting {df.shape[0]}")
 
  for k, v in branches[mode].items():
    if k not in df.keys():
      if "hlt1b" in k:
        df.eval(f"{k} = {v}", inplace=True)
        #Its a bool to int (0: unbiased, 1: biased)
        df = df.astype({f"{k}" : int})
      else:
        df.eval(f"{k} = {v}", inplace=True)

  cdf = df.query(cut)
  cdf = cdf[b]

  if mode=="Bu2JpsiKplus": #Thanks Simon xd
    cdf.eval("B_SSKaonLatest_TAGDEC = -1*B_SSKaonLatest_TAGDEC", inplace=True)

  if binvar != "full":
    print(binvar)
    list_of_cuts = [v for k, v in binvars[mode].items() if k == binvar ]
    cut = f"({') & ('.join(list_of_cuts)} )"
    cdf = cdf.query(cut)

  print(f"Second cut applied (binvar): {cut}")


  print(f"Number of events after selecting {cdf.shape[0]}")


  with uproot.recreate(args['output']) as f:
    _branches = {}
    for k, v in cdf.items():
      if 'int' in v.dtype.name:
        _v = np.int32
      elif 'bool' in v.dtype.name:
        _v = np.int32
      else:
        _v = np.float64
      _branches[k] = _v
    mylist = list(dict.fromkeys(_branches.values()))
    f[tree] = uproot.newtree(_branches)
    f[tree].extend(cdf.to_dict(orient='list'))

  print('Succesfully written.')

  






