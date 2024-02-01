# Merge sweights
#
__all__ = []
__author__ = ["Ramón Ángel Ruiz Fernandez"]

import uproot3 as uproot
import argparse
from ipanema import Sample
import numpy as np
import os




if __name__ == '__main__':
  p = argparse.ArgumentParser(description="mass fit")
  p.add_argument('--input-sample')
  p.add_argument('--output-sample')
  p.add_argument('--weights')
  p.add_argument('--mode')
  p.add_argument('--version', default=None)
  args = vars(p.parse_args())

  mode = args['mode']
  version = args['version']

  # Load full dataset and creae prxoy to store new sWeights
  sample = Sample.from_root(args['input_sample'], flatten=None)
  _proxy = np.float64(sample.df['time']) * 0.0

  # List all set of sWeights to be merged
  pars = args["weights"].split(",")

  upars = [i for i in pars if "unbiased" in i]
  bpars = [i for i in pars if "biased" in i]
  

  for sw in upars:
    _weight_u = np.load(sw, allow_pickle=True)
    _weight_b = np.load(sw.replace('unbiased', 'biased'), allow_pickle=True)
    _weight_u = _weight_u.item()
    _weight_b = _weight_b.item()
    for w in _weight_u.keys():
        sample.df[f"{w[1:]}SW"] = _weight_u[w] + _weight_b[w]
  with uproot.recreate(args['output_sample']) as f:
    _branches = {}
    for k, v in sample.df.items():
      if 'int' in v.dtype.name:
        _v = np.int32
      elif 'bool' in v.dtype.name:
        _v = np.int32
      elif "DEC" in k:
        _v = np.int32
      else:
        _v = np.float64
      _branches[k] = _v
    mylist = list(dict.fromkeys(_branches.values()))
    f["DecayTree"] = uproot.newtree(_branches)
    f["DecayTree"].extend(sample.df.to_dict(orient='list'))
