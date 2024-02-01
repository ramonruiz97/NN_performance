# Flavour Combination of OS taggers
__author__ = ["Ramón Ángel Ruiz Fernández"],


import numpy as np
import ipanema
import complot
from uncertainties import unumpy as unp
import argparse
import yaml
import os
import uproot3 as uproot

def cal(id, eta, f1=0, f2=0):
  id /= abs(id)
  result = 0
  omb = f1*eta + 0.5*(1 - f1)
  ombbar = f2*eta + 0.5*(1 - f2)
  om = 0.5 * (omb + ombbar)
  dom = 0.5*(omb - ombbar)
  result += om + id*dom
  return result




def taggerCombination(qs, etas):
    pB = np.ones_like(qs[0])
    pBbar = np.ones_like(qs[0])
    
    #Events w/ omega > 0.5 < 0. do not contribute 
    #Warning Sevda try ! -> Comment following lines for Sevda try
    #Check this
    # for t in range(len(etas)):
    #   condition = (etas[t]>0.5) | (etas[t]<0.)
    #   etas[t] = np.where(condition, 0.5, etas[t])
    #   qs[t] = np.where(condition, 0., qs[t])


    for i in range(len(qs)):
        pB    *= 0.5*(1.+qs[i]) - qs[i]*(1.-etas[i])
        pBbar *= 0.5*(1.-qs[i]) + qs[i]*(1.-etas[i])

    q = np.zeros_like(qs[0])
    eta = 0.5*np.ones_like(qs[0])

    PB = pB/(pB+pBbar)
    PBbar = 1. - PB

    q = np.where(PB > PBbar, -1., q)
    eta = np.where(PB > PBbar, 1.-PB, eta)
    q = np.where(PBbar > PB, 1., q)
    eta = np.where(PBbar > PB, 1.-PBbar, eta)
    return q, eta




if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Tagging Calibration for B+ and MC')
  parser.add_argument('--data', 
                      help='Input data.')

  parser.add_argument('--weight', 
                      help='Weight to apply to data')

  parser.add_argument('--mode', 
                      help='Mode')

  parser.add_argument('--version', 
                      help='Version')

  parser.add_argument('--model', default='linear')

  parser.add_argument('--os_calibration', default=False)
  parser.add_argument('--ss_calibration', default=False)

  parser.add_argument('--output-sample', 
						help='root file with Combined omega')

  args = vars(parser.parse_args())

  tp = args["data"]
  data = ipanema.Sample.from_root(tp)
  # tagger = args["tagger"]
  mode = args["mode"]
  order = args["model"]
  version = args["version"]
  final_comb = False

  if "final" in args["output_sample"]:
    final_comb = True

  
  
  os_calibrations = False
  ss_calibrations = False

  if args['os_calibration']:
    os_calibrations = args["os_calibration"].split(",")
  if args['ss_calibration']:
    ss_calibrations = args["ss_calibration"].split(",")

  
  with open('config/tagger.yml') as config:
    config = yaml.load(config, Loader=yaml.FullLoader)

  qos = config["list_OSComb"][mode]["decision"]
  etaos = config["list_OSComb"][mode]["eta"]
  qss = config["list_SSComb"][mode]["decision"]
  etass = config["list_SSComb"][mode]["eta"]
 
  if final_comb:
    qos = ["OSCombination_TAGDEC"]
    etaos = ["OSCombination_TAGETA"]
    qss = ["SSCombination_TAGDEC"]
    etass = ["SSCombination_TAGETA"]


  idb = "B_ID"
  if "MC" in mode:
    idb = "B_ID_GenLvl"
  
  


  qos_list = [np.float64(data.df[b]) for b in qos]
  etaos_list = [np.float64(data.df[b]) for b in etaos]
  qss_list = [np.float64(data.df[b]) for b in qss]
  etass_list = [np.float64(data.df[b]) for b in etass]
  if final_comb:
    omega = []
    # names = ["p0", "p1", "p2", "dp0", "dp1", "dp2", "eta_bar"]
    names = ["f1", "f2"]
    calibrations = os_calibrations + ss_calibrations
    etas = etaos + etass
    for i, eta_b in enumerate(etas):
      p = ipanema.Parameters.load(calibrations[i])
      p = ipanema.Parameters.build(p, names)
      omega.append(cal(data.df[f"{idb}"], data.df[f"{eta_b}"], **p.valuesdict()))
    
    qs = qos_list + qss_list
    q, eta = taggerCombination(qs, omega)
    data.df["Combination_TAGDEC"] = q
    data.df["Combination_TAGETA"] = eta

  else:
    if os_calibrations:
      omega = []
      for i, eta_b in enumerate(etaos):
        p = ipanema.Parameters.load(os_calibrations[i])
        omega.append(cal(data.df[f"{idb}"], data.df[f"{eta_b}"], **p.valuesdict()))

      qos, etaos = taggerCombination(qos_list, omega)

    else:
      qos, etaos = taggerCombination(qos_list, etaos_list)

      if ss_calibrations:
        omega = []
        for i, eta_b in enumerate(etass):
          p = ipanema.Parameters.load(os_calibrations[i])
          omega.append(cal(data.df[f"{idb}"], data.df[f"{eta_b}"], **p.valuesdict()))
        qss, etass = taggerCombination(qss_list, omega)

      else:
        qss, etass = taggerCombination(qss_list, etass_list)

    data.df["OSCombination_TAGDEC"] = qos
    data.df["OSCombination_TAGETA"] = etaos
    data.df["SSCombination_TAGDEC"] = qss
    data.df["SSCombination_TAGETA"] = etass

  with uproot.recreate(args['output_sample']) as f:
    _branches = {}
    for k, v in data.df.items():
      if 'int' in v.dtype.name:
        _v = np.int32
      elif 'bool' in v.dtype.name:
        _v = np.int32
      else:
        _v = np.float64
      _branches[k] = _v
    mylist = list(dict.fromkeys(_branches.values()))
    f["DecayTree"] = uproot.newtree(_branches)
    f["DecayTree"].extend(data.df.to_dict(orient='list'))
