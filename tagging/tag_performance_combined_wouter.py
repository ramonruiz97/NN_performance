# tag_power
# This script sucks
# TODO: Improve
# PLEASE REWRITE ME!!

__all__ = []
__author__ = ["Ramón Ángel Ruiz Fernández"]
__email__ = ["rruizfer@CERN.CH"]



import numpy as np
import uproot3 as uproot
import ipanema
import complot
import uncertainties as unc
import argparse
import yaml
import os
import json

def calibration(eta, dec, p):
  p = p.valuesdict()
  omb = p["f1"]*eta + 0.5*(1 - p["f1"])
  ombbar = p["f2"]*eta + 0.5*(1 - p["f2"])
  om = 0.5*(omb + ombbar)
  dom = 0.5*(omb - ombbar)
  omega = om + dec*dom 
  return omega


def tag_efficiency(N, Nt, Neff):
   eff = Nt/N
   unt = N - Nt
   return 100.*eff, 100.*np.sqrt((Nt*unt)/Neff)/N


def dn(om, w, n):
  sum = np.sum(w)
  return np.sum(w*(1-2*om)**int(n))/sum


def dn_cal(eta, om, 
			q, w, pars, norm):

  const = w * (1.-2.*om)
  deriv = np.array([
                    np.sum(const*(-0.5 + eta)*(0.5 + 0.5*q)),
                    np.sum(const*(-0.25 + eta*(0.5 - 0.5*q) + 0.25*q))
                   ])
  print(deriv)
  pars_err = ipanema.Parameters.build(pars, ["f1", "f2"])
  cov = pars_err.cov()

  err = np.dot(np.dot(deriv, cov), deriv.T)
  # print(err)
  err = np.sqrt(16*err/norm**2)
  return err


def tag_power(om, dec, w, norm):
  A = (1 + dec*(1.-2.*om))
  B = (1 - dec*(1.-2.*om))
  return np.sum(w*((A-B)**2)/(A+B)**2)/norm

def tag_power_err(eta, om,
				 q, w, norm,
				 pars):
  qD = q*q*(1. - 2. * om)*w
  deriv = np.array([
                    qD*(0.5*(-0.5+eta) + 0.5*(eta - 0.5)*q),
                    qD*(0.5*(-0.5+eta) - 0.5*(eta - 0.5)*q)
                   ])

  pars_err = ipanema.Parameters.build(pars, ["f1", "f2"])
  cov = pars_err.cov()

  err = 0

  for i in range(len(pars_err.cov())):
    for j in range(len(pars_err.cov())):
        err += np.sum(w*deriv[i])*np.sum(w*deriv[j]) * cov[i][j]
  err *= 16./norm**2
  return np.sqrt(err)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Tagging Calibration for B+ and MC')
  parser.add_argument('--calibrations', 
                      help='calibration-SS', required=False)
  parser.add_argument('--data', help="input data")
  parser.add_argument('--mode', help="Mode of the data")
  parser.add_argument('--year', help="Year of the data")
  parser.add_argument('--offset', help="Type of offset")
  parser.add_argument('--output-table', 
						help='Location to save fit parameters')
  parser.add_argument('--output-params', 
						help='Location to save fit parameters')

  args = vars(parser.parse_args())  

  mode = args["mode"]
  dp = args["data"]
  year = args["year"]
  offset = args["offset"]

  taggers = ["Combination", "IFT"]
  if mode == "Bu2JpsiKplus":
    weight = "sigBuSW"
  elif mode == "Bs2DsPi":
    weight = "sigBsSW"
  elif mode == "Bd2JpsiKstar":
    weight = "sigBdSW"



  with open('config/tagger.yml') as config:
	  config = yaml.load(config, Loader=yaml.FullLoader)


  eta_dict, q_dict = {}, {}
  for tagger in taggers:
	  eta_dict[tagger] = config[tagger]['branch']['eta']
	  q_dict[tagger] = config[tagger]['branch']['decision']

  branches = [config[tagger]['branch']['decision'] for tagger in taggers]
  branches += [config[tagger]['branch']['eta'] for tagger in taggers]
  branches += [weight]


  #Load samples
  data = uproot.open(dp)["DecayTree"].pandas.df(branches=branches)

  
  # Load Calibration parameters
  dict_pars = {}
  cals = args["calibrations"].split(",")
  for t, c in zip(taggers, cals):
     dict_pars[t] = ipanema.Parameters.load(c)
  


  
  	
  print("==============================================")
  print(f"Tagging power")
  print("==============================================")

  Tp_to_table = {}
  Teff_to_table = {}
  Dil_to_table = {}
  stat = {}

  

  N = data[weight].sum()
 
  #Combination
  taggers = ["Combination", "IFT"]
  for t in taggers:
    cdata = uproot.open(dp)["DecayTree"].pandas.df(branches=branches)
    #Check impact of also cutting etas >0.5 
    # cut = f"{eta_dict[t]}> 0 & {eta_dict[t]}<0.5 & {q_dict[t]} != 0"
    cut = f"{q_dict[t]} != 0"
    cdata = cdata.query(f"{cut}")

    cdata["omega"] = calibration(cdata[eta_dict[t]],
                                 cdata[q_dict[t]],
                                 dict_pars[t])         

    #Also to check this
    #Overflow events ... -> Check other conventions
    # cdata = cdata.query("omega<0.5")
    om = cdata["omega"]
    nwts = np.sum(cdata[weight])
    dsq = dn(om, cdata[weight].values, 2)
    dsq_stat_err = np.sqrt((dn(om, cdata[weight].values, 4) - (dn(om, cdata[weight].values, 2)**2))/(nwts-1))
    dsq_cal_err = dn_cal(cdata[eta_dict[t]].values,
                    cdata["omega"].values,
                    cdata[q_dict[t]].values,
                    cdata[weight].values,
                    dict_pars[t], nwts) 
    print(dsq_cal_err)
    print(dsq_stat_err)

    edsq = np.sqrt(dsq_cal_err**2 + dsq_stat_err**2) 

    Dil_to_table[t] = unc.ufloat(dsq, edsq)

    Nt = cdata[weight].sum()
    Neff = (data[weight].sum())**2/data.eval(f"{weight}*{weight}").sum()
    tageff, etageff = tag_efficiency(N, Nt, Neff)

    Teff_to_table[t] = unc.ufloat(tageff, etageff)
    tagpower = tageff*dsq
    etagpower_cal = tageff*dsq_cal_err 
    print(etagpower_cal)
    etagpower_stat = np.sqrt(tageff**2*dsq_stat_err**2 + dsq**2*etageff**2)
    etagpower = np.sqrt(etagpower_cal**2 + etagpower_stat**2)
    Tp_to_table[t] = unc.ufloat(tagpower, etagpower)

    print(np.round(100*tag_power(
                            cdata["omega"].values,
                            cdata[q_dict[t]].values,
                            cdata[weight].values,
                            N), 5))
    print(np.round(100*tag_power_err(
                           cdata[eta_dict[t]].values,
                           cdata["omega"].values,
                           cdata[q_dict[t]].values,
                           cdata[weight].values,
                           N,
                           dict_pars[t]), 5))


   
  #OLD CODE:
  #Combination
  # data_excl = data.df.query(f"`{eta_dict['Combination']}` > 0. & `{eta_dict['Combination']}` < 0.5")
  # data_excl = data.df.query(f"{q_dict['Combination']} != 0")
  #
  # omegaCombination = calibration(data_excl[eta_dict["Combination"]],
  #                                data_excl[q_dict["Combination"]],
  #                                 dict_pars["Combination"])
  #
  # Nt["Combination"] = data_excl.shape[0]
  # Neff["Combination"] = (data_excl.eval(f"{weight}").sum())**2/data_excl.eval(f"{weight}*{weight}").sum()
  # tageff["Combination"], etageff["Combination"] = tag_eff(N, Nt["Combination"], Neff["Combination"])
  # 
  # norm_tag = data_excl[f"{weight}"].sum()
  # dil_err_cal = Dil_err(omegaCombination.array, 
  #                   data_excl[q_dict["Combination"]].array,
  #                   data_excl[weight].array,
  #                   norm_tag
  #                   )
  


  # TP["Combination"] = np.round(100*tag_power(
		# 					omegaCombination.array,
		# 					data_excl[q_dict["Combination"]].array,
		#                     data_excl[weight].array,
		# 	                norm), 5)

  # err = np.round(100*tag_power_err(data_excl[eta_dict["Combination"]].array,
		# 		omegaCombination.array,
		# 		data_excl[q_dict["Combination"]].array,
		# 		data_excl[weight].array,
		# 		norm,
		# 	    dict_pars["Combination"]
		#         ), 5)

  # dil2["Combination"] = TP["Combination"]/tageff["Combination"]/100
  # edil2["Combination"] = dil2["Combination"]*np.sqrt((err/TP["Combination"])**2 + (etageff["Combination"]/tageff["Combination"])**2)

  # Teff_to_table["Combination"] = unc.ufloat(tageff["Combination"], etageff["Combination"])

  # cal_error = np.round(np.sqrt(tageff["Combination"]**2 * dil_err_cal**2 + etageff["Combination"]**2 *dil2["Combination"]**2), 4)
  # print(f"Combination {TP['Combination']} +- {cal_error} (stat.) +- {err} (cal.)")
  # Tp_to_table["Combination"] = unc.ufloat(TP["Combination"], np.sqrt(err**2+cal_error**2))
  # Dil_to_table["Combination"] = Tp_to_table["Combination"]/Teff_to_table["Combination"]
  
  #IFT
  #TODO-Change the name in the tuple
  # data_excl = data.df.query(f"`{eta_dict['IFT']}` > 0. & `{eta_dict['IFT']}` < 0.5")
  # data_excl = data.df.query(f"`{q_dict['IFT']}` != 0")
  #
  # omegaIFT = calibration(data_excl[eta_dict["IFT"]],
  #                       data_excl[q_dict["IFT"]],
  #                       dict_pars["IFT"])
  #
  # Nt["IFT"] = data_excl.shape[0]
  # Neff["IFT"] = (data_excl.eval(f"{weight}").sum())**2/data_excl.eval(f"{weight}*{weight}").sum()
  # tageff["IFT"], etageff["IFT"] = tag_eff(N, Nt["IFT"], Neff["IFT"])
  #
  # norm_tag = data_excl[f"{weight}"].sum()
  #
  # dil_err_cal = Dil_err(omegaIFT.array, 
  #                   data_excl[q_dict["IFT"]].array,
  #                   data_excl[weight].array,
  #                   norm_tag
  #                   )
  #
  # TP["IFT"] = np.round(100*tag_power(
		# 					omegaIFT.array,
		# 					data_excl[q_dict["IFT"]].array,
		#                     data_excl[weight].array,
		# 	                norm), 5)
  #
  # err = np.round(100*tag_power_err(data_excl[eta_dict["IFT"]].array,
		# 		omegaIFT.array,
		# 		data_excl[q_dict["IFT"]].array,
		# 		data_excl[weight].array,
		# 		norm,
		# 	    dict_pars["IFT"]
		#         ), 5)
  #
  # dil2["IFT"] = TP["IFT"]/tageff["IFT"]/100
  # edil2["IFT"] = dil2["IFT"]*np.sqrt((err/TP["IFT"])**2 + (etageff["IFT"]/tageff["IFT"])**2)

  # Tp_to_table["IFT"] = unc.ufloat(TP["IFT"], err)
  # Teff_to_table["IFT"] = unc.ufloat(tageff["IFT"], etageff["IFT"])
  # Dil_to_table["IFT"] = Tp_to_table["IFT"]/Teff_to_table["IFT"]

  # print(f"IFT {TP['IFT']} +- {err}")
  # cal_error = np.round(np.sqrt(tageff["IFT"]**2 * dil_err_cal**2 + etageff["IFT"]**2 *dil2["IFT"]**2), 4)
  # print(f"IFT {TP['IFT']} +- {cal_error} (stat.) +- {err} (cal.)")
  # Tp_to_table["IFT"] = unc.ufloat(TP["IFT"], np.sqrt(err**2+cal_error**2))
  # Dil_to_table["IFT"] = Tp_to_table["IFT"]/Teff_to_table["IFT"]
  

  #Save also to a json:
  _pars = ipanema.Parameters()
  
  names = {
           "IFT" : "IFT",
           "Combination" : "Old"
    }

  for k in Tp_to_table.keys():
    _pars.add(dict(name=f"Teff_{names[k]}", 
                value=Teff_to_table[k].n,
                stdev=Teff_to_table[k].s,
                latex=rf"\epsilon (\%) [{names[k]}]"
                ))
    _pars.add(dict(name=f"D_{names[k]}", 
                value=Dil_to_table[k].n,
                stdev=Dil_to_table[k].s,
                latex=rf"D^2 [{names[k]}]"
                ))
    _pars.add(dict(name=f"TP_{names[k]}", 
                value=Tp_to_table[k].n,
                stdev=Tp_to_table[k].s,
                latex=rf"\epsilon D^2 (\%) [{names[k]}]"
                ))
  print(_pars)
  # exit()
  _pars.dump(args["output_params"])
  
  asym = "no asym"
  if "v1r0":
    asym = "with asym"


  
  table = []
  names = {
            "Combination" : r"\mathrm{Old}",
            "IFT" : r"\mathrm{IFT}",
    }
  table.append(r"\begin{tabular}{|c|ccc|}")
  table.append(r"\toprule")
  col0 = f"{mode[0:2]} {year} {asym} {offset}"
  col1 = "$\epsilon (\%)$"
  col2 = "$D^2$"
  col3 = "\epsilon D^2 (\%)"
  table.append(f"{col0:<40} & {col1:<50} & {col2:<30} & {col3:<30} \\\\")
  table.append(r"\midrule")
  for k in Tp_to_table.keys():
    line = []
    name = f"{names[k]}"
    line.append(f"${name:<40}$")
    teff = f"{Teff_to_table[k]:.2uL}"
    line.append(f"${teff:<50}$")
    dil = f"{Dil_to_table[k]:.2uL}"
    line.append(f"${dil:<30}$")
    tp = f"{Tp_to_table[k]:.2uL}"
    line.append(f"${tp:<30}$")
    table.append("&".join(line)+r"\\")
    if k in ["Total", "OSandSS"]:
        table.append(r"\midrule")
  table.append(r"\bottomrule")
  table.append(r"\end{tabular} \\")
  with open(args['output_table'], "w") as tex_file:
    tex_file.write("\n".join(table))
  tex_file.close()










	

	







