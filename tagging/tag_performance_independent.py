# tag_power
# This script sucks
# TODO: Improve
# PLEASE REWRITE ME!!

__all__ = []
__author__ = ["Ramón Ángel Ruiz Fernández"]
__email__ = ["rruizfer@CERN.CH"]



import numpy as np
import ipanema
import complot
import uncertainties as unc
import argparse
import yaml
import os
import json

#Function by tagger and year only 
#By default the json in gitlab has all the years so one must identify the correct year
def get_pars(path, year):
  y = year if year != '201516' else '2016'
  data = json.load(open(str(path)))

  pars = ipanema.Parameters()
  pars_names = ["p0", "p1", "dp0", "dp1"]
  index = {
		        "2016" : 0,
		        "2017" : 1,
		        "2018" : 2	}
	
  for i, n in enumerate(pars_names):
    pars.add(dict(name = f"{n}",
                value = data['ResultSet'][index[y]]['Parameter'][i]["Value"],
                stdev = np.sqrt(data['ResultSet'][index[y]]['Parameter'][i]["Error"]**2  +
                            + data['ResultSet'][index[y]]['SystematicErrors'][0]['Values'][i]**2 +
                            + data['ResultSet'][index[y]]['SystematicErrors'][1]['Values'][i]**2 +
                            + data['ResultSet'][index[y]]['SystematicErrors'][2]['Values'][i]**2)
                ))

  for l, k in enumerate(pars_names):
    pars[k].correl = {}
    for m, k2 in enumerate(pars_names):
      pars[k].correl[k2] = data['ResultSet'][index[y]]['StatisticalCorrelationMatrix'][l][m]

  pars.add(dict(name="eta_bar",
				value=data['Parameter'][0]["Value"],
                stdev = 0.))	


	  
  return pars

def get_pars_epm(path, year):
  line = []
  with open(path) as f:
    line = [line.split('=')[1].replace('\n', '') for line in f.readlines()]
  # data = np.loadtxt(open(str(path)))
  pars = ipanema.Parameters()
  pars.add(dict(name= 'p0',
				value = float(line[1]) + float(line[2]),
				stdev = float(line[4])
					      ))
  pars.add(dict(name= 'dp0',
				value = 0.,
				stdev = 0.
				))

  pars.add(dict(name= 'dp1',
			    value = 0.,
    	        stdev = 0.
				))
  pars.add(dict(name= 'eta_bar',
				value = float(line[1]),
				stdev = 0.
				))

  pars.add(dict(name= 'p1', 
				value = float(line[3]),
				stdev = float(line[5])
				))

  pars['p0'].correl = {}
  pars['p0'].correl['p1'] = float(line[6])
  pars['p1'].correl = {}
  pars['p1'].correl['p0'] = float(line[6])
  return pars

def calibration(eta, dec, p):
  p = p.valuesdict()
  omega = (eta-p['eta_bar'])*(p['p1']+0.5*dec*p['dp1']) + (p['p0']+0.5*dec*p['dp0'])
  return omega
#
#
def tag_power(om, dec, w, norm):
  A = (1 + dec*(1.-2.*om))
  B = (1 - dec*(1.-2.*om))
  return np.sum(w*((A-B)**2)/(A+B)**2)/norm

def tag_eff(N, Nt, Neff):
   eff = Nt/N
   unt = N - Nt
   return 100.*eff, 100.*np.sqrt((Nt*unt)/Neff)/N

def tag_power_comb(om_ss, dec_ss, #always ss first :)
				   om_os, dec_os,
				   w, norm):

  A = (1 + dec_os*(1.-2.*om_os))*(1 + dec_ss*(1.-2.*om_ss)) 
  B = (1 - dec_os*(1.-2.*om_os))*(1 - dec_ss*(1.-2.*om_ss)) 
  return np.sum(w*((A-B)**2)/(A+B)**2)/norm

def tag_power_err(eta, om,
				q, w, norm,
				pars):
  qD = q*q*(1. - 2. * om)
  deriv = np.array([qD, qD*(eta - pars["eta_bar"].value), 
					qD*0.5*np.sign(q), 
					qD*0.5*(eta-pars["eta_bar"].value)*np.sign(q)])

  pars_err = ipanema.Parameters.build(pars, ["p0", "p1", "dp0", "dp1"])
  cov = pars_err.cov()
  err = 0

  for i in range(len(pars_err.cov())):
    for j in range(len(pars_err.cov())):
	    err += np.sum(w*deriv[i])*np.sum(w*deriv[j]) * cov[i][j]
  err *= 16./norm**2
  return np.sqrt(err)

def Dil_err(om, q, w, sum_tag):
  D = 1. -2*om
  mean_Dsq = np.sum(D**2*w)
  mean_D4 = np.sum(D**4*w)
  err = np.sqrt((mean_D4 - mean_Dsq**2) / sum_tag-1)
  return err


def comb_err(eta_ss, eta_os,
			 om_ss, om_os,
	         dec_ss, dec_os,
	         w, norm,
			 pars_ss, pars_os):
  #SS -> 0
  #OS -> 1
  D_ss = dec_ss * (1. - 2. *om_ss)
  D_os = dec_os * (1. - 2. *om_os)
  dw0 = (dec_ss * (D_ss + D_os)*(1.-D_os**2))/(1.+D_ss*D_os)**3
  dw1 = (dec_os * (D_ss + D_os)*(1.-D_ss**2))/(1.+D_ss*D_os)**3

  deriv_0 = np.array([dw0, dw0*(eta_ss-pars_ss['eta_bar'].value), 
										 dw0 * 0.5 * np.sign(dec_ss),
	                   dw0 * 0.5 * (eta_ss - pars_ss['eta_bar'].value)*np.sign(dec_ss)])

  deriv_1 = np.array([dw1, dw1*(eta_os-pars_os['eta_bar'].value), 
										dw1 * 0.5 * np.sign(dec_os),
	                   dw1 * 0.5 * (eta_os - pars_os['eta_bar'].value)*np.sign(dec_os)])

  pars_err = ipanema.Parameters.build(pars_ss, ["p0", "p1", "dp0", "dp1"])
  cov = pars_err.cov()
  err = 0

  for i in range(len(pars_err.cov())):
    for j in range(len(pars_err.cov())):
      err += np.sum(w*deriv_0[i])*np.sum(w*deriv_0[j]) * cov[i][j]

  pars_err = ipanema.Parameters.build(pars_os, ["p0", "p1", "dp0", "dp1"])
  cov = pars_err.cov()

  for i in range(len(pars_err.cov())):
    for j in range(len(pars_err.cov())):
      err += np.sum(w*deriv_1[i])*np.sum(w*deriv_1[j]) * cov[i][j]

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

  taggers = ["OSCombination", "SSCombination", "IFT"]
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
  data = ipanema.Sample.from_root(dp, branches=branches)

  
  # Load Calibration parameters
  dict_pars = {}
  cals = args["calibrations"].split(",")
  for t, c in zip(taggers, cals):
     dict_pars[t] = ipanema.Parameters.load(c)


  
  	
  exclude = {'OSCombination' : 'SSCombination', 'SSCombination': 'OSCombination'}
  print("==============================================")
  print(f"Tagging power")
  print("==============================================")
  norm = data.df[weight].sum()
  TP = {}
  Tp_to_table = {}
  Teff_to_table = {}
  Dil_to_table = {}
  stat = {}

  

  omega, eta, errs, q = [], [], [], []
  
  

  #OS and SS exclusive
  Nt, Neff  = {}, {}
  tageff, etageff = {}, {}
  dil2, edil2 = {}, {}
  for tagger in ["OSCombination", "SSCombination"]:
    data_excl = data.df.query(f"{eta_dict[tagger]} > 0. & {eta_dict[tagger]} < 0.5 & {eta_dict[exclude[tagger]]} == 0.5")
    data_incl = data.df.query(f"{eta_dict[tagger]} > 0. & {eta_dict[tagger]} < 0.5 & {eta_dict[exclude[tagger]]} > 0.0 & {eta_dict[exclude[tagger]]} < 0.5")

    N = data.df.shape[0]
    Nt[tagger] = data_excl.shape[0]
    Neff[tagger] = (data_excl.eval(f"{weight}").sum())**2/data_excl.eval(f"{weight}*{weight}").sum()
    tageff[tagger], etageff[tagger] = tag_eff(N, Nt[tagger], Neff[tagger])
    
    #TODO Improve this we are always calculating this thing twice
    Nt["OSandSS"] = data_incl.shape[0]
    Neff["OSandSS"] = (data_incl.eval(f"{weight}").sum())**2/data_excl.eval(f"{weight}*{weight}").sum()
    tageff["OSandSS"], etageff["OSandSS"] = tag_eff(N, Nt["OSandSS"], Neff["OSandSS"])

     

      
    omegaExcl = calibration(data_excl[eta_dict[tagger]],
                            data_excl[q_dict[tagger]],
                            dict_pars[tagger])

    omegaIncl = calibration(data_incl[eta_dict[tagger]],
                              data_incl[q_dict[tagger]],
                              dict_pars[tagger])

    omega.append(omegaIncl.array)
    eta.append(data_incl[eta_dict[tagger]].array)
    q.append(data_incl[q_dict[tagger]].array)

    TP[tagger] = np.round(100*tag_power(
						  omegaExcl.array,
                          data_excl[q_dict[tagger]].array,
						  data_excl[weight].array,
			              norm), 5)

    err = np.round(100*tag_power_err(data_excl[eta_dict[tagger]].array,
					omegaExcl.array,
				    data_excl[q_dict[tagger]].array,
					data_excl[weight].array,
				    norm,
			        dict_pars[tagger]
				    ) , 5)

    Tp_to_table[tagger] = unc.ufloat(TP[tagger], err)
    Teff_to_table[tagger] = unc.ufloat(tageff[tagger], etageff[tagger])
    Dil_to_table[tagger] = Tp_to_table[tagger]/Teff_to_table[tagger]

    errs.append(err)
    #Dilution**2
    dil2[tagger] = TP[tagger]/tageff[tagger]/100
    edil2[tagger] = dil2[tagger]*np.sqrt((err/TP[tagger])**2 + (etageff[tagger]/tageff[tagger])**2)
			
    print(f"{tagger} {TP[tagger]} +- {err}")

  #(OS + SS ) exclusive
    
  TP["comb"] = np.round(100*tag_power_comb(omega[0], q[0],
                                                  omega[1], q[1],
                                                  data_incl[weight].array, norm)
												, 5)

  err_comb = np.round(100*comb_err(eta[1], eta[0],
											               omega[1], omega[0],
											               q[1], q[0],
											              data_incl[weight].array, norm,
											              dict_pars["SSCombination"], dict_pars["OSCombination"])
												, 5) 
  
  Tp_to_table["OSandSS"] = unc.ufloat(TP["comb"], err_comb)
  Teff_to_table["OSandSS"] = unc.ufloat(tageff["OSandSS"], etageff["OSandSS"])
  Dil_to_table["OSandSS"] = Tp_to_table["OSandSS"]/Teff_to_table["OSandSS"]

  dil2["OSandSS"] = TP["comb"]/tageff["OSandSS"]/100
  edil2["OSandSS"] = dil2["OSandSS"]*np.sqrt((err_comb/TP["comb"])**2 + (etageff["OSandSS"]/tageff["OSandSS"])**2)

  print(f"OSandSSK {TP['comb']} +- {err_comb}")

  #Combination of regulars

  TP_comb = np.round(TP[f"OSCombination"] + TP[f"SSCombination"] + TP[f"comb"] ,3)
  total_err = np.round(np.sqrt(errs[0]**2 + errs[1]**2 + err_comb**2), 3)
  

  tag_eff_final = np.round(tageff["OSCombination"] + tageff["SSCombination"] + tageff["OSandSS"], 3)
  etag_eff_final = np.round(np.sqrt(etageff["OSCombination"]**2 + etageff["SSCombination"]**2 + etageff["OSandSS"]**2), 3)

  Tp_to_table["Total"] = unc.ufloat(TP_comb, total_err)
  Teff_to_table["Total"] = unc.ufloat(tag_eff_final, etag_eff_final)
  Dil_to_table["Total"] = Tp_to_table["Total"]/Teff_to_table["Total"]



  dil2_final = TP_comb/tag_eff_final/100
  edil2_final = dil2_final*np.sqrt((total_err/TP_comb)**2 + (etag_eff_final/tag_eff_final)**2)

  print("-------------------------------------------------------")
  print(f"Combination {TP_comb} +- {total_err}")
  
  #IFT
  #TODO-Change the name in the tuple
  data_excl = data.df.query(f"`{eta_dict['IFT']}` > 0. & `{eta_dict['IFT']}` < 0.5")
  omegaIFT = calibration(data_excl[eta_dict["IFT"]],
                        data_excl[q_dict["IFT"]],
                        dict_pars["IFT"])

  Nt["IFT"] = data_excl.shape[0]
  Neff["IFT"] = (data_excl.eval(f"{weight}").sum())**2/data_excl.eval(f"{weight}*{weight}").sum()
  tageff["IFT"], etageff["IFT"] = tag_eff(N, Nt["IFT"], Neff["IFT"])

  TP["IFT"] = np.round(100*tag_power(
							omegaIFT.array,
							data_excl[q_dict["IFT"]].array,
		                    data_excl[weight].array,
			                norm), 5)

  err = np.round(100*tag_power_err(data_excl[eta_dict["IFT"]].array,
				omegaIFT.array,
				data_excl[q_dict["IFT"]].array,
				data_excl[weight].array,
				norm,
			    dict_pars["IFT"]
		        ), 5)

  dil2["IFT"] = TP["IFT"]/tageff["IFT"]/100
  edil2["IFT"] = dil2["IFT"]*np.sqrt((err/TP["IFT"])**2 + (etageff["IFT"]/tageff["IFT"])**2)

  Tp_to_table["IFT"] = unc.ufloat(TP["IFT"], err)
  Teff_to_table["IFT"] = unc.ufloat(tageff["IFT"], etageff["IFT"])
  Dil_to_table["IFT"] = Tp_to_table["IFT"]/Teff_to_table["IFT"]

  print(f"IFT {TP['IFT']} +- {err}")
  
  #OSCombination 
  data_excl = data.df.query(f"`{eta_dict['OSCombination']}` > 0. & `{eta_dict['OSCombination']}` < 0.5")
  omegaOSC = calibration(data_excl[eta_dict["OSCombination"]],
                        data_excl[q_dict["OSCombination"]],
                        dict_pars["OSCombination"])

  TP["OSCombination"] = np.round(100*tag_power(
							omegaOSC.array,
							data_excl[q_dict["OSCombination"]].array,
		                    data_excl[weight].array,
			                norm), 5)

  err = np.round(100*tag_power_err(data_excl[eta_dict["OSCombination"]].array,
				omegaOSC.array,
				data_excl[q_dict["OSCombination"]].array,
				data_excl[weight].array,
				norm,
			    dict_pars["OSCombination"]
		        ), 5)

  print(f"OSComb {TP['OSCombination']} +- {err}")
    
  data_excl = data.df.query(f"`{eta_dict['SSCombination']}` > 0. & `{eta_dict['SSCombination']}` < 0.5")
  omegaSSC = calibration(data_excl[eta_dict["SSCombination"]],
                        data_excl[q_dict["SSCombination"]],
                        dict_pars["SSCombination"])
  
  #SSCombination

  TP["SSCombination"] = np.round(100*tag_power(
							omegaSSC.array,
							data_excl[q_dict["SSCombination"]].array,
		                    data_excl[weight].array,
			                norm), 5)

  err = np.round(100*tag_power_err(data_excl[eta_dict["SSCombination"]].array,
				omegaSSC.array,
				data_excl[q_dict["SSCombination"]].array,
				data_excl[weight].array,
				norm,
			    dict_pars["SSCombination"]
		        ), 5)

  print(f"SSComb {TP['SSCombination']} +- {err}")

  #Save also to a json:
  _pars = ipanema.Parameters()
  
  names = {
           "OSCombination" : "OS",
           "SSCombination" : "SS",
           "IFT" : "IFT",
           "OSandSS" : "OS_SS",
           "Total" : "Old"
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
                latex=rf"\epsilon D^2 (%) [{names[k]}]"
                ))
  print(_pars)
  # exit()
  _pars.dump(args["output_params"])
  
  asym = "no asym"
  if "v1r0":
    asym = "with asym"


  
  table = []
  names = {
            "OSCombination" : r"\mathrm{OS_{excl}}",
            "SSCombination" : r"\mathrm{SS_{excl}}",
            "OSandSS" : r"\mathrm{(OS + SS)_{excl}}",
            # "Total" : r"\mathrm{Total}",
            "Total" : r"\mathrm{Old}",
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










	

	







