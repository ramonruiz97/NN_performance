# merge_years_performance
#
#

__all__ = []
__author__ = ["Mierda"]
__email__ = ["email"]


#Script for testing
import uproot3 as uproot
import ipanema
import yaml
import numpy as np
import pandas as pd
import argparse
import uncertainties as unc
from iminuit import Minuit
from scipy.stats import chi2



def calibration(eta, dec, p):
  p = p.valuesdict()
  omega = (eta-p['eta_bar'])*(p['p1']+0.5*dec*p['dp1'])
  omega += (p['p0']+0.5*dec*p['dp0'])
  return omega

def tag_efficiency(N, Nt, Neff):
   eff = Nt/N
   unt = N - Nt
   return 100.*eff, 100.*np.sqrt((Nt*unt)/Neff)/N


def dn(om, w, n):
  sum = np.sum(w)
  return np.sum(w*(1-2*om)**int(n))/sum


def dn_cal(eta, om, 
					 q, w, pars):
  
  const = w * (1-2*om)
  deriv = np.array([
										np.sum(const),
										np.sum(q/2*const), 
										np.sum((eta - pars["eta_bar"].value)*const),
										np.sum(const*q/2*(eta-pars["eta_bar"].value))
									])
  # deriv *= -4/np.sum(w)	
  pars_err = ipanema.Parameters.build(pars, ["p0", "dp0", "p1", "dp1"])
  cov = pars_err.cov()

  err = np.dot(np.dot(deriv, cov), deriv.T)
	# err = np.sqrt(err)
  return err
	
#Already checked gives exactly same result:  
# def tag_power(om, dec, w, norm):
#   A = (1 + dec*(1.-2.*om))
#   B = (1 - dec*(1.-2.*om))
#   return np.sum(w*((A-B)**2)/(A+B)**2)/norm




if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Tagging Calibration for B+ and MC')
  parser.add_argument('--calibrations-IFT', 
                      help='calibration-IFT', required=False)
  parser.add_argument('--calibrations-Combination', 
                      help='calibration-IFT', required=False)
  parser.add_argument('--calibrations-OSCombination', 
                      help='calibration-IFT', required=False)
  parser.add_argument('--calibrations-SSCombination', 
                      help='calibration-IFT', required=False)
  parser.add_argument('--data', help="input data")
  parser.add_argument('--mode', help="Mode of the data")
  parser.add_argument('--year', help="Year of the data")
  parser.add_argument('--binvar', help="binvar")
  parser.add_argument('--offset', help="Type of offset")
  parser.add_argument('--output-table', 
						help='Location to save fit parameters')
  parser.add_argument('--output-params', 
						help='Location to save fit parameters')

  args = vars(parser.parse_args())  

  mode = args["mode"]
  with open('config/tagger.yml') as config:
	  config = yaml.load(config, Loader=yaml.FullLoader)

  t_p = args["data"].split(",")
  p = {}
  p_comb = args["calibrations_Combination"].split(",")
  p_OS = args["calibrations_OSCombination"].split(",")
  p_SS = args["calibrations_SSCombination"].split(",")
  p_ift = args["calibrations_IFT"].split(",")
  p["Combination"] = p_comb
  p["OSCombination"] = p_OS
  p["SSCombination"] = p_SS
  p["IFT"] = p_ift

  eta_b = {}
  q_b = {}
  w_b = "sigBsSW"
  if "Bd" in args["mode"]:
    w_b = "sigBdSW"
  if "MC" in args["mode"]:
	  w_b = "B_TRUEID/B_TRUEID"

	
  years = ["2016", "2017", "2018"]
  branches = []
  tree = "DecayTree"
  binvar = args["binvar"]
  nbin = int(len(p_ift)/len(years))

  taggers = ["OSCombination", "SSCombination", "Combination", "IFT"]

  bins = [f"{binvar}{i}" for i in range(0,nbin)]
  pars = {}
  for t in taggers:
    pars[t] = {}
    for i, y in enumerate(years):
      pars[t][y] = {}
      for j, b in enumerate(bins):
	      pars[t][y][b] = ipanema.Parameters.load(p[t][i*len(bins) + j])
  
  for t in taggers:
    q_b[t] = config[t]["branch"]["decision"]
    eta_b[t] = config[t]["branch"]["eta"]
    if (mode == "MC_Bs2JpsiPhi") and (t=="IFT"):
	    q_b["IFT"]="B_IFT_InclusiveTagger_TAGDEC"
	    eta_b["IFT"]="B_IFT_InclusiveTagger_TAGETA"

    branches += [eta_b[t]]
    branches += [q_b[t]]
  if "MC" in mode: 
    branches += ["B_TRUEID"]
  else:
    branches += [w_b]

  
  dilution = {}	
  tag_eff = {}
  tag_power = {}
  for t in taggers:
    dilution[t] = {}
    tag_eff[t] = {}
    tag_power[t] = {}
    for j, b in enumerate(bins):
      data = {}
      cdata = {}
      for i,y in enumerate(years):
        data[y] = uproot.open(t_p[i*len(bins) + j])[tree].pandas.df(branches=branches)
        cut = f"{eta_b[t]}> 0 & {eta_b[t]}<0.5 & {q_b[t]} != 0"
        cdata[y] = uproot.open(t_p[i*len(bins) + j])[tree].pandas.df(branches=branches)
        cdata[y] = cdata[y].query(cut)
        cdata[y].loc[:, "omega"] = calibration(cdata[y][eta_b[t]].values, cdata[y][q_b[t]].values, pars[t][y][b])
        cdata[y] = cdata[y].query("omega<0.5")


      total_tag = pd.concat(cdata[y] for y in years)
      total = pd.concat(data[y] for y in years)
      tot_om = total_tag["omega"]
      norm = total_tag.eval(w_b).sum()

       
      nwts = np.sum(total_tag.eval(w_b))
      dsq =  dn(tot_om, total_tag.eval(w_b).values, 2)
      dsq_stat_err = np.sqrt((dn(tot_om, total_tag.eval(w_b).values, 4) - (dn(tot_om, total_tag.eval(w_b).values, 2)**2))/(nwts-1))

      err = 0
      for y in years:
        err += dn_cal(cdata[y][eta_b[t]].values,
                      cdata[y]["omega"].values,
											cdata[y][q_b[t]].values,
											cdata[y].eval(w_b).values,
											pars[t][y][b])

      dsq_cal_err = np.sqrt(16*err/nwts**2)

      edsq = np.sqrt(dsq_cal_err**2 + dsq_stat_err**2)

      dilution[t][b] = unc.ufloat(dsq, edsq)

      N = total.eval(w_b).sum()
      Nt = total_tag.eval(w_b).sum()
      Neff = (total.eval(w_b).sum())**2/total.eval(f"{w_b}*{w_b}").sum()

      tageff, etageff = tag_efficiency(N, Nt, Neff)

      tag_eff[t][b] = unc.ufloat(tageff, etageff)

      tagpower = tageff*dsq
      etagpower_cal = tageff*dsq_cal_err
      etagpower_stat = np.sqrt(tageff**2*dsq_stat_err**2 + dsq**2*etageff**2)

      etagpower = np.sqrt(etagpower_cal**2 + etagpower_stat**2)

      tag_power[t][b] = unc.ufloat(tagpower, etagpower)
	
  pars = ipanema.Parameters()
  names = {
           "IFT" : "IFT",
           "Combination" : "Old",
           "OSCombination" : "OS",
           "SSCombination" : "SS"
    }
  for k, b in enumerate(bins):
    for t in taggers:
      pars.add(dict(name=f"Teff_{names[t]}_{b}", 
                value=tag_eff[t][b].n,
                stdev=tag_eff[t][b].s,
                latex=rf"\epsilon (\%) [{names[t]}]"
                ))
      pars.add(dict(name=f"D_{names[t]}_{b}", 
                value=dilution[t][b].n,
                stdev=dilution[t][b].s,
                latex=rf"D^2 [{names[t]}]"
                ))
      pars.add(dict(name=f"TP_{names[t]}_{b}", 
                value=tag_power[t][b].n,
                stdev=tag_power[t][b].s,
                latex=rf"\epsilon D^2 (\%) [{names[t]}]"
                ))
  
  print(pars)
  pars.dump(args["output_params"])
      

  table = []
  _calign = 'l|' + 'c'*nbin + '|c|c|c'
  # table.append(r"\top")
  _header = [f"{'Parameter ':<34}"]
  for i in bins:
    b = f"{i}"
    _header += [f"{b :<34}"] 

  _header += [f"{'Error ':<10}"]
  _header += [f"{'Diff ':<10}"]
  _header += [f"{'Pull ':<10}"]
  
	#rows
  _r = ["Teff_OS", "D_OS", "TP_OS"]
  _r += ["Teff_SS", "D_SS", "TP_SS"]
  _r += ["Teff_Old", "D_Old", "TP_Old"]
  _r += ["Teff_IFT", "D_IFT", "TP_IFT"]

  for r in _r:
    _row = [  f"$ {pars[f'{r}_{binvar}0'].latex:<30} $" ]
    c = []; std=[];  _= []
    if "Teff_IFT"==r:
      table.append("\hline")
    for b in bins:
      n = f"{r}_{b}"
      _row.append( f"$ {f'{pars[n].uvalue:.2uL}':<30} $" )
      c.append(pars[n].uvalue.n)
      std.append(pars[n].uvalue.s)
    for i in range(len(c)):
      for j in range(len(c)):
        _.append(c[i] - c[j])
    def compute_chi2(reference_value):
      chi2 = 0.0
      for v, s in zip(c[1:], std[1:]):
        chi2 += ((v - reference_value)/s)**2
      return chi2
    result = Minuit(compute_chi2, reference_value=0.0, errordef=1)
    result.migrad()
    pval = chi2.sf(result.fval, len(c)-1-1)
    diff = np.max(_)
    std = np.min(std)
    _row.append(f"$ {np.float64(std):+.3f}$")
    _row.append(f"$ {np.float64(diff):+.3f} $")
    _row.append(f"$ {pval:+.3f} $")
    table.append(" & ".join(_row))

  # table.append(r"\bottomrule")
  # table.append(r"\end{tabular} \\")
  with open(f'{args["output_table"]}','w') as f:
    f.write(rf"\begin{{tabular}}{{{_calign}}}")
    f.write("\n")
    f.write("\hline")
    f.write("\n")
    f.write(" & ".join(_header))
    f.write(" \\\ ")
    f.write("\n")
    f.write("\hline")
    f.write("\n")
    f.write(" \\\ \n".join(table))
    f.write(" \\\ ")
    f.write("\n")
    f.write("\hline")
    f.write("\n")
    f.write(rf"\end{{tabular}}")
    f.close()



				
#

# # vim: fdm=marker ts=2 sw=2 sts=2 sr noet
