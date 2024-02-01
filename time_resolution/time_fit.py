# time_fit
#
#

__all__ = []
__author__ = ["Ramon Angel Ruiz Fernandez"]
__email__ = ["rruizfer@CERN.CH"]

#Modules needed:
import uproot3 as uproot
import os
import argparse
import uncertainties
import ipanema
import numpy as np
import complot
import matplotlib.pyplot as plt
from uncertainties import umath, ufloat


#Initialize backend
ipanema.initialize(os.environ["IPANEMA_BACKEND"], 1)
prog = ipanema.compile(open("kernels/time_resolution.cu").read())

def time_res(prob=False, time=False,
						 fsig=0, fg1=0, fg2=0, fg3=0,
						 mu=0, sigma1=0, sigma2=0, sigma3=0,
						 fexp=0,
	           fbkg1=0, fbkg2=0,
						 tau1=0, tau2=0,
	           tLL=0, tUL=0, norm=1.):

  prog.kernel_time_fit_Bd(prob, time, 
											 #Signal
											 np.float64(fsig), 
											 np.float64(fg1), np.float64(fg2), np.float64(fg3), 
											 np.float64(mu),
											 np.float64(sigma1), np.float64(sigma2), np.float64(sigma3),
											 #Background
											 np.float64(fexp),
											 np.float64(fbkg1), np.float64(fbkg2),
											 np.float64(tau1), np.float64(tau2),
											 np.float64(tLL), np.float64(tUL),
                       global_size=(len(time),))

  return norm * prob

def fcn(params, time, prob, weight, norm=1.):
  p = params.valuesdict()
  fsig = p["fsig"]
	#Gauss
  fg1 = p["f1"]
  fg2 = p["f2"]
  fg3 = p["f3"]

  mu = p["mu"]
  sigma1 = p["sigma1"]
  sigma2 = p["sigma2"]
  sigma3 = p["sigma3"]
  #Bkg
  fexp = p["fexp"]
  fbkg1 = p["fbkg1"]
  fbkg2 = p["fbkg2"]

  tau1 = p["tau1"]
  tau2 = p["tau2"]
	#Time limints
  tLL = p["tLL"]
  tUL = p["tUL"]
  
  time_res(prob=prob, time=time,
					 #Signal
					 fsig = fsig, 
					 fg1 = fg1, fg2=fg2, fg3=fg3,
					 mu = mu, 
					 sigma1 = sigma1, sigma2 = sigma2, sigma3 = sigma3,
					 #Background
					 fexp = fexp,
					 fbkg1 = fbkg1, fbkg2=fbkg2,
					 tau1 = tau1, tau2 = tau2,
	         tLL = tLL, tUL = tUL)

	#To CPU :)

  num = ipanema.ristra.get(prob)
  return -2. * np.log(num)*ipanema.ristra.get(weight)

def time_fit(data=None, tLL=-1000, tUL=1000, 
						 TRIPLE=True, BKG=True, binvar=999, year = 2015, version = "v0r0",
						 weight=False, figs=""):

  if weight:
	  print(f"Time Fit using weight = {weight}")
	
	 
  
   
  time = "time"
  if "Promptno" in mode:
    time = "time - truetime"

		

  sigmat = "sigmat"
  corr = np.sum(data.eval(weight))/np.sum(data.eval(f"{weight}*{weight}"))
  print(f"Corr = {corr}")

  data.eval(f"t={time}", inplace=True)
  data.eval(f"weight = {weight}*{corr}", inplace=True)
  data = data.query(f"t< {tUL} & t> {tLL}")

  data = data.query(f"t< {tUL} & t> {tLL}")
  
  DM = 17.766e-3
  sDM = 0.0051e-3
  sUL = 0.6*1000
  sLL = 0.0005*1000
  tauLL = 1e-3*1000
  tauUL = 30*1000
  muUL = 0.05*1000
  muLL = -0.05*1000

  if not BKG:
    sLL = 0.
    sUL = 1000

  # if mode=="MC_Bs2DsPi_Promptno":
  #   TRIPLE=True

  pars = ipanema.Parameters()
  pars.add(dict(name="DM", value=DM, stdev=sDM, free=False))

  pars.add(dict(name="fsig", value=0.9,
                min=0.4, max=1., free=True, 
                latex=r"f_{\mathrm{sig}}"))

  pars.add(dict(name="fexp", 
								formula = "1-fsig",
                latex=r"f_{\mathrm{sig}}"))

	#Gauss
  pars.add(dict(name="fproxy1", value=0.5,
                min=0., max=1, free=True,
                latex=r"f_{\mathrm{1}}"))

  single=False
  # if ("sigmat9" in binvar or "sigmat8" in binvar) and ("MC_Bs2DsPi_Prompt"==mode):
  # # if not BKG:
  #   pars.add(dict(name="fproxy1", value=1.,
  #               min=0., max=1, free=False,
  #               latex=r"f_{\mathrm{1}}"))
  #   single =True
  # else:
  #   pars.add(dict(name="fproxy1", value=0.5,
  #               min=0., max=1, free=True,
  #               latex=r"f_{\mathrm{1}}"))
  if TRIPLE:
    pars.add(dict(name="fproxy2", value=0.5,
                min=0., max=1, free=True,
                latex=r"f_{\mathrm{2}}"))
  else:
    pars.add(dict(name="fproxy2", value=1.,
                min=0., max=1, free=False,
                latex=r"f_{\mathrm{2}}"))

  pars.add(dict(name="f1", formula="fproxy2*fproxy1",
                latex=r"f_{\mathrm{g1}}"))

  pars.add(dict(name="f2", formula="fproxy2*(1-fproxy1)",
                latex=r"f_{\mathrm{g2}}"))

  pars.add(dict(name="f3", formula="(1-fproxy2)",
                latex=r"f_{\mathrm{g3}}"))


 
  pars.add(dict(name="mu", value=-0.1,
                min=muLL, max=muUL, free=True))

  pars.add(dict(name="sigma1", value=10.,
                min=sLL, max=sUL, free=True))

  # if ("sigmat9" in binvar or "sigmat8" in binvar) and ("MC_Bs2DsPi_Prompt"==mode):
  # # if not BKG:
  # # if ("MC_Bs2DsPi_Prompt"==mode):
  #   pars.add(dict(name="sigma2", value=30.,
  #               min=sLL, max=sUL, free=False))
  # else:
  #   pars.add(dict(name="sigma2", value=30.,
  #               min=sLL, max=sUL, free=True))

  pars.add(dict(name="sigma2", value=40.,
                min=sLL, max=sUL, free=True))

  pars.add(dict(name="sigma3", value=1.,
                min=sLL, max=sUL, free=TRIPLE))
	
	#Background -> WPV + RD
  pars.add(dict(name="fbkg1", value=0.5,
                min=0., max=1., 
								free=BKG, 
                latex=r"f_{\mathrm{exp}}"))

  pars.add(dict(name="fbkg2",
								formula = "1-fbkg1",
                latex=r"f_{\mathrm{exp}}"))

  pars.add(dict(name="tau1", value=(tauUL - tauLL)/2.,
                min=tauLL, max=tauUL, free=BKG)) 

  pars.add(dict(name="tau2", value=(tauUL-tauLL)/2.,
                min=tauLL, max=tauUL, free=BKG)) 

  if not BKG:
	  pars["fsig"].set(value=1., free=False)
	  pars["fexp"].set(value=0., free=False)
  
  # pars.add(
  #     dict(name="part1", formula="f1 * exp(-(1/2.) * (sigma1*sigma1) * (DM*DM))"))
  #
  # pars.add(
  #     dict(name="part2", formula="f2  * exp(-(1/2.) * (sigma2*sigma2) * (DM*DM))"))
  #
  # pars.add(
  #     dict(name="part3", formula="f3  * exp(-(1/2.) * (sigma3*sigma3) * (DM*DM))"))

  # pars.add(dict(name="dilution"))
  # pars.add(dict(name="sigmaeff"))


	#Time limits
  pars.add(dict(name="tLL", value=tLL, free=False))
  pars.add(dict(name="tUL", value=tUL, free=False))
  
  print(f"The following parameters will be fitted \n")
  print(pars)
  # exit()

  timed = ipanema.ristra.allocate(np.float64(data['t'].values))
  sigmatd = ipanema.ristra.allocate(np.float64(data[sigmat].values))
  prob = 0*timed
  weightd = ipanema.ristra.allocate(np.float64(data['weight'].values))

  res = ipanema.optimize(fcn, pars, fcn_args=(timed, prob, weightd),
                         method='minuit', tol=0.3, strategy = 1, verbose=True)

  print("Parameters minimized:\n")
  print(res.params)
	
  pars = res.params

	
  species_to_plot = ["f1", "f2", "f3", "fbkg1", "fbkg2"]
  if not TRIPLE:
    species_to_plot.remove("f3")
  if single:
    species_to_plot.remove("f2")
  if not BKG:
    species_to_plot.remove("fbkg1")
    species_to_plot.remove("fbkg2")
	
  _label = {
		          "f1" : "Gaussian 1",
		          "f2" : "Gaussian 2",
		          "f3" : "Gaussian 3",
		          "fbkg1" : "WPV",
		          "fbkg2" : "Real Decays",
	}

  comp_pars = ["fsig", "fexp", "mu", "sigma1", "sigma2", "sigma3", "tau1", "tau2", "tLL", "tUL"]

  complementary = {"f1": ["fexp", "f2", "f3"],
	                 "f2": ["fexp", "f1", "f3"],
	                 "f3": ["fexp", "f1", "f2"],
		               "fbkg1": ["fsig", "fbkg2"],
		               "fbkg2": ["fsig", "fbkg1"]}

  _pars = ipanema.Parameters.clone(res.params)

  for k, v in _pars.items():
    v.min = -np.inf
    v.max = +np.inf
    v.set(value=res.params[k].value, min=-np.inf, max=np.inf)

	
	#Calculate the Dilution and the sigma_eff
  # wexp = uncertainties.wrap(np.exp)
  # wlog = uncertainties.wrap(np.log)
  # wsqrt = uncertainties.wrap(np.sqrt)
  DM = ufloat(17.766e-3, 0.0057e-3)
  fproxy1 = ufloat(pars["fproxy1"].value, pars["fproxy1"].stdev)
  fproxy2 = ufloat(pars["fproxy2"].value, pars["fproxy2"].stdev)
  f1 = fproxy1*fproxy2
  f2 = fproxy2*(1 -fproxy1)
  print(f1)
  print(f2)
  # exit()
  # f1 = ufloat(pars["f1"].value, pars["f1"].stdev)
  # f2 = ufloat(pars["f2"].value, pars["f2"].stdev)

  sigma1 = ufloat(pars["sigma1"].value, pars["sigma1"].stdev)
  sigma2 = ufloat(pars["sigma2"].value, pars["sigma2"].stdev)

  D1 = f1 * umath.exp(-sigma1**2*DM**2/2)
  D2 = f2 * umath.exp(-sigma2**2*DM**2/2)
  D3 = 0.

  if TRIPLE:
    f3 = ufloat(pars["f3"].value, pars["f3"].stdev)
    # f3 = (1-fproxy2)
    sigma3 = ufloat(pars["sigma3"].value, pars["sigma3"].stdev)
    D3 = f3 * umath.exp(-sigma3**2*DM**2/2)

  D = D1 + D2  + D3
  sigmaeff = umath.sqrt(-(2/DM**2)*umath.log(D))

  # f1 = pars['f1'].uvalue
  # f2 = pars['f2'].uvalue
  # f3 = pars['f3'].uvalue

 #  f1 = pars["fproxy2"].uvalue*pars["fproxy1"].uvalue
 #  f2 = pars["fproxy2"].uvalue*(1. - pars["fproxy1"].uvalue)
 #  f3 = (1. - pars["fproxy2"].uvalue)*(1. - pars["fproxy1"].uvalue)
	#
	#
 #  sigma1 = pars['sigma1'].uvalue
 #  sigma2 = pars['sigma2'].uvalue
 #  sigma3 = pars['sigma3'].uvalue
	#
 #  DM = uncertainties.ufloat(17.766e-3, 0.0051e-3)
	#
 #  exp1 = wexp(-(1 / 2.) * (sigma1 * DM)**2)
 #  exp2 = wexp(-(1 / 2.) * (sigma2 * DM)**2)
 #  exp3 = wexp(-(1 / 2.) * (sigma3 * DM)**2)
	#
 #  part1 =  f1 * exp1
 #  part2 =  f2 * exp2
 #  part3 =  f3 * exp3
	#
	# 
 #  dilution = part1 + part2 + part3
 #  sigmaeff = wsqrt(-2 * wlog(part1 + part2 + part3)) / DM

  # pars['part1'].set(value=part1.n, stdev=part1.s)
  # pars['part2'].set(value=part2.n, stdev=part2.s)
  # pars['part3'].set(value=part3.n, stdev=part3.s)

  pars.add(dict(name="dilution", value=D.n, stdev=D.s))
  pars.add(dict(name="sigmaeff", value=sigmaeff.n, stdev=sigmaeff.s))

  sigma_ave = np.sum(ipanema.ristra.get(weightd*sigmatd))/(np.sum(ipanema.ristra.get(weightd)))
  pars.add(dict(name='sigmaAverage', value=sigma_ave, stdev=0))
  nevts = len(timed)
  nevts = uncertainties.ufloat(nevts, np.sqrt(nevts))
  pars.add(dict(name='nevts', value=nevts.n, stdev=nevts.s))

  print(f"D:          {D:.2uL}")
  print(f"Sigma:             {sigmaeff:.2uL}")
  print(f"Average of sigmat: {sigma_ave}")
  print(f"Number of events:  {nevts:.2uL}")
  print("New set of parameters")

  if figs:
    ranges = {
      # 'sigmat0':   [0. , 0.026]*1000,
      # 'sigmat1':   [0.026 , 0.032]*1000,
      # 'sigmat2':   [0.032,  0.037]*1000,
      # 'sigmat3':   [36.92 , 0.042]*1000,
      # 'sigmat4':   [41.57 , 0.046]*1000,
      # 'sigmat5':   [45.95 , 0.050]*1000,
      # 'sigmat6':   [0.050 , 0.055]*1000,
      # 'sigmat7':   [0.055 , 0.061]*1000,
      # 'sigmat8':   [0.061 , 0.069]*1000,
      # 'sigmat9':   [0.069 , 0.10]*1000
      'sigmat0':   [0. , 25.7],
      'sigmat1':   [25.7, 31.4],
      'sigmat2':   [31.4, 36,4],
      'sigmat3':   [36.4, 41.0],
      'sigmat4':   [41.0, 45.4],
      'sigmat5':   [45.4, 49.8],
      'sigmat6':   [49.8, 54.5],
      'sigmat7':   [54.5, 63.6],
      'sigmat8':   [63.6, 71.5],
      'sigmat9':  [71.5, 100.0] 
    }

    timeh = ipanema.ristra.get(timed)
    weighth = ipanema.ristra.get(weightd)
    probh = 0 * timeh


    fig, axplot, axpull = complot.axes_plotpull()
    bins = 500
    if ("sigmat" in binvar) and "MC" in mode:
      bins = 150

    hdata = complot.hist(timeh, bins=bins, weights=weighth, density=False)

    axplot.errorbar(hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr,
                  fmt=".k")
    
    TLL = tLL
    TUL = tUL
    if "MC" in mode:
      TLL = -300
      TUL = 300
    else:
      TLL = -1000
      TUL = 1000

    proxy_time = ipanema.ristra.linspace(TLL, TUL, 5000)
    proxy_prob = 0 * proxy_time


    def pdf(params, time, prob, norm=1.):
      # Function already has the params w/ weights taken into account:
      weight = time/time #Only not to give an error
      lkhd = fcn(params, time, prob, weight, norm=1)
      return norm * np.exp(-lkhd / 2)


    for c, i in enumerate(species_to_plot):
      _p = ipanema.Parameters.clone(_pars)
      for k in complementary[i]:
        _p[k].set(value=0)

      
      _prob = pdf(_p, proxy_time, proxy_prob, norm=hdata.norm)
      if _p[i].value>1.0e-9:
        print(f"Plotting {i}")
        axplot.plot(ipanema.ristra.get(proxy_time), (_p["fsig"].value + _p["fexp"].value)*ipanema.ristra.get(_prob),
                  color=f"C{c+1}", linestyle='--', label=_label[i])

    _p = ipanema.Parameters.clone(res.params)

    _prob = pdf(_p, proxy_time, proxy_prob, norm=hdata.norm)
    axplot.plot(ipanema.ristra.get(proxy_time), _prob, color="C0",
              label=rf"Full fit")


    pulls = complot.compute_pdfpulls(ipanema.ristra.get(proxy_time), ipanema.ristra.get(_prob),
                                   hdata.bins, hdata.counts, *hdata.yerr)

    axpull.fill_between(hdata.bins, pulls, 0, facecolor="C0", alpha=0.5)
    
    if mode =="MC_Bs2DsPi_Promptno":
      axplot.set_title(rf"MC $B_s \rightarrow D_s^- \pi^+$", fontsize=15)
      axpull.set_xlabel("$\Delta t = t_{reco} - t_{true}$  [fs]", fontsize=15)
    else:
      axplot.set_title(rf"Prompt $D_s(\rightarrow \phi \pi)$", fontsize=15)
      axpull.set_xlabel("t [fs]", fontsize=15)

    axplot.set_ylabel(rf"Candidates / ({np.round(hdata.xerr[1][0],1)} fs)", fontsize=15)
    axplot.text(0.15, 0.8, f"LHCb {year}", fontsize=13,
		            ha='left', va = 'center', transform = axplot.transAxes, alpha=1.)
    # if "sigmat" in binvar:
    axplot.text(0.10, 0.7, rf"$\sigma_t \in {ranges[f'{binvar}']}$ fs", fontsize=13,
		            ha='left', va = 'center', transform = axplot.transAxes, alpha=1.)


    axplot.legend(fontsize=11, loc="upper right")
    os.makedirs(figs, exist_ok=True)
    axplot.set_ylim(0., 1.5*np.max(_prob))
    axpull.set_xlim(TLL, TUL)
    fig.savefig(os.path.join(figs, f"{year}_{binvar}_fit.pdf"))
    axplot.set_yscale("log")
    try:
      axplot.set_ylim(1e0, 10 * np.max(_prob))
    except:
      print("axes not scaled")

    fig.savefig(os.path.join(figs, f"{year}_{binvar}_logfit.pdf"))
    plt.close()

  return pars

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
        description='Calibration of the decay time resolution.')

  parser.add_argument('--data', help='Input prompt data.')
  parser.add_argument('--tLL', default=-1000, required=False)
  parser.add_argument('--tUL', default=1000, required=False)
  parser.add_argument('--weight', help='Weight applied to data')
  parser.add_argument('--year', help='Weight applied to data')
  parser.add_argument('--mode', help='Weight applied to data')
  parser.add_argument('--version', help='Version of tuple v0r0: Jordy, v1r0: JpsiPhi')
	
  parser.add_argument('--bin', help='Bin of the decay-time-error')

  parser.add_argument('--output-json', help='Location to save fit parameters')
  parser.add_argument('--output-plots', help='Location to create plots')

  args = vars(parser.parse_args())
   
  data = args["data"]
  branches = ["time", "sigmat"]
  weight = args["weight"]
  year = args["year"]
  mode = args["mode"]
  tLL = float(args["tLL"])
  tUL = float(args["tUL"])
  if "MC" not in mode:
    branches += [weight]

  TRIPLE = False
  BKG = True

  if "Promptno" in mode:
    branches += ["truetime"]

  if "MC" in mode and "full" in data:
    TRIPLE = True

  
  version = args["version"]
  bin = args["bin"][-1]
  binvar = args["bin"]

  df = uproot.open(data)["DecayTree"].pandas.df(branches=branches)
  
  if "MC" in mode:
    BKG = False

  if "Promptno" in mode:
    TRIPLE = True
    # tLL = -1000
    # tUL = 1000
  TRIPLE = False
  pars = time_fit(data=df, 
									tLL=tLL, 
									tUL=tUL, 
									TRIPLE = TRIPLE,
									BKG = BKG,
									binvar=binvar, 
									year = year,
									weight=weight, 
									version = version,
									figs = args["output_plots"])
  print(pars)
  pars.dump(args["output_json"])


	







# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
