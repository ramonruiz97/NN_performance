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

def time_fit(data=None, 
						 time_branch="time",
						 tLL=-1000, tUL=1000, 
						 TRIPLE=True, RD = True,
						 bin=999, year = 2015,
						 weight=False, figs=""):

  if weight:
	  print(f"Time Fit using weight = {weight}")
	
  data.eval(f"t = {time_branch}", inplace=True) 
  data = data.query(f"t< {tUL} & t> {tLL}")
  # data = data.query(f"sigDsSW>-0.8")
  
  time = "t"
  sigmat = "sigmat"
  corr = np.sum(data.eval(weight))/np.sum(data.eval(f"{weight}*{weight}"))
  data.eval(f"weight = {weight}*{corr}", inplace=True)
	
  DM = 0.5065
  sDM = 0.0019
  # DM = 17.757e-3

  pars = ipanema.Parameters()
  pars.add(dict(name="DM", value=DM, stdev=sDM, free=False))

  pars.add(dict(name="fsig", value=1.,
                min=0.8, max=1., free=True, 
                latex=r"f_{\mathrm{sig}}"))

  pars.add(dict(name="fexp", 
								formula = "1-fsig",
                latex=r"f_{\mathrm{sig}}"))

	#Gauss
  pars.add(dict(name="fproxy1", value=0.3,
                min=0.05, max=1, free=True,
                latex=r"f_{\mathrm{1}}"))

  pars.add(dict(name="fproxy2", value=1.,
                min=0.05, max=1, free=TRIPLE,
                latex=r"f_{\mathrm{2}}"))

  pars.add(dict(name="f1", formula="fproxy2*fproxy1",
                latex=r"f_{\mathrm{g1}}"))

  pars.add(dict(name="f2", formula="fproxy2*(1-fproxy1)",
                latex=r"f_{\mathrm{g2}}"))

  pars.add(dict(name="f3", formula="(1-fproxy2)",
                latex=r"f_{\mathrm{g3}}"))



  pars.add(dict(name="mu", value=-0.1,
                min=tLL, max=tUL, free=True))

  pars.add(dict(name="sigma1", value=0.01,
                min=0.005, max=tUL, free=True))

  pars.add(dict(name="sigma2", value=0.03,
                min=0.005, max=tUL, free=True))

  pars.add(dict(name="sigma3", value=0.1,
                min=0.01, max=tUL, free=TRIPLE))
	
	#Background -> WPV + RD
  pars.add(dict(name="fbkg2", value=0.,
                min=0., max=1., 
								free=RD, 
                latex=r"f_{\mathrm{exp}}"))

  pars.add(dict(name="fbkg1",
								formula = "1-fbkg2",
                latex=r"f_{\mathrm{exp}}"))

  pars.add(dict(name="tau1", value=0.2,
                min=0.01, max=1, free=True)) 

  pars.add(dict(name="tau2", value=0.1,
                min=0.001, max=2., free=RD)) 
 


	#Time limits
  pars.add(dict(name="tLL", value=tLL, free=False))
  pars.add(dict(name="tUL", value=tUL, free=False))
  
  print(f"The following parameters will be fitted \n")
  print(pars)

  timed = ipanema.ristra.allocate(np.float64(data[time].values))
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
  DM = ufloat(DM, sDM)
  print(DM)
  fproxy1 = ufloat(pars["fproxy1"].value, pars["fproxy1"].stdev)
  fproxy2 = ufloat(pars["fproxy2"].value, pars["fproxy2"].stdev)
  f1 = fproxy1*fproxy2
  f2 = fproxy2*(1 -fproxy1)
  print(f1)
  print(f2)

  sigma1 = ufloat(pars["sigma1"].value, pars["sigma1"].stdev)
  sigma2 = ufloat(pars["sigma2"].value, pars["sigma2"].stdev)

  D1 = f1 * umath.exp(-sigma1**2*DM**2/2)
  D2 = f2 * umath.exp(-sigma2**2*DM**2/2)
  D3 = 0.

  if TRIPLE:
    f3 = (1-fproxy2)
    sigma3 = ufloat(pars["sigma3"].value, pars["sigma3"].stdev)
    D3 = f3 * umath.exp(-sigma3**2*DM**2/2)

  D = D1 + D2  + D3
  sigmaeff = umath.sqrt(-(2/DM**2)*umath.log(D))

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
      'sigmat0':   [0.0 , 0.019],
      'sigmat1':   [0.019 , 0.023],
      'sigmat2':   [0.023, 0.026],
      'sigmat3':   [0.026, 0.028],
      'sigmat4':   [0.028 , 0.031],
      'sigmat5':   [0.031 , 0.034],
      'sigmat6':   [0.034 , 0.036],
      'sigmat7':   [0.036 , 0.040],
      'sigmat8':   [0.04 , 0.045],
      'sigmat9':   [0.045 , 0.15],
    }

    timeh = ipanema.ristra.get(timed)
    weighth = ipanema.ristra.get(weightd)
    probh = 0 * timeh


    fig, axplot, axpull = complot.axes_plotpull()
    hdata = complot.hist(timeh, bins=150, weights=weighth, density=False)

    axplot.errorbar(hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr,
                  fmt=".k")

    proxy_time = ipanema.ristra.linspace(min(timeh), max(timeh), 5000)
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
      if _p[i].value>1.0e-9:
        _prob = pdf(_p, proxy_time, proxy_prob, norm=hdata.norm)
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

    # TLL = -0.65
    # TUL = 0.65

    if np.abs(min(timeh))< np.abs(max(timeh)):
      TLL = min(timeh)
      TUL = -TLL

    else:
      TUL = max(timeh)
      TLL = -TUL

		  

    axpull.set_xlabel("$\mathrm{t}_{\mathrm{reco}} - \mathrm{t}_{\mathrm{true}}$ [ps]", fontsize=15)
    axplot.set_ylabel(rf"Candidates / ({np.round(hdata.xerr[1][0], 4)} ps)", fontsize=15)
    axplot.set_title(rf" $B_d \rightarrow J/\psi K^*$", fontsize=15)
    axplot.text(0.15, 0.8, f"LHCb {year}", fontsize=13,
		            ha='left', va = 'center', transform = axplot.transAxes, alpha=1.)
    axplot.text(0.10, 0.7, rf"$\sigma_t \in {ranges[f'sigmat{bin[-1]}']}$ ps", fontsize=13,
		            ha='left', va = 'center', transform = axplot.transAxes, alpha=1.)


    axplot.legend(fontsize=11, loc="right")
    os.makedirs(figs, exist_ok=True)
    axplot.set_ylim(0., 1.5*np.max(_prob))
    axpull.set_xlim(TLL, TUL)
    fig.savefig(os.path.join(figs, f"{year}_{binvar}_fit.pdf"))
    axplot.set_yscale("log")
    try:
      axplot.set_ylim(1e0, 2 * np.max(_prob))
    except:
      print("axes not scaled")

    # fig.savefig(os.path.join(figs, "logfit.pdf"))
    fig.savefig(os.path.join(figs, f"{year}_{binvar}_logfit.pdf"))
    plt.close()

  return pars

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
        description='Calibration of the decay time resolution.')

  parser.add_argument('--data', help='Input prompt data.')
  parser.add_argument('--mode', help='Input prompt data.')
  parser.add_argument('--tLL', default=-1., required=False)
  parser.add_argument('--tUL', default=1., required=False)
  parser.add_argument('--weight', help='Weight applied to data')
  parser.add_argument('--year', help='Weight applied to data')
  parser.add_argument('--version', help='Weight applied to data')
	
  parser.add_argument('--bin', help='Bin of the decay-time-error')

  parser.add_argument('--output-json', help='Location to save fit parameters')
  parser.add_argument('--output-plots', help='Location to create plots')

  args = vars(parser.parse_args())
   
  data = args["data"]
  branches = ["time", "sigmat"]
  weight = args["weight"]
  year = args["year"]
  mode = args["mode"]
  print(weight)
  if "MC" not in mode:
    branches += [weight]
  bin = args["bin"][-1]
  binvar = args["bin"]
  tLL = float(args["tLL"])
  tUL = float(args["tUL"])
  time_branch = "time"
  branches += ["B_BKGCAT"]
  if "Prompt" not in mode:
    branches += ["truetime"]
    time_branch = "time-truetime"

  if "MC" in mode:
    weight = "time/time"
  
  df = uproot.open(data)["DecayTree"].pandas.df(branches=branches)
  pars = time_fit(data=df, 
									time_branch=time_branch,
									tLL=tLL, 
									tUL=tUL, 
									TRIPLE = False,
									RD = False,
									bin=binvar, 
									year = year,
									weight=weight, 
									figs = args["output_plots"])
  print(pars)
  pars.dump(args["output_json"])


	







# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
