# calibration
#
#

__all__ = []
__author__ = ["Ramon Angel Ruiz Fernandez"]
__email__ = ["rruizfer@CERN.CH"]


import os
import uproot3 as uproot
import ipanema
import numpy as np
import complot
import matplotlib.pyplot as plt
import argparse
from uncertainties import unumpy as unp
from ipanema.confidence import get_confidence_bands
from ipanema import uncertainty_wrapper



#Initialize backend
ipanema.initialize(os.environ["IPANEMA_BACKEND"], 1)


def linear_fit(sigmat, sigmaeff, sigmaeff_err, sigmat_err, av):
  def model(x, *args): #Model
	  return args[0] + args[1]*(x-args[2])

  def fcn(params, x, y): #Func to minimize by Minuit
    p = list(params.valuesdict().values()) #p0 and p1
    chi2 = (y - model(x, *p))**2/(sigmaeff_err**2)
    return chi2

  pars = ipanema.Parameters()
  pars.add(dict(name='p0', value=0, min = 0., max=50., free=True))
  pars.add(dict(name='p1', value=1., min=0, max=50, free=True))
  pars.add(dict(name='average', value=0., free=False))
  res = ipanema.optimize(fcn, pars, fcn_args=(sigmat, sigmaeff),
												 method='minos', verbose=True)
  pars = ipanema.Parameters.clone(res.params)
  pars.remove("average")

  _pars = ipanema.Parameters()
  _pars.add(dict(name='p0p', value=0, min = 0., max=50., free=True))
  _pars.add(dict(name='p1p', value=1., min=0 , max=50., free=True))
  _pars.add(dict(name='average', value=av, free=False))
  res = ipanema.optimize(fcn, _pars, fcn_args=(sigmat, sigmaeff),
												 method='minos', verbose=True)
  _pars = ipanema.Parameters.clone(res.params)
  pars = pars + _pars
  return pars


def calibration(data=False, 
								list_pars =False,
								year = 2017,
								mode = "Bs2DsPi",
								figs=False):


  if "DsPi" in mode: 
	  weight = "sigDsSW"
  elif "Bd" in mode:
	  weight = "sigBdSW"
  else:
	  weight = "sigBsSW"
	
  x_error = False 

  branches = ["time", "sigmat"]#, f"{weight}"]

  if "MC" in mode:
	  weight = "time/time"
  else:
    branches += [f"{weight}"]

  sigmat = []
  sigmat_errh = [] #For plotting
  sigmat_errl = [] #For plotting
  sigma_eff = []
  sigmaeff_err = []
  sigmat_err = []
  _n = [] # To norm
  num = 0
  den = 0
  
  for i, p in enumerate(data):
    cdf = uproot.open(p)["DecayTree"].pandas.df(branches=branches)
    print(f"weight = {weight}")
    cdf.eval(f"weight = {weight}", inplace = True)
    print(f"weight = {cdf['weight']}")
    num += cdf.eval(f"weight*sigmat").sum()
    den += cdf.eval(f"weight").sum()
    sigmat.append(list_pars[i]['sigmaAverage'].value)
    sigma_eff.append(list_pars[i][f'sigmaeff'].value)
    sigmaeff_err.append(list_pars[i][f'sigmaeff'].stdev)
    sigmat_err.append(list_pars[i]["sigmaAverage"].stdev)
    #For plotting:
    sigmat_errl.append(sigmat[-1] - cdf["sigmat"].min())
    sigmat_errh.append(cdf["sigmat"].max() - sigmat[-1])
    _n.append(cdf.shape[0])

  sigmat = np.array(sigmat)
  stmax = sigmat.max()
  stmin = sigmat.min()
  sigmat_errl = np.array(sigmat_errl)
  sigmat_errh = np.array(sigmat_errh)
  sigma_eff = np.array(sigma_eff)
  sigmaeff_err = np.array(sigmaeff_err)
  sigmat_err = np.array(sigmat_err)

  if not x_error:
    sigmat_err = 0*sigmat_err

  print(sigmat)
  print(sigmat_errl)
  print(sigmat_errh)
  print(sigma_eff)
  print(sigmaeff_err)
  print(num/den)


  pars = linear_fit(sigmat, sigma_eff, sigmaeff_err, sigmat_err, num/den)

  if figs:
    def model(x, p):
      return p[0] + p[1]*(x)
    sigma_proxy = np.linspace(0., 1.3*stmax, 120)
    fig, axplot = complot.axes_plot()
    axplot.plot(sigma_proxy, 
								model(sigma_proxy, pars.valuesarray()),
		            label = "Calibration")

    y_unc = uncertainty_wrapper(lambda p: model(sigma_proxy, p), pars)
    yl, yh = get_confidence_bands(y_unc)
    yl2, yh2 = get_confidence_bands(y_unc, sigma=2)
    axplot.fill_between(sigma_proxy, yh, yl, color = "red", alpha =0.5)
    axplot.fill_between(sigma_proxy, yh2, yl2, color = "indianred", alpha =0.5)

    # axplot.errorbar(sigmat, sigma_eff, yerr=sigmaeff_err,
    #                 xerr=[sigmat_errl, sigmat_errh],
    #                 fmt='.', color=f'k', label=rf"Prompt $D_s$")
    label = rf"$B_d \rightarrow J/\psi K^*$"
    if "DsPi" in mode:
      label = rf"Prompt $D_s$"

    axplot.errorbar(sigmat, sigma_eff, yerr=sigmaeff_err,
                    xerr=[sigmat_errl, sigmat_errh],
                    fmt='none', 
										linewidth = 0.5,
										capthick = 0.5,
										color=f'k', label=label)

    result = "\n".join((f"p0 = {np.round(pars['p0'].value,5)} +- {np.round(pars['p0'].stdev,5)}",
		                f"p1 = {np.round(pars['p1'].value,5)} +- {np.round(pars['p1'].stdev,5)}"))

    # axplot.text(0.55, 0.42, result, transform=axplot.transAxes, 
				# 				fontsize=14,
		  #           verticalalignment='top')

    axplot.text(0.65, 0.5, f"LHCb {year}", transform=axplot.transAxes, 
								fontsize=14,
	  	            verticalalignment='top')
    units = "ps"
    if "DsPi" in mode:
      units = "fs"

    axplot.set_xlabel(rf'$\sigma_t$ [{units}]', fontsize=17)
    axplot.set_ylabel(rf'$\sigma_{{eff}}$ [{units}]', fontsize=17)
    # axplot.set_yticks(np.arange(stmin, 1.3*stmax, (1.3*stmax-stmin)/10.))
    # axplot.set_xticks(np.arange(stmin, 1.3*stmax, (1.3*stmax-stmin)/10.))
    axplot.set_ylim(0., 1.1*max(sigma_eff))
    axplot.set_xlim(0., 1.3*stmax)
    axplot.legend(fontsize=15, loc="upper left")
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, "linear.pdf"))

  return pars




if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Calibration of the decay time resolution.')
  parser.add_argument('--data', help='Input prompt data.')
  parser.add_argument('--mode', help='Input prompt data.')
  parser.add_argument('--json-bin', help='Calibrations of each bin')
  parser.add_argument('--year', help='Year of data taking')
  parser.add_argument('--output-json', help='Result of fit parameters')
  parser.add_argument('--output-plots', help='Linear plot')


  args = vars(parser.parse_args())
  branches = ["time", "sigmat"]
  data_paths = args["data"].split(',')
  year = args["year"]
  mode = args["mode"]
	
	  
  
  p = [ipanema.Parameters.load(p) for p in args['json_bin'].split(',')]


  pars = calibration(data = data_paths, 
										 list_pars = p,
										 year = year,
										 mode = mode,
										 figs=args['output_plots'])
	
  print(pars)

  pars.dump(args["output_json"])



# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
