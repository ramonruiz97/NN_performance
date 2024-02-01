# ss_calibration
__author__ = ["Ramón Ángel Ruiz Fernández"]


import numpy as np
import ipanema
import complot
from uncertainties import unumpy as unp
from ipanema import uncertainty_wrapper
from ipanema.confidence import get_confidence_bands
import argparse
import hjson
import yaml
import os
from tagging.spline import print_spline

#Taking Kernels
spline_model = True

ipanema.initialize(os.environ["IPANEMA_BACKEND"], 1)
prog = ipanema.compile(open('kernels/kernel.cu').read())

#Take the functions needed
pdf_spline_wouter = prog.ss_calibration_wouter
pdf_plot = prog.plot_ss

def line(eta, p):
  omb = p[0]*eta + 0.5*(1-p[0])
  ombbar = p[1]*eta + 0.5*(1-p[1])
  result = 0.5*(omb + ombbar)
  return result
    
def calib(eta, 
          f1, f2, **kwards):
    omb = f1*eta + 0.5*(1. - f1) 
    ombbar = f2*eta + 0.5*(1. - f2)
    result = 0.5 * (omb + ombbar)
    return result


def get_sizes(size, BLOCK_SIZE=256):
    '''
    i need to check if this worls for 3d size and 3d block
    '''
    a = size % BLOCK_SIZE
    if a == 0:
        gs, ls = size, BLOCK_SIZE
    elif size < BLOCK_SIZE:
        gs, ls = size, 1
    else:
        a = np.ceil(size/BLOCK_SIZE)
        gs, ls = a*BLOCK_SIZE, BLOCK_SIZE
    return int(gs), int(ls)


def omega_plot(time, sigma, q, id, prob,
	            G=0.6, DG=0.007, DM=17.6, sigma_0 = 0., sigma_1 = 1.,
				tLL=0.3, tUL=15., omega=0.5,
	            **coeffs):

  knots = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 12.0, 15.0]
  coeffs_d = ipanema.ristra.allocate(np.float64(
              print_spline(knots[1:-1], list(coeffs.values()), interpolate=1)
							))
  g_size, l_size = get_sizes(prob.shape[0], 256)
  pdf_plot(time, sigma, q, id, prob, coeffs_d,
		  np.float64(G), np.float64(DG), np.float64(DM),
			np.float64(omega), 
			np.float64(sigma_0), np.float64(sigma_1),
		  np.float64(tLL), np.float64(tUL),
			np.int32(len(prob)), 
           global_size=g_size, local_size = l_size)

  return ipanema.ristra.get(prob)



def model_spline(x, prob,
                 f1 = 1., f2=1.,
	             G=0.6, DG=0.007, DM=17.6, sigma_0 = 0., sigma_1 = 1., mu=0.,
				  tLL=0.3, tUL=15.,
	               **coeffs):

  knots = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 12.0, 15.0]
  coeffs_d = ipanema.ristra.allocate(np.float64(
              print_spline(knots[1:-1], list(coeffs.values()), interpolate=1)
							))
  

  g_size, l_size = get_sizes(prob.shape[0], 256)
  pdf_spline_wouter(x, prob, coeffs_d,
		            np.float64(G), np.float64(DG), np.float64(DM),
			        np.float64(f1), np.float64(f2),
			        np.float64(sigma_0), np.float64(sigma_1), np.float64(mu),
		            np.float64(tLL), np.float64(tUL),
			        np.int32(len(prob)), 
                   global_size=g_size, local_size = l_size)



  return ipanema.ristra.get(prob)





def fcn(pars, data):
    p = pars.valuesdict()

    prob = model_spline(data.x, data.prob, **p) #already in cpu



    chi2 = -2.*np.log(prob)*ipanema.ristra.get(data.weight)

    chi2_gauss = 0

    #Time Parameters:
    if "Bs" in mode:
      chi2_gauss += (p['DG'] - time_pars["DG"].value)**2/(time_pars["DG"].stdev**2)

    chi2_gauss += (p['DM'] - time_pars["DM"].value)**2/(time_pars["DM"].stdev**2)
    chi2_gauss += (1./p['G'] - time_pars["tau"].value)**2/(time_pars["tau"].stdev**2)
    

	#Time resolution:
    cov = np.asmatrix(time_pars.cov(['p0', 'p1']))
    covInv = cov.I

    diff = np.matrix([
			                p['sigma_0'] - time_pars['p0'].value,
		                  p['sigma_1'] - time_pars['p1'].value
		              ])
    chi2_gauss += np.dot(np.dot(diff, covInv), diff.T).item(0) #warning

    chi2_gauss  /= len(chi2)
    return chi2 + chi2_gauss



def fcn_plot(pars, time, sigma, q, id, prob, weight):
  p = pars.valuesdict()
  prob = omega_plot(time, sigma, q, id, prob, **p)
  return -2.*np.log(prob)*ipanema.ristra.get(weight)

def calibration_ss(data, time_pars, mode, tagger, DP=False,
                order="first", weight=False, figs=False):
  
  with open('config/tagger.yml') as config:
    config = yaml.load(config, Loader=yaml.FullLoader)
        
	
  cut = config[tagger]['cut']
  q_b = config[tagger]['branch']['decision']
  eta_b = config[tagger]['branch']['eta']
	
  SECOND = True if "second" in order else False
  
  #Parameters for the fit
  pars = ipanema.Parameters()
  # pars.add(dict(name='p0', value=0.39, min=0.0, max=.6, free=True))
  # pars.add(dict(name='p1', value=1., min=0.4, max=1.3, free=True))
  pars.add(dict(name='f1', value=0.39, min=0.0, max=2., free=True))
  pars.add(dict(name='f2', value=1., min=0., max=2., free=True))
  

  #Time resolution calibration -> we should do from a json
  pars.add(dict(name='sigma_0', value=time_pars['p0'].value, min=0., max=0.7, free=True))
  pars.add(dict(name='sigma_1', value=time_pars['p1'].value, min=0.5, max=1.5, free=True))
  #Decay Time bias
  pars.add(dict(name='mu', value=time_pars['mu'].value, free=False))

  

  pars.add(dict(name=f'c1', value=0.300, min=0., max=100., free=True)) #Convention
  pars.add(dict(name=f'c2', value=0.789, min=0., max=100., free=True))
  pars.add(dict(name=f'c3', value=0.630, min=0., max=100., free=True))
  pars.add(dict(name=f'c4', value=1.2, min=0., max=100., free=True))
  pars.add(dict(name=f'c5', value=1.2, min=0., max=100., free=True))
  pars.add(dict(name=f'c6', value=1.7, min=0., max=100., free=True))
  pars.add(dict(name=f'c7', value=1., min=0., max=100., free=False))
  pars.add(dict(name=f'c8', formula = f'c7 + (c6-c7)*({data.df["time"].max()}-12)/(3-12)'))
  

  Bs = False
  if "Bs" in mode:
    Bs = True

  pars.add(dict(name='G', value=time_pars['G'].value, min=0.8*time_pars["G"].value, max=1.2*time_pars["G"].value, free=True))
  if "Bs" in mode:
    pars.add(dict(name='DG', value=time_pars["DG"].value, min=0.8*time_pars["DG"].value, max=1.2*time_pars["DG"].value, free=Bs))
  else:
    pars.add(dict(name='DG', value=0., free=False))

  pars.add(dict(name='DM', value=time_pars['DM'].value, min=0.8*time_pars["DM"].value, max=1.2*time_pars["DM"].value, free=True))

  pars.add(dict(name='tLL', value=0.3, free=False))
  pars.add(dict(name='tUL', value=15., free=False))

	
  q = data.df[q_b]
  eta = data.df[eta_b]
  corr = np.sum(data.df.eval(weight))/np.sum(data.df.eval(f"{weight}*{weight}"))
	
  data.df['q'] = q
  data.df['eta'] = eta
  b_id = 'B_TRUEID' if 'MC' in mode else 'B_ID'
  data.df['id'] = data.df[f'{b_id}']
  data.df['weight'] = data.df.eval(f"{weight}*{corr}")
  # print(f"Cut of untag events {data.df.query('q != 0').shape[0]}")
  data.chop(cut) #Warning -> I have changed cut of the tagger -> CHECK
  # print(f"Cut of config {data.shape[0]}")
  # exit()
  # if "Bs" in mode:
  #   data.chop("B_PT > 2000")
  print("Number of events: ", data.df.shape[0])
  print("Number of weighted events: ", np.sum(data.df.eval(f'{weight}')))
  print("Number of weighted events corrected: ", np.sum(data.df['weight']))


  data.allocate(weight='weight')
  data.allocate(x=['q', 'eta', 'id', 'time', 'sigmat'])
  data.allocate(prob="0*weight")

  print("Parameters before the fit")
  print(pars)

  res = ipanema.optimize(fcn, pars, fcn_args=(data,),
                         method='minuit', verbose=True, tol=0.05, strategy=1)

  print("Parameters after the fit")
  print(res.params)
  if figs:
    os.makedirs(figs, exist_ok=True)
    fig, axplot = complot.axes_plot()
    nbins = 8
    nplot = 50 * nbins
    _pars = ipanema.Parameters.clone(res.params)
    _pars.remove("f1", "f2")
    _pars.lock()
    _pars.add(dict(name="omega", value=0.3, min=0., max=1., free=True))
    x = np.linspace(0, 0.55, nplot)
    w, err, eta, d = [], [], [], []
    eta_arr = np.array(data.df["eta"].array)
    sorted_eta = sorted(eta_arr)
    splitting = np.array_split(sorted_eta, nbins)
    binning    = np.array([[s[0], s[-1]] for s in splitting])
    # dil =  data.df.eval(f'weight*q').sum()/data.df.eval('weight').sum()
    s = 0
    for eta_lo, eta_hi in binning:
      eta_cut = f"(eta >= {eta_lo} & (eta < {eta_hi}))"
      binweights = data.df.query(eta_cut)['weight']
      sumW = np.sum(binweights)
      eta.append(np.sum(data.df.query(eta_cut)['eta'].array * binweights) / sumW)
      df = data.df.query(eta_cut)
      qd = ipanema.ristra.allocate(np.float64(df["q"].values))
      idd = ipanema.ristra.allocate(np.float64(df["id"].values))
      timed = ipanema.ristra.allocate(np.float64(df["time"].values))
      sigmatd = ipanema.ristra.allocate(np.float64(df["sigmat"].values))
      prob = 0*qd
      dil =  data.df.eval(f'weight*q').sum()/data.df.eval('weight').sum()
      print(f"Dil per bin: {dil}")
      weightd = ipanema.ristra.allocate(np.float64(df["weight"].values))
      omega = ipanema.optimize(fcn_plot, _pars, fcn_args=(timed, sigmatd, qd, idd, prob, weightd),
		                             method='minuit', verbose=False, tol=0.1).params["omega"]
      w.append(omega.value)
      err.append(omega.stdev)
      s += 1
   
    y = calib(x, **res.params.valuesdict())
    pars_plot = ipanema.Parameters.build(res.params, ["f1", "f2"])
    y_unc = uncertainty_wrapper(lambda p: line(x, p), pars_plot)
    yl, yh = get_confidence_bands(y_unc)
    yl2, yh2 = get_confidence_bands(y_unc, sigma=2)
    axplot.fill_between(x, yh, yl, color = "red", alpha =0.5)
    axplot.fill_between(x, yh2, yl2, color = "indianred", alpha =0.5)

    axplot.plot(x, y, label="Calibration")

    axplot.errorbar(x = eta,
                      y = w,
                      xerr = [eta-binning[:,0], binning[:,1]-eta],
                      yerr = err,
                      color = "k",
                      fmt = ".",
                      label = r"$\omega$")

    axplot.set_xlabel(f"$\eta^{{{config[tagger]['label']}}}$")
    axplot.set_ylabel(f"$\omega(\eta^{{{config[tagger]['label']}}})$")
    # axplot.set_title(f"{config[tagger]['label']} Calibration")
    axplot.set_title(f"{tagger} Calibration")
    axplot.set_ylim(0., 0.55)
    axplot.set_xlim(0.1, 0.55)
    axplot.set_yticks(np.arange(0.1, 0.6, 0.1))
    axplot.set_xticks(np.arange(0.1, 0.6, 0.1))
    axplot.legend()
    fig.savefig(os.path.join(figs, f"{year}_{tagger}_calibration.pdf"))
      
  return res.params
	




if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Tagging Calibration for B+ and MC')
  parser.add_argument('--data', help='Input data.')

  parser.add_argument('--weight', help='Weight to apply to data')

  parser.add_argument('--time-res', 
						help='Path parameters calibration resolution')

  parser.add_argument('--time-bias', 
						help='Bias')

  parser.add_argument('--tagger', help='Branch to calibrate')
  parser.add_argument('--offset', help='Offset wildcard')

  parser.add_argument('--mode', help='Input data.')
  parser.add_argument('--version', help='Input data.')

  parser.add_argument('--year', help='Year of data taking')

  parser.add_argument('--model', default='linear')

  parser.add_argument('--output-json', 
						help='Location to save fit parameters')

  parser.add_argument('--output-plots', 
						help='Location to create plots')

  args = vars(parser.parse_args())


  tp = args["data"]
  #Warning
  data = ipanema.Sample.from_root(tp)
  weight = args["weight"]
  tagger = args["tagger"]
  order = args["model"]
  year = args["year"]
  mode = args["mode"]
  if "Bd" in mode:
    weight = "sigBdSW"
  _res = ipanema.Parameters.load(args["time_res"])
  _bias = ipanema.Parameters.load(args["time_bias"])

  time_pars = _res + _bias
  DP = True

  
  


  pars = calibration_ss(data, time_pars, mode, tagger,  DP, order, weight, figs=args["output_plots"])

  pars.dump(args['output_json'])



