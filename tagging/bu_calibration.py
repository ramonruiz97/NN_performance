# Flavour Specific Calibration: Valid for MC and B+
__author__ = ["Ramón Ángel Ruiz Fernández",
              "Marcos Romero Lamas",
              "Veronika Georgieva Chobanova"]


import numpy as np
import ipanema
import complot
from uncertainties import unumpy as unp
from ipanema import uncertainty_wrapper
from ipanema.confidence import get_confidence_bands
import argparse
import yaml
import os
import plotting.configuration


ipanema.initialize(os.environ["IPANEMA_BACKEND"], 1)

prog = ipanema.compile(open('kernels/kernel.cu').read())
pdf = prog.os_calibration
pdf_plot = prog.plot_os

def model(x, prob, p0=0, dp0=0, p1=1, dp1=0, p2=0, dp2=0, eta_bar=0.5):
  pdf(x, prob,
			np.float64(p0), np.float64(dp0),
			np.float64(p1),np.float64(dp1),
			np.float64(p2), np.float64(dp2),
			np.float64(eta_bar), 
			np.int32(len(prob)), global_size=(len(prob),))
  return ipanema.ristra.get(prob)

def omega_plot(q, id, prob,
               omega):
  
  pdf_plot(q, id, prob,
           np.float64(omega),
           np.int32(len(prob)), global_size=(len(prob),))

  return ipanema.ristra.get(prob)
  


def fcn(pars, data):
  p = pars.valuesdict()
  prob = model(data.x, data.prob, **p)
  return -2. * np.log(prob) * ipanema.ristra.get(data.weight)

def fcn_plot(pars, q, id, prob, weight):
  p = pars.valuesdict()
  prob = omega_plot(q, id, prob, **p)
  return -2.*np.log(prob)*ipanema.ristra.get(weight)


def calib(eta, p0, p1, p2, dp0, dp1, dp2, eta_bar, tag=0):
    result = 0
    result += (p0 + tag * 0.5 * dp0)
    result += (p1 + tag * 0.5 * dp1) * (eta - eta_bar)
    result += (p2 + tag * 0.5 * dp2) * (eta - eta_bar) * (eta - eta_bar)
    return result

#TODO: Improve this
def line(eta, x_id,  eta_bar, p):
  # 0-> p0, 1 -> p1, 2->dp0, 3->dp1, 4->p2, 5->dp2
  if len(p)==2:
    result = p[0] + p[1]*(eta-eta_bar)
  elif len(p)==4:
    result = (p[0] + x_id*0.5*p[2]) + (p[1] + x_id*0.5*p[3])*(eta-eta_bar)  
  elif len(p)==6:
    result = (p[0] + x_id*0.5*p[2]) + (p[1] + x_id*0.5*p[3])*(eta-eta_bar) + (p[4] + x_id*0.5*p[5])*(eta-eta_bar)**2

  return result

def omega(eta, b_id, omega, p0=0, dp0=0, p1=1, dp1=0, p2=0, dp2=0, eta_bar=0.5):
  prog.calibrated_mistag(eta, b_id,  omega, 
                         np.float64(p0), np.float64(p1), np.float64(p2),
                         np.float64(dp0), np.float64(dp1), np.float64(dp2), 
                         np.float64(eta_bar),
                         np.int32(len(omega)), global_size=(len(omega),))
  return ipanema.ristra.get(omega)



def os_calibration(data, mode, tagger, order="linear", 
                weight=False, calibrations=False, figs=False):

    with open('config/tagger.yml') as config:
      config = yaml.load(config, Loader=yaml.FullLoader)
	
    cut = config[tagger]['cut']
    q_b = config[tagger]['branch']['decision']
    eta_b = config[tagger]['branch']['eta']

    if "MC_Bs2JpsiPhi" in mode:
      q_b = "B_IFT_InclusiveTagger_TAGDEC"
      eta_b = "B_IFT_InclusiveTagger_TAGETA"

	
    DP = False if "Bs" in mode else True 

    SECOND = True if "parabolic" in order else False

    print(f"Calibrations options are set to \n")
    print(f"Fit assymetries: {DP}")
    print(f"Second Order: {SECOND}")

    pars = ipanema.Parameters()
    pars.add(dict(name='p0', value=0.4, min=0.0, max=2.0, free=True))
    pars.add(dict(name='p1', value=0.9, min=0., max=2.1, free=True))
    pars.add(dict(name='dp0', value=0., min=-1, max=1, free=DP))
    pars.add(dict(name='dp1', value=0., min=-1, max=1, free=DP))
    pars.add(dict(name='p2', value=0.+0.5*SECOND, min=-1., max=2.1, free=SECOND))
    pars.add(dict(name='dp2', value=0., min=-1, max=1., free=SECOND))

    pars.add(dict(name='eta_bar', value=0.5, min=-1, max=1, free=False))

    q = data.df[q_b]
    eta = data.df[eta_b]


    print("Settings")
    print("Calibrations: ", calibrations)
    print("Branch Decision: ", q_b)
    print("Branch Mistag: ", eta_b)
    print("Model: ", order)
    corr = np.sum(data.df.eval(weight))/np.sum(data.df.eval(f"{weight}*{weight}"))
    data.df['q'] = q
    data.df['eta'] = eta
    b_id = 'B_ID_GenLvl' if 'MC' in mode else 'B_ID'
    if ("MC" in mode) and ("Bu" not in mode):
      b_id = "B_TRUEID"

    data.df['id'] = data.df[f'{b_id}']
    data.df['weight'] = data.df.eval(f"{weight}*{corr}")
    print("Weight used to calibrate:", weight)


    data.chop(cut) #TODO: B_BKGCAT cuts?
    print("Cut applied to sample:", cut)

    data.allocate(weight='weight')
    data.allocate(x=['q', 'eta', 'id'])
    data.allocate(prob="0*weight")

    pars['eta_bar'].set(value=data.df.eval(f'weight*eta').sum()/data.df.eval('weight').sum())
    res = ipanema.optimize(fcn, pars, fcn_args=(data,),
							method='minuit', tol=0.05, verbose=False)

    if figs:
      os.makedirs(figs, exist_ok=True)
      fig, axplot = complot.axes_plot()
      nbins = 10
      nplot = 80*nbins
      _pars = ipanema.Parameters()
      _pars.add(dict(name="omega", value=0.3, min=0., max=1., free=True))
      x = np.linspace(0, 0.5, nplot)
      eta_bar = res.params["eta_bar"].value

      w, err, eta, d = [], [], [], []
      eta_arr = np.array(data.df["eta"].array)
      sorted_eta = sorted(eta_arr)
      nbins = 10
      splitting = np.array_split(sorted_eta, nbins)
      binning    = np.array([[s[0], s[-1]] for s in splitting])
      dil =  data.df.eval(f'weight*q').sum()/data.df.eval('weight').sum()
      for eta_lo, eta_hi in binning:
        eta_cut = f"(eta >= {eta_lo} & (eta < {eta_hi}))"
        binweights = data.df.query(eta_cut)['weight']
        sumW = np.sum(binweights)
        eta.append(np.sum(data.df.query(eta_cut)['eta'].array * binweights) / sumW)
        df = data.df.query(eta_cut)
        qd = ipanema.ristra.allocate(np.float64(df["q"].values))
        idd = ipanema.ristra.allocate(np.float64(df["id"].values))
        prob = 0*qd
        weightd = ipanema.ristra.allocate(np.float64(df["weight"].values))

        omega = ipanema.optimize(fcn_plot, _pars, fcn_args=(qd, idd, prob, weightd),
		                             method='minuit', verbose=True).params["omega"]
        w.append(omega.value)
        err.append(omega.stdev)



      y = calib(x, **res.params.valuesdict(), tag=dil)

      names = ["p0", "p1"]
      if DP:
        names += ["dp0", "dp1"]
      if SECOND:
        names += ["p2", "dp2"]

      pars_plot = ipanema.Parameters.build(res.params, names)
      y_unc = uncertainty_wrapper(lambda p: line(x, dil, eta_bar, p), pars_plot)
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
      axplot.set_ylim(0., 0.5)
      axplot.set_yticks(np.arange(0.1, 0.6, 0.1))
      axplot.set_xticks(np.arange(0.1, 0.6, 0.1))
      axplot.legend()
      fig.savefig(os.path.join(figs, "calibration.pdf"))
      
    return res.params


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Tagging Calibration for B+ and MC')
  parser.add_argument('--data', 
                      help='Input data.')

  parser.add_argument('--weight', 
                      help='Weight to apply to data')

  parser.add_argument('--tagger', 
						help='Branch to calibrate')

  parser.add_argument('--mode', 
                      help='Mode')

  parser.add_argument('--year', 
                      help='year of data taking')

  parser.add_argument('--model', default='linear')

  parser.add_argument('--output-json', 
						help='Location to save fit parameters')

  parser.add_argument('--output-plots', 
						help='Location to create plots')

  parser.add_argument('--calibrations', default=False)

  args = vars(parser.parse_args())

  tp = args["data"]
  data = ipanema.Sample.from_root(tp, flatten=False)
  weight = args["weight"]
  mode = args["mode"]

  if ("MC" in mode) and ("Bu" not in mode):
    weight = "B_TRUEID/B_TRUEID"

  tagger = args["tagger"]
  order = args["model"]
  figs = args["output_plots"]

  if args['calibrations']:
    calibrations = args["calibrations"].split(",")
  else:
    calibrations = False
  
  if ("Bu2JpsiKplus" in mode) and ("SSKaon" in tagger):
    order = "parabolic"

  pars = os_calibration(data, mode, tagger, order, weight, calibrations, figs)
  print(pars)
  pars.dump(args['output_json'])
	
