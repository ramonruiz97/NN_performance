# ds
#
#

__all__ = ["hypatia_exponential"]
__author__ = ["Marcos Romero", 
              "Ramón Ángel Ruiz Fernández",
              "Asier Pereiro Castro"]
__email__ = ["mromerol@cern.ch",
             "rruizfer@cern.ch"]


# Modules {{{

# from logging import exception
import os
import typing
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
# import re
from ipanema import (ristra, Sample, splot)
import matplotlib.pyplot as plt
# from utils.strings import printsec, printsubsec
from utils.helpers import cuts_and
from scipy import interpolate 
import pandas as pd

import complot
import ipanema
import matplotlib.pyplot as plt
import numpy as np
import uproot3 as uproot
from ipanema import Sample, ristra, splot
import yaml

import config



# initialize ipanema3 and compile lineshapes
ipanema.initialize(os.environ["IPANEMA_BACKEND"], 1)
# ipanema.initialize("opencl", 1)

prog = ipanema.compile(
"""
#define USE_DOUBLE 1
#include <exposed/kernels.ocl>
"""
)

# }}}


# ipatia + exponential {{{

def hypatia_exponential(
    mass, prob,
    fsigBs=0, fsigBd=0, fcomb=0, # fbackground=0,
    muBs=0, sigmaBs=10,
    muBd=0, sigmaBd=1,
    lambd=0, zeta=0,
    beta=0, aL=0, nL=0, aR=0, nR=0, alpha=0, norm=1, mLL=None, mUL=None,
    *args, **kwargs):
    """
    Hypatia exponential Bs and Bd lineshape for DsPi
    """
    # Bs hypatia {{{
    prog.py_ipatia(prob, mass, np.float64(muBs), np.float64(sigmaBs),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
    )
    pdfBs = 1.0 * prob.get()
    prob = 0 * prob
    # }}}
    # Bd hypatia {{{
    prog.py_ipatia(prob, mass, np.float64(muBd), np.float64(sigmaBd),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
    )
    pdfBd = 1.0 * prob.get()
    prob = 0 * prob
    # }}}

    backgr = ristra.exp(mass * alpha).get()

    # normalize
    _x = ristra.linspace(mLL, mUL, 2000)
    _y = _x * 0
    # Bs hypatia integral {{{
    prog.py_ipatia( _y, _x, np.float64(muBs), np.float64(sigmaBs),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(_x)),
    )
    nBs = np.trapz(ristra.get(_y), ristra.get(_x))
    # }}}
    # Bd hypatia integral {{{
    prog.py_ipatia( _y, _x, np.float64(muBd), np.float64(sigmaBd),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(_x)),
    )
    nBd = np.trapz(ristra.get(_y), ristra.get(_x))
    # }}}
    nbackgr = np.trapz(ristra.get(ristra.exp(_x * alpha)), ristra.get(_x))

    # compute pdf value
    ans = fsigBs * pdfBs/nBs                              # Bs
    ans += fsigBd * pdfBd/nBd                             # Bd
    ans += fcomb * backgr/nbackgr                         # comb
    return norm * ans

# }}}


# Bs mass fit function {{{

def mass_fitter(
    odf: pd.DataFrame,
    mass_range: typing.Optional[tuple]=None,
    mass_branch: str="B_ConstJpsi_M_1",
    mass_weight: str="B_ConstJpsi_M_1/B_ConstJpsi_M_1",
    cut: typing.Optional[str]=None,
    figs: typing.Optional[str]=None,
    model: typing.Optional[str]=None,
    templates: typing.Optional[str]=None,
    # has_bd: bool=False,
    trigger: typing.Optional[str]="combined",
    input_pars: typing.Optional[str]=None,
    sweights: bool=False,
		prefit: bool=False,
    verbose: bool=False) -> typing.Tuple[ipanema.Parameters, typing.Optional[dict]]:

    # mass range cut
    if not mass_range:
        mass_range = (min(odf[mass_branch]), max(odf[mass_branch]))
    mLL, mUL = mass_range
    mass_cut = f"{mass_branch} > {mLL} & {mass_branch} < {mUL}"

    # mass cut and trigger cut
    current_cut = cuts_and(mass_cut, cut)

    # Select model and set parameters {{{
    #    Select model from command-line arguments and create corresponding set
    #    of paramters

    # Chose model {{{

    if model == 'ipatia':
        signal_pdf = hypatia_exponential
    else:
        raise ValueError(f"{model} cannot be assigned as mass model")


    def roo_hist_pdf(hist):
      h, e = hist
      c = 0.5*(e[1:]+e[:-1])
      # fun = interpolate.interp1d(c, h, fill_value=(0.,0.), bounds_error=False)
      fun = interpolate.interp1d(c, h, fill_value="extrapolate", bounds_error=False)
      _x = np.linspace(mLL, mUL, 2000)
      norm = np.trapz(fun(_x), _x)
      if norm==0:
        norm=1.
      return interpolate.interp1d(_x, fun(_x)/norm)

    def norm_hist(hist, mLL, mUL):
      h, e = hist
      c = 0.5*(e[1:]+e[:-1])
      fun = interpolate.interp1d(c, h, fill_value="extrapolate", bounds_error=False)
      _x = np.linspace(mLL, mUL, 2000)
      norm = np.trapz(fun(_x), _x)
      return norm

    bs_dskx = uproot.open(templates[0])
    bs_dskx = bs_dskx[bs_dskx.keys()[0]].numpy()
    pdf_bs_dskx = roo_hist_pdf(bs_dskx)

    bs_dsx = uproot.open(templates[1])
    bs_dsx = bs_dsx[bs_dsx.keys()[0]].numpy()
    pdf_bs_dsx = roo_hist_pdf(bs_dsx)

    DsK_mass_template = uproot.open(templates[2])


    h_Bs2DsstPi = DsK_mass_template['m_Bs2DsstPi'].numpy()
    pdf_Bs2DsstPi = roo_hist_pdf(h_Bs2DsstPi)
    h_Bs2DsRho = DsK_mass_template['m_Bs2DsRho'].numpy()
    pdf_Bs2DsRho = roo_hist_pdf(h_Bs2DsRho)
    h_Lb2LcPi = DsK_mass_template['m_Lb2LcPi'].numpy()
    pdf_Lb2LcPi = roo_hist_pdf(h_Lb2LcPi)
    h_Bd2DPiX = DsK_mass_template['m_Bd2DPiX'].numpy()
    pdf_Bd2DPiX =roo_hist_pdf(h_Bd2DPiX)

    _x = np.linspace(mLL, mUL, 2000)
    plt.plot(_x, pdf_Bd2DPiX(_x), label="pdf-Bd2DPiX")
    plt.plot(_x, pdf_Bs2DsRho(_x), label="pdf-Bs2DsRho")
    plt.plot(_x, pdf_Bs2DsstPi(_x), label="pdf-Bs2DsstPi")
    plt.plot(_x, pdf_Lb2LcPi(_x), label="pdf-Lb2LcPi")
    plt.plot(_x, pdf_bs_dskx(_x), label="pdf-bs-dskx")
    plt.plot(_x, pdf_bs_dsx(_x), label="pdf-bs-dsx")
    plt.legend()
    plt.savefig("temp.pdf")
    plt.close()
    # h_Bs2DsstPi = DsK_mass_template['m_Bs2DsstPi'].numpy()
    # h_Bs2DsRho =  DsK_mass_template['m_Bs2DsRho'].numpy()
    # h_Lb2LcPi = DsK_mass_template['m_Lb2LcPi'].numpy()
    # h_Bd2DPiX = DsK_mass_template['m_Bd2DPiX'].numpy()


    def pdf(mass, prob, norm=1, *args, **kwargs):
      mass_h = ristra.get(mass)
      _prob = ristra.get(signal_pdf(mass=mass, prob=prob, norm=norm, *args, **kwargs))
      _prob += norm * kwargs['fDsK'] * pdf_bs_dskx(mass_h)
      _prob += norm * kwargs['fDsX'] * pdf_bs_dsx(mass_h)
      _prob += norm * kwargs['fDsstPi'] * pdf_Bs2DsstPi(mass_h)
      _prob += norm * kwargs['fDsRho'] * pdf_Bs2DsRho(mass_h)
      _prob += norm * kwargs['fLb'] * pdf_Lb2LcPi(mass_h)
      _prob += norm * kwargs['fBdDPi'] * pdf_Bd2DPiX(mass_h)
      return _prob


    def fcn(params, data):
        p = params.valuesdict()
        prob = ristra.get(pdf(mass=data.mass, prob=data.pdf, **p, norm=1))
        return -2.0 * np.log(prob) * ristra.get(data.weight)

    # }}}


    pars = ipanema.Parameters()
	
    # add parameters for all fractions
    pars.add(dict(name="fsigBs", value=0.46, min=0.0, max=1, free=True,
                  latex=r"f_{B_s^0 \rightarrow D_s^- \pi^+}"))

    pars.add(dict(name="muBs", value=5366, min=5350, max=5390, free=True,
                    latex=r"\mu_{B_s^0}"))

    pars.add(dict(name="sigmaBs", value=18, min=0.1, max=20, free=True,
                    latex=r"\sigma_{B_s^0}"))

    pars.add(dict(name='fcomb', 
                  min=0.,
                  max=1.,
                  free=True,
                  latex=r"f_{\textrm{Combinatorial}}"))

    pars.add(dict(name="fbkg", formula="(1-fsigBs-fcomb)",
                latex=r"f_{\textrm{bkg}}"))

    #Bkgs 7
    # pars.add(dict(name="fsigBd_proxy", value=0.017, min=0.0, max=1, free=True))
    # pars.add(dict(name='fLb_proxy', value=0.03, min=0.0, max=1, free=True))
    # pars.add(dict(name='fBdDsPi_proxy', value=0.00, min=0.0, max=1, free=True))
    # pars.add(dict(name='fDsX_proxy', value=0.0, min=0.0, max=1, free=True))
    # pars.add(dict(name='fDsstPi_proxy', value=0.35, min=0.0, max=1, free=True))
    # pars.add(dict(name='fDsRho_proxy',  value=0.1, min=0.0, max=1, free=True))
    # pars.add(dict(name='fDsK_proxy', formula="1-fDsRho_proxy-fDsstPi_proxy-fDsX_proxy-fBdDsPi_proxy-fLb_proxy-fsigBd_proxy"))
    #
    #
    # pars.add(dict(name='fDsK',
    #                 formula = f"fbkg*fDsK_proxy",
    #                 latex=r"f_{B_s^0 \rightarrow D_s^- K^+}"))
    # pars.add(dict(name='fBdDsPi',
    #             formula = f"fbkg*fBdDsPi_proxy",
    #             latex=r"f_{B^0 \rightarrow D_s^- \pi^+ x}"))
    # pars.add(dict(name='fLb',
    #             formula = f"fbkg*fLb_proxy",
    #             latex=r"f_{\Lambda_b \rightarrow \Lambda_c^- \pi^+}"))
    # pars.add(dict(name='fDsstPi',
    #             formula = f"fbkg*fDsstPi_proxy",
    #             latex=r"f_{B_s^0 \rightarrow D_s^{*-} \pi^+}"))
    # pars.add(dict(name='fDsRho', 
    #             formula = f"fbkg*fDsRho_proxy",
    #             latex=r"f_{B_s^0 \rightarrow D_s^- \rho^+}"))
    # pars.add(dict(name='fDsX',
    #             formula = f"fbkg*fDsX_proxy",
    #             latex=r"f_{B_s^0 \rightarrow D_s^- x}"))
    # pars.add(dict(name='fsigBd',
    #             formula = f"fbkg*fsigBd_proxy",
    #             latex=r"f_{B^0 \rightarrow D_s^- \pi^+}"))

    pars.add(dict(name="fsigBd_proxy", value=0.017, min=0.0, max=1, free=True))
    pars.add(dict(name='fDsK_proxy', value=0.03, min=0.0, max=1, free=True))
    pars.add(dict(name='fBdDPi_proxy', value=0.00, min=0.0, max=1, free=True)) #To 0 as china do
    pars.add(dict(name='fLb_proxy', value=0.2, min=0.0, max=1, free=True))
    pars.add(dict(name='fDsstPi_proxy', value=0.35, min=0.0, max=1, free=True))
    pars.add(dict(name='fDsRho_proxy',  value=0.1, min=0.0, max=1, free=True))
    pars.add(dict(name='fDsX_proxy', value=0.00, min=0.0, max=1, free=False)) #To 0 as china do (actually not sure what it is)

    pars.add(dict(name='norm_bkg',
                  formula = "fDsK_proxy + fBdDPi_proxy + fLb_proxy + fDsstPi_proxy + fDsRho_proxy + fDsX_proxy + fsigBd_proxy",
                  latex=r"norm_bkg"))
    
	  #Not taken into account
    pars.add(dict(name='fBdDPi',
                formula = f"fbkg*fBdDPi_proxy/norm_bkg",
                latex=r"f_{B_d^0 \rightarrow D^- \pi^+ X}")) #To 0
    pars.add(dict(name='fDsX',
                formula = f"fbkg*fDsX_proxy/norm_bkg",
                latex=r"f_{B_s^0 \rightarrow D_s^- \pi^+ x}")) #To 0
    
	  #Backgrounds
    pars.add(dict(name='fDsK',
                    formula = f"fbkg*fDsK_proxy/norm_bkg",
                    latex=r"f_{B_s^0 \rightarrow D_s^- K^+}"))
    pars.add(dict(name='fLb',
                formula = f"fbkg*fLb_proxy/norm_bkg",
                latex=r"f_{\Lambda_b \rightarrow \Lambda_c^- \pi^+}"))
    pars.add(dict(name='fDsstPi',
                formula = f"fbkg*fDsstPi_proxy/norm_bkg",
                latex=r"f_{B_s^0 \rightarrow D_s^{*-} \pi^+}"))
    pars.add(dict(name='fDsRho', 
                formula = f"fbkg*fDsRho_proxy/norm_bkg",
                latex=r"f_{B_s^0 \rightarrow D_s^- \rho^+}"))


    pars.add(dict(name='fsigBd',
                formula = f"fbkg*fsigBd_proxy/norm_bkg",
                latex=r"f_{B^0 \rightarrow D_s^- \pi^+}"))



    pars.add(dict(name="muBd", formula=f"muBs-87.19",
                latex=r"\mu_{B_d^0}"))

    # standard deviation for Bs and Bd lineshape
    pars.add(dict(name="sigmaBs", value=18, min=0.1, max=100, free=True,
                latex=r"\sigma_{B_s^0}"))
    pars.add(dict(name="sigmafrac", value=0.984, min=0., max=2., free=False,
                latex=r"\sigma_{B_d^0}/\sigma_{B_s^0}"))
    pars.add(dict(name="sigmaBd", formula="sigmaBs*sigmafrac",
                latex=r"\sigma_{B_d^0}"))


    # add parameters depending on model
    # print(input_pars)
    if input_pars:
      _pars = ipanema.Parameters.clone(input_pars)
      _pars.lock()
      _pars.remove("fsigBs", "sigmaBs", "muBs", "sigmafrac", "sigBd", "fsigBd", "fcomb", "muBd", "sigmaB0", "mLL", "mUL")
      pars = pars + _pars


    # Combinatorial background
    pars.add(dict(name='alpha', value=-0.00313, min=-1, max=1, free=True,
                  latex=r'b'))


    if "fbkg" in input_pars:
      pars = ipanema.Parameters()
      pars.add(dict(name="fsigBs", value=0.46, min=0.3, max=1., free=True,
                  latex=r"f_{B_s^0 \rightarrow D_s^- \pi^+}"))

      pars.add(dict(name="fcomb", value=0.46, min=0.0, max=1., free=True,
                  latex=r"f_{comb}"))

      pars.add(dict(name="muBs", value=5366, min=5350, max=5390, free=True,
                    latex=r"\mu_{B_s^0}"))

      pars.add(dict(name="sigmaBs", value=18, min=0.1, max=100, free=True,
                    latex=r"\sigma_{B_s^0}"))

      _pars = ipanema.Parameters.clone(input_pars)
      _pars.lock()
      _pars.remove("fsigBs", "fcomb", "muBs", "sigmaBs")
      pars = pars + _pars

    # finally, set mass lower and upper limits
    pars.add(dict(name="mLL", value=mLL, free=False,
                  latex=r"m_{ll}"))
    pars.add(dict(name="mUL", value=mUL, free=False,
                  latex=r"m_{ul}"))



    # }}}
    # Allocate the sample variables {{{

    print(f"Cut: {current_cut}")
    print(f"Mass branch: {mass_branch}")
    print(f"Mass weight: {mass_weight}")
    rd = Sample.from_pandas(odf)
    print(f"Events (before cut): {rd.shape}")
    _proxy = np.float64(rd.df[mass_branch]) * 0.0
    rd.chop(current_cut)
    print(f"Events (after cut): {rd.shape}")
    rd.allocate(mass=mass_branch)
    rd.allocate(pdf=f"0*{mass_branch}", weight=mass_weight)

    # }}}

    # fit {{{

    res = False
    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                           method='minuit', verbose=verbose, strategy=2, tol=0.1)

    if res and not prefit:
        print(res)
        fpars = ipanema.Parameters.clone(res.params)
        fpars.lock()
        fpars.unlock("fsigBs", "fcomb")
        res = ipanema.optimize(fcn, fpars, fcn_kwgs={'data': rd},
                           method='minuit', verbose=verbose, strategy=1,
                           tol=0.05)
        fpars = ipanema.Parameters.clone(res.params)
        
    else:
        fpars = ipanema.Parameters.clone(res.params)
    
    # }}}


    # plot the fit result {{{

    _mass = ristra.get(rd.mass)
    _weight = rd.df.eval(mass_weight)

    fig, axplot, axpull = complot.axes_plotpull()
    hdata = complot.hist(_mass, weights=_weight, bins=100, density=False)
    axplot.errorbar(hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr,
                    fmt=".k")

    proxy_mass = ristra.linspace(min(_mass), max(_mass), 1000)
    proxy_prob = 0 * proxy_mass

    # plot subcomponents
    names = ['fsigBs', 'fsigBd', 'fDsK', 'fBdDPi', 'fLb', 'fDsstPi', 'fDsRho', 'fDsX', 'fcomb']
    label = {  
		          'fsigBs' : r"f_{B_s^0 \rightarrow D_s^- \pi^+}",
              'fDsK' : r"f_{B_s^0 \rightarrow D_s^- K^+}", 
              'fBdDPi': r"f_{B^0 \rightarrow D^- \pi^+ }",
              'fLb' : r"f_{\Lambda_b \rightarrow \Lambda_c^- \pi^+}",
              'fDsstPi' : r"f_{B_s^0 \rightarrow D_s^{*-} \pi^+}",
              'fDsRho' : r"f_{B_s^0 \rightarrow D_s^- \rho^+}",
              'fDsX':  r"f_{B_s^0 \rightarrow D_s^- x}",
              'fsigBd' : r"f_{B^0 \rightarrow D_s^- \pi^+}",
		          'fcomb' : r"f_{\textrm{Combinatorial}}"
	}
    for icolor, pspecie in enumerate(names):
        _color = f"C{icolor+1}"
        if pspecie.startswith('f'):
          _label = rf"${label[pspecie].split('f_{')[-1][:-1]}$"
          print(_label)
          _p = ipanema.Parameters.clone(fpars)
          for f in _p.keys():
                if f.startswith('f'):
                  if f != pspecie:
                    _p[f].set(value=0, min=-np.inf, max=np.inf)
                  else:
                    _p[f].set(value=fpars[pspecie].value, min=-np.inf, max=np.inf)
          if _p[pspecie].value>=1.0e-7:
            _prob = pdf(proxy_mass, proxy_prob, **_p.valuesdict(), norm=hdata.norm)
            axplot.plot(ristra.get(proxy_mass), ristra.get(_prob), color=_color, label=_label)

    # plot fit with all components and data
    _p = ipanema.Parameters.clone(fpars)
    _prob = pdf(proxy_mass, proxy_prob, **_p.valuesdict(), norm=hdata.norm)
    axplot.plot(ristra.get(proxy_mass), _prob, color="C0")
    pulls = complot.compute_pdfpulls(ristra.get(proxy_mass), ristra.get(_prob),
                                     hdata.bins, hdata.counts, *hdata.yerr)
    axpull.fill_between(hdata.bins, pulls, 0, facecolor="C0", alpha=0.5)


    # label and save the plot
    axpull.set_xlabel(r"$m(D_s \pi)$ [MeV/$c^2$]")
    axplot.set_ylabel(rf"Candidates / ({np.round(hdata.xerr[1][0], 1)} MeV/$c^2$)")
    axpull.set_ylim(-6.5, 6.5)
    axplot.set_ylim(1., 1.15 * np.max(_prob))
    axpull.set_yticks([-5, 0, 5])
    # axpull.set_yticks([-2, 0, 2])
    axpull.hlines(3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
    axpull.hlines(-3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
    axplot.legend(loc="upper right", prop={'size': 8})
    type = "fit"
    if prefit:
      type = "prefit"
    if figs:
        os.makedirs(figs, exist_ok=True)
        fig.savefig(os.path.join(figs, f"{year}_{type}_fit.pdf"))
    axplot.set_yscale("log")
    try:
        axplot.set_ylim(1., 50 * np.max(_prob))
    except:
        print("axes not scaled")
    if figs:
        fig.savefig(os.path.join(figs, f"{year}_{type}_logfit.pdf"))
    plt.close()

    # }}}

    if prefit:
      norm_dskx = norm_hist(bs_dskx, 5300, 5600)
      norm_dsx = norm_hist(bs_dsx, 5300, 5600)
      norm_Bs2DsstPi = norm_hist(h_Bs2DsstPi, 5300, 5600)
      norm_Bs2DsRho = norm_hist(h_Bs2DsRho, 5300, 5600)
      norm_Lb2LcPi = norm_hist(h_Lb2LcPi, 5300, 5600)
      norm_Bd2DPiX = norm_hist(h_Bd2DPiX, 5300, 5600)
      export = {}
		  
      fDsK_proxy = fpars["fDsK_proxy"].value * norm_dskx
      fDsX_proxy = fpars["fDsX_proxy"].value * norm_dsx
      fDsstPi_proxy = fpars["fDsstPi_proxy"].value*norm_Bs2DsstPi
      fDsRho_proxy = fpars["fDsRho_proxy"].value*norm_Bs2DsRho
      fLb_proxy = fpars["fLb_proxy"].value*norm_Lb2LcPi
      fBdDPi_proxy = fpars["fBdDPi_proxy"].value*norm_Bd2DPiX
		  
      sum =  fDsK_proxy 
      sum += fDsX_proxy
      sum += fDsstPi_proxy 
      sum += fDsRho_proxy
      sum += fLb_proxy
      sum += fBdDPi_proxy 


      __mass = ristra.linspace(5300, 5600, 1000)
      __prob = 0 * __mass
      __p = ipanema.Parameters.clone(fpars)
      for f in __p.keys():
        if f.startswith('f'):
          if f != "fsigBd":
            __p[f].set(value=0, min=-np.inf, max=np.inf)
          else:
            __p[f].set(value=1., min=-np.inf, max=np.inf)

      __prob = pdf(__mass, __prob, **__p.valuesdict())
      norm_sigBd = np.trapz(ipanema.ristra.get(__prob), ipanema.ristra.get(__mass))

      fsigBd_proxy = fpars["fsigBd_proxy"].value*norm_sigBd
      sum += fsigBd_proxy

      export["fDsK_proxy"]  = fDsK_proxy/sum
      export["fDsX_proxy"] = fDsX_proxy/sum
      export["fDsstPi_proxy"] = fDsstPi_proxy/sum
      export["fDsRho_proxy"]  = fDsRho_proxy / sum
      export["fLb_proxy"]     = fLb_proxy / sum
      export["fBdDPi_proxy"] = fBdDPi_proxy / sum
      export["fsigBd_proxy"]  = fsigBd_proxy / sum
      print("Export")
      print(export)
      print("Fpars before")
      print(fpars)

      for k in res.params.keys():
        try:
          if "_proxy" in k:
            fpars.remove(k)
            fpars.remove(k.replace("_proxy", ""))
            fpars.add(dict(name=f"{k}", value = export[k] ))
            fpars.add(dict(name=f"{k.replace('_proxy', '')}", formula=f"fbkg*{export[k]}"))
        except:
          continue

      fpars.remove("norm_bkg")
      fpars.add(dict(name='norm_bkg',
                  formula = "fDsK_proxy + fBdDPi_proxy + fLb_proxy + fDsstPi_proxy + fDsRho_proxy + fDsX_proxy + fsigBd_proxy",
                  latex=r"norm_bkg"))

			 

			
      print("Fpars after")
      print(fpars)
		  
		
    if sweights:
        # compute sWeights if asked {{{
        print(list(fpars))
        _fpars = ipanema.Parameters.clone(fpars)
        names = ["fDsK", "fBdDPi", "fLb", "fDsstPi", "fDsRho", "fDsX", "fsigBd"]
        for p in names:
          _fpars.add(dict(name =_fpars[p].name, value=fpars[p].value))
        print(_fpars)
        # separate paramestes in yields and shape parameters
        # names = ["fsigBs", "fcomb"]
        # print
        _yields = ipanema.Parameters.find(_fpars, "fsigBs.*") + ["fcomb"] + ["fbkg"]
        # _yields = [p for p in pars if pars[p].name in names]
        # _yields = [p for p in pars if pars[p].name=="fsigBs"]
        _pars = list(_fpars)
        [_pars.remove(_y) for _y in _yields]
        _yields = ipanema.Parameters.build(_fpars, _yields)
        _pars = ipanema.Parameters.build(_fpars, _pars)

        # WARNING: Breaking change!  -- February 4th
        #          sw(p, y, len(data)) * wLb != sw(p, y, wLb.sum())
        #          which one is correct? Lera does the RS and I did LS
        # sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.merr, rd.pdf, *x, **y), _pars, _yields, ristra.get(rd.weight).sum())
        sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.pdf, *x, **y), _pars, _yields, rd.weight)
        for k,v in sw.items():
            _sw = np.copy(_proxy)
            _sw[list(rd.df.index)] = v
            sw[k] = _sw
        # print("sum of wLb", np.sum( rd.df.eval(mass_weight).values ))
        return (fpars, sw)
        # print("sum of wLb", np.sum( rd.df.eval(mass_weight).values ))
        return (fpars, sw)

    # }}}

    return (fpars, False)


# }}}


# command-line interface {{{

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="arreglatelas!")
    p.add_argument('--sample')
    p.add_argument('--input-params', default=False)
    p.add_argument('--templates', default=None)
    p.add_argument('--output-params')
    p.add_argument('--output-figures')
    p.add_argument('--output-sample')
    p.add_argument('--mass-model', default='ipatia')
    p.add_argument('--mass-weight', default="")
    p.add_argument('--mass-branch', default='B_ConstJpsi_M_1')
    p.add_argument('--mass-bin', default=False)
    p.add_argument('--trigger')
    p.add_argument('--sweights')
    p.add_argument('--mode')
    p.add_argument('--year')
    p.add_argument('--version')
    p.add_argument('--tagger')
    args = vars(p.parse_args())

    templates = args['templates'].split(',')


    if args["output_sample"]:
        sweights = True
    else:
        sweights = False

    if args["input_params"]:
        input_pars = ipanema.Parameters.load(args["input_params"])
    else:
        input_pars = False



    mass_branch = args['mass_branch']
    branches = [ mass_branch ]
    year = args["year"]

    if args["mass_weight"]:
        mass_weight = args["mass_weight"]
        branches += [mass_weight]
    else:
        mass_weight = f"{mass_branch}/{mass_branch}"

    mass_range = (5300, 5600)  # narrow window

    cut = None
    cut = "B_PT>2e3"

    branches += ["B_PT"]
    prefit = False
    if "prefit" in args["output_params"]: 
      mass_range = (5100, 5600)  # wide window
      prefit = True
    print(args["output_params"])
    print(mass_range)


    sample = Sample.from_root(args["sample"], branches=branches)


    pars, sw = mass_fitter(
        sample.df,
        mass_range=mass_range,
        mass_branch=mass_branch,
        mass_weight=mass_weight,
        trigger=args["trigger"],
        figs=args["output_figures"],
        model=args["mass_model"],
        cut=cut,
        sweights=sweights,
		    prefit = prefit,
        input_pars=input_pars,
        verbose=True,
        templates=templates)

    pars.dump(args["output_params"])
    if sweights:
        sample = Sample.from_root(args["sample"], flatten=None)
        _proxy = np.float64(sample.df['time']) * 0.0
        for w in ["fsigBs", "fcomb", "fbkg"]:
          _weight = sw
          _proxy += _weight[w]
          sample.df[f"{w[1:]}SW"] = _proxy
			  #Warning FIXME
        sample.chop("B_M>5300 & B_M<5600")
        # sample.chop("sigmat < 0.15")
        with uproot.recreate(args['output_sample']) as f:
          _branches = {}
          for k, v in sample.df.items():
            if 'int' in v.dtype.name:
              _v = np.int32
            elif 'bool' in v.dtype.name:
              _v = np.int32
            else:
             _v = np.float64
            _branches[k] = _v
          mylist = list(dict.fromkeys(_branches.values()))
          f["DecayTree"] = uproot.newtree(_branches)
          f["DecayTree"].extend(sample.df.to_dict(orient='list'))

# }}}


# vim: fdm=marker



# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
