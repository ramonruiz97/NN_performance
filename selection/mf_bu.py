__author__ = ["Marcos Romero", "Ramón Ángel Ruiz Fernández"]
__email__ = ["mromerol@cern.ch", "rruizfer@CERN.CH"]
__all__ = ["mass_fitter"]

import argparse
import os

import complot
import ipanema
import matplotlib.pyplot as plt
import numpy as np
from ipanema import Sample, ristra, splot
from utils.helpers import cuts_and, trigger_cut
import uproot3 as uproot


# initialize ipanema3 and compile lineshapes
ipanema.initialize(os.environ["IPANEMA_BACKEND"], 1)

prog = ipanema.compile(
    """
#define USE_DOUBLE 1
#include <exposed/kernels.ocl>
"""
)


# }}}


def ipatia_exponential(mass, signal, fsigBu=0, fexp=0,
    muBu=0, sigmaBu=10, lambd=0, zeta=0, beta=0,
    aL=0, nL=0, aR=0, nR=0, b=0, norm=1, mLL=None, mUL=None,
    ):
    mLL, mUL = ristra.min(mass), ristra.max(mass)
    # ipatia
    prog.py_ipatia( signal, mass, np.float64(muBu), np.float64(sigmaBu),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
    )
    pdfBs = 1.0 * signal.get()
    # normalize
    _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
    _y = _x * 0
    prog.py_ipatia( _y, _x, np.float64(muBu), np.float64(sigmaBu),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(_x)),
    )
    nBs = np.trapz(ristra.get(_y), ristra.get(_x))
    pComb = 0
    if fexp > 0:
        prog.kernel_exponential(
            signal,
            mass,
            np.float64(b),
            np.float64(mLL),
            np.float64(mUL),
            global_size=(len(mass)),
        )
        pComb = ristra.get(signal)
    # compute pdf value
    ans = fsigBu * (pdfBs / nBs) + fexp * pComb
    return norm * ans





# }}}


# Bu mass fit function {{{


def mass_fitter(
    odf,
    mass_range=False,
    mass_branch="B_ConstJpsi_M",
    mass_weight="B_ConstJpsi_M/B_ConstJpsi_M",
    mode="Bu2JpsiKplus",
    cut=False,
    figs=False,
    model=False,
    trigger=False,
    input_pars=False,
    sweights=False,
    verbose=False,
    tagger=False
):

    # mass range cut
    if not mass_range:
        mass_range = (min(odf[mass_branch]), max(odf[mass_branch]))
    mLL, mUL = mass_range
    mass_cut = f"{mass_branch} >= {mLL} & {mass_branch} <= {mUL}"

    current_cut = cuts_and(mass_cut, cut)
    current_cut = trigger_cut(trigger, cuts_and(mass_cut, cut))

    print(f"CUT:  {current_cut}")

    pars = ipanema.Parameters()
    # Create common set of parameters (all models must have and use)
    pars.add(
             dict(name=f"fsigBu", value=0.95, min=0.2, max=1, free=True, latex=r"N_{B_d}")
          )
    pars.add(dict(name="muBu", value=5280, min=5200, max=5350, latex=r"\mu_{B_d}"))
    pars.add(
            dict(
                name="sigmaBu",
                value=11,
                min=1,
                max=50,
                free=True,
                latex=r"\sigma_{B_d}",
            )
        )
    if input_pars:
      _pars = ipanema.Parameters.clone(input_pars)
      _pars.lock()
      pars.remove("fsigBu", "muBu", "sigmaBu")
      _pars.unlock("fsigBu", "muBu", "sigmaBu", "b")
      pars = pars + _pars
    else:
      #Prefit stage
      pars["fsigBu"].value = 1
      pars["fsigBu"].free = False
      # Hypatia tails {{{
      pars.add(
          dict(
              name="lambd",
              value=-2.5,
              min=-4,
              max=-1.1,
              free=True,
              latex=r"\lambda",
          )
      )
      pars.add(dict(name="zeta", value=1e-5, min =-1e-4, max= 1e-4, free=False, latex=r"\zeta"))
      pars.add(dict(name="beta", value=0.0, min =-1.,  max=1., free=False, latex=r"\beta"))
      pars.add(
            dict(name="aL", value=1.23, min=0.5, max=5.5, free=True, latex=r"a_l")
              )
      pars.add(dict(name="nL", value=1.05, min=0, max=6, free=True, latex=r"n_l"))
      pars.add(
          dict(name="aR", value=1.03, min=0.5, max=5.5, free=True, latex=r"a_r")
      )
      pars.add(dict(name="nR", value=1.02, min=0, max=6, free=True, latex=r"n_r"))
      # }}}

      #Combinatorial:
      pars.add(dict(name='b', value=-0.05, min=-1, max=1, free=False, latex=r'b'))
      pars.add(dict(name='fexp', formula="1-fsigBu", latex=r'f_{comb}'))


    print("The following parameters will be fitted")
    print(pars)
    pdf = ipatia_exponential


    def fcn(params, data):
        p = params.valuesdict()
        prob = pdf(data.mass, data.pdf, **p)
        return -2.0 * np.log(prob) * ristra.get(data.weight)

    # }}}

    # Allocate the sample variables {{{

    print(f"Cut: {current_cut}")
    print(f"Mass branch: {mass_branch}")
    print(f"Mass weight: {mass_weight}")
    rd = Sample.from_pandas(odf)
    _proxy = np.float64(rd.df[mass_branch]) * 0.0
    rd.chop(current_cut)
    rd.allocate(mass=mass_branch, pdf=f"0*{mass_branch}", weight=mass_weight)

    # }}}

    # perform the fit {{{

    res = ipanema.optimize(
        fcn,
        pars,
        fcn_kwgs={"data": rd},
        method="minuit",
        verbose=True,
        strategy=1,
        tol=0.05,
    )
    # if res:
        # print(res)
    fpars = ipanema.Parameters.clone(res.params)
    # else:
    #     #This is to at least have the plot
    #     print("Could not fit it!. Cloning pars to res")
    #     fpars = ipanema.Parameters.clone(pars)
    #     print(fpars)

    fig, axplot, axpull = complot.axes_plotpull()
    hdata = complot.hist(
        ristra.get(rd.mass), weights=rd.df.eval(mass_weight), bins=60, density=False
    )
    axplot.errorbar(
        hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr, fmt=".k"
    )

    mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
    signal = 0 * mass

    # plot signal: nbkg -> 0 and nexp -> 0
    _p = ipanema.Parameters.clone(fpars)
    if "fsigBu" in _p:
        _p["fsigBu"].set(value=0, min=-np.inf, max=np.inf)
        _p["fexp"].set(value=1-fpars["fsigBu"].value, min=-np.inf, max=np.inf)
    
    #Plot fcomb
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm)
    )

    axplot.plot(_x, _y, color="C2", label=rf"Combinatorial")

    if "fexp" in _p:
        _p["fexp"].set(value=0, min=-np.inf, max=np.inf)
        _p["fsigBu"].set(value=fpars["fsigBu"].value, min=-np.inf, max=np.inf)
    
    #Plot fBu
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm)
    )
    # axplot.plot(_x, _y, color="C3", label=rf"$B^+$ {model}")
    axplot.plot(_x, _y, color="C3", label=rf"Signal {model}")

    # plot fit with all components and data
    _p = ipanema.Parameters.clone(fpars)
    x, y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm)
    )
    axplot.set_ylim(0., 1.1*np.max(y))
    axplot.plot(x, y, color="C0", label=rf"Full Fit")
    axpull.fill_between(
        hdata.bins,
        complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts, *hdata.yerr),
        0,
        facecolor="C0",
        alpha=0.5,
    )
    axpull.set_xlabel(r"$m(J/\psi K^+)$ [MeV/$c^2$]")
    axpull.set_ylim(-6.5, 6.5)
    axpull.set_yticks([-5, 0, 5])
    axplot.set_ylabel("Candidates")
    axplot.legend(loc="upper left")
    if figs:
        os.makedirs(figs, exist_ok=True)
        fig.savefig(os.path.join(figs, f"fit.pdf"))

    axplot.set_yscale("log")
    axplot.set_ylim(1e0, 10 * np.max(y))
    if figs:
        fig.savefig(os.path.join(figs, f"logfit.pdf"))

    plt.close()

    # }}}

    # compute sWeights if asked {{{

    if sweights:
        # separate parameters in yields and shape parameters
        _yields = ipanema.Parameters.find(fpars, "fsig.*") + ["fexp"]
        _pars = list(fpars)
        [_pars.remove(_y) for _y in _yields]
        _yields = ipanema.Parameters.build(fpars, _yields)
        _pars = ipanema.Parameters.build(fpars, _pars)

        sw = splot.compute_sweights(
            lambda *x, **y: pdf(rd.mass, rd.pdf, *x, **y), _pars, _yields
        )
        if tagger:
          sw = dict((key+f"{tagger}", value) for (key, value) in sw.items())

        for k, v in sw.items():
            _sw = np.copy(_proxy)
            _sw[list(rd.df.index)] = v * np.float64(rd.df.eval(mass_weight))
            sw[k] = _sw
        print(sw)
        return (fpars, sw)

    # }}}

    return (fpars, False)


# }}}


# command-line interface {{{

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="mass fit")
    p.add_argument("--sample")
    p.add_argument("--input-params", default=False)
    p.add_argument("--output-params")
    p.add_argument("--output-figures")
    p.add_argument("--mass-model", default="ipatia")
    p.add_argument("--mass-branch", default="B_ConstJpsi_M")
    p.add_argument("--mass-weight")
    p.add_argument("--mass-bin", default=False)
    p.add_argument("--tagger", default=False)
    p.add_argument("--trigger", default="combined")
    p.add_argument("--sweights")
    p.add_argument("--mode")
    args = vars(p.parse_args())

    if args["sweights"]:
        sweights = True
    else:
        sweights = False

    if args["input_params"]:
        input_pars = ipanema.Parameters.load(args["input_params"])
    else:
        input_pars = False
    
    branches = [args["mass_branch"]]

    branches += ["hlt1b"]


    if args["mass_weight"]:
        mass_weight = args["mass_weight"]
        branches += [mass_weight]
    else:
        mass_weight = f"{branches[0]}/{branches[0]}"

    cut = False
    if "prefit" in args["output_params"]:
      if "Bu" in args["mode"]:
        cut = "(B_BKGCAT == 0 | B_BKGCAT == 10 | B_BKGCAT == 50)"
        branches += ["B_BKGCAT"]
    
    #In case we want to do a cut for each of the taggers
    # tagger = args["tagger"] 
    # if tagger:
    #   with open('config/tagger.yaml') as config:
    #     config = yaml.load(config, Loader=yaml.FullLoader)
    #   q_b = config[tagger]['branch']['decision']
    #   eta_b = config[tagger]['branch']['eta']
    #   if not cut:
    #     cut =  f" ({eta_b} >= 0 & {eta_b} < 0.5 & {q_b} != 0)"
    #     # cut =  f" ({q_b} != 0)" #Warning Sevda try!
    #   else:
    #     cut += f" & ({eta_b} >= 0 & {eta_b} < 0.5 & {q_b} != 0)"
    #     # cut += f" & ({q_b} != 0)" #Warning Sevda try!
    #   branches += [f"{q_b}", f"{eta_b}"]


    sample = Sample.from_root(args["sample"], branches=branches)#, entrystop=100000)
    print("Branches:", branches)
    print(sample)

    pars, sw = mass_fitter(
        sample.df,
        mass_range=False,
        mass_branch=args["mass_branch"],
        mass_weight=mass_weight,
        trigger=args["trigger"],
        mode = args["mode"],
        figs=args["output_figures"],
        cut=cut,
        model=args["mass_model"],
        sweights=sweights,
        input_pars=input_pars,
        tagger=args["tagger"],
        verbose=True,

    )
    pars.dump(args["output_params"])
    if sweights:
      if "root" in args["sweights"]:
        sample = Sample.from_root(args["sample"], flatten=None)
        _proxy = np.float64(sample.df['time']) * 0.0
        for w in ["fsigBu", "fexp"]: #TODO: fix
          _weight = sw
          _proxy += _weight[w]
          sample.df[f"{w[1:]}SW"] = _proxy
        # if "Bs" in args["mode"]:
          # sample.df.rename(columns={"sigBuSW" : "sigBsSW"}, inplace=True)
        with uproot.recreate(args['sweights']) as f:
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
      else:
        np.save(args["sweights"], sw)

# }}}
