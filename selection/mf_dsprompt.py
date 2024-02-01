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


def ipatia_exponential(mass, signal, fsigDs=0, fexp=0,
    muDs=0, sigmaDs=10, lambd=0, zeta=0, beta=0,
    aL=0, nL=0, aR=0, nR=0, b=0, norm=1, mLL=None, mUL=None,
    ):
    mLL, mUL = ristra.min(mass), ristra.max(mass)
    # ipatia
    prog.py_ipatia( signal, mass, np.float64(muDs), np.float64(sigmaDs),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
    )
    pdfBs = 1.0 * signal.get()
    # normalize
    _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
    _y = _x * 0
    prog.py_ipatia( _y, _x, np.float64(muDs), np.float64(sigmaDs),
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
    ans = fsigDs * (pdfBs / nBs) + fexp * pComb
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
             dict(name=f"fsigDs", value=0.95, min=0.1, max=1, free=True, latex=r"N_{B_d}")
          )
    pars.add(dict(name="muDs", value=1969, min=1950, max=1989, latex=r"\mu_{B_d}"))

    pars.add(
            dict(
                name="sigmaDs",
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
      pars.remove("fsigDs", "muDs", "sigmaDs")
      _pars.unlock("fsigDs", "muDs", "sigmaDs", "b")
      pars = pars + _pars
    else:
      #Prefit stage
      pars["fsigDs"].value = 1
      pars["fsigDs"].free = False
      # Hypatia tails {{{
      pars.add(
          dict(
              name="lambd",
              value=-2.8,
              min=-5,
              max=-1.05,
              free=True,
              latex=r"\lambda",
          )
      )
      pars.add(dict(name="zeta", value=1e-5, min =-1e-4, max= 1e-4, free=False, latex=r"\zeta"))
      pars.add(dict(name="beta", value=0.0, min =-1.,  max=1., free=False, latex=r"\beta"))
      pars.add(
            dict(name="aL", value=2., min=0.2, max=5.5, free=True, latex=r"a_l")
              )
      pars.add(dict(name="nL", value=2., min=0.01, max=6, free=True, latex=r"n_l"))
      pars.add(
          dict(name="aR", value=2, min=0.2, max=5.5, free=True, latex=r"a_r")
      )
      pars.add(dict(name="nR", value=2, min=0.01, max=6, free=True, latex=r"n_r"))
      # }}}

      #Combinatorial:
      pars.add(dict(name='b', value=-0.05, min=-1, max=1, free=False, latex=r'b'))
      pars.add(dict(name='fexp', formula="1-fsigDs", latex=r'f_{comb}'))


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
    print(f"After cut: {rd.shape[0]}")
    # exit()
    rd.allocate(mass=mass_branch, pdf=f"0*{mass_branch}", weight=mass_weight)

    # }}}

    # perform the fit {{{
    try: 
      res = ipanema.optimize(
        fcn,
        pars,
        fcn_kwgs={"data": rd},
        method="minuit",
        verbose=True,
        strategy=2,
        tol=0.1,
        )
    except:
      res = False
    # if res:
        # print(res)
    if res: 
      fpars = ipanema.Parameters.clone(res.params)

    else:
      print("Lets do another try......!")

      if not input_pars:
        pars['lambd'].set(value=-1.5, free=True)

      res = ipanema.optimize(
          fcn,
          pars,
          fcn_kwgs={"data": rd},
          method="minuit",
          verbose=True,
          strategy=1,
          tol=0.1,
          )
      if res:
        fpars = ipanema.Parameters.clone(res.params)
      else:
        print("WARNING IT didnt convergeeeeed")
        exit()
        fpars = ipanema.Parameters.clone(pars)


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
    if "fsigDs" in _p:
        _p["fsigDs"].set(value=0, min=-np.inf, max=np.inf)
        _p["fexp"].set(value=1-fpars["fsigDs"].value, min=-np.inf, max=np.inf)
    
    #Plot fcomb
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm)
    )

    axplot.plot(_x, _y, color="C2", label=rf"Combinatorial")

    if "fexp" in _p:
        _p["fexp"].set(value=0, min=-np.inf, max=np.inf)
        _p["fsigDs"].set(value=fpars["fsigDs"].value, min=-np.inf, max=np.inf)
    
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
    
    axplot.set_title(r"Prompt $D_s^-$ ($\phi\pi$)")
    axpull.set_xlabel(r"$m(K^- K^+ \pi^-)$ [MeV/$c^2$]")
    axpull.set_ylim(-6.5, 6.5)
    axpull.set_yticks([-5, 0, 5])
    axplot.set_ylabel("Candidates")
    axplot.legend(loc="upper left")
    type = "fit"
    if "prefit" in args["output_params"]:
      type = "prefit"
      
    if figs:
        os.makedirs(figs, exist_ok=True)
        fig.savefig(os.path.join(figs, f"{year}_{type}_fit.pdf"))

    axplot.set_yscale("log")
    axplot.set_ylim(1e0, 10 * np.max(y))
    if figs:
        fig.savefig(os.path.join(figs, f"{year}_{type}_logfit.pdf"))

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
    p.add_argument("--year")
    args = vars(p.parse_args())

    mc_truth = [
  "abs(lab2_TRUEID)==431", # Ds
  "&&abs(lab1_TRUEID)==211" # pi
  "&abs(lab2_MC_MOTHER_ID)!=531", # no Bs
  "&abs(lab2_MC_MOTHER_ID)!=511", # no B+
  "&abs(lab2_MC_MOTHER_ID)!=521",  # no B0
  "&abs(lab2_MC_MOTHER_ID)!=541", # no Bc
  "&abs(lab2_MC_MOTHER_ID)<1000", # no L's
  "&abs(lab2_MC_GD_MOTHER_ID)!=531", # no Bs
  "&abs(lab2_MC_GD_MOTHER_ID)!=511", # no B+
  "&abs(lab2_MC_GD_MOTHER_ID)!=521", # no B0
  "&abs(lab2_MC_GD_MOTHER_ID)!=541", # no Bc
  "&abs(lab2_MC_GD_MOTHER_ID)<1000", # no L's
  "&abs(lab3_TRUEID)==321", # K
  "&abs(lab4_TRUEID)==321", # K
  "&abs(lab5_TRUEID)==211", # pi
  "&abs(lab3_MC_MOTHER_ID)==431", # Ds
  "&abs(lab4_MC_MOTHER_ID)==431", # Ds
  "&abs(lab5_MC_MOTHER_ID)==431", # Ds
  "&abs(lab3_MC_GD_MOTHER_ID)!=531", # no Bs
  "&abs(lab4_MC_GD_MOTHER_ID)!=531",
  "&abs(lab5_MC_GD_MOTHER_ID)!=531",
  "&abs(lab3_MC_GD_MOTHER_ID)!=511", # no B+
  "&abs(lab4_MC_GD_MOTHER_ID)!=511",
  "&abs(lab5_MC_GD_MOTHER_ID)!=511",
  "&abs(lab3_MC_GD_MOTHER_ID)!=521",
  "&abs(lab4_MC_GD_MOTHER_ID)!=521",
  "&abs(lab5_MC_GD_MOTHER_ID)!=521",
  "&abs(lab3_MC_GD_MOTHER_ID)!=541",
  "&abs(lab4_MC_GD_MOTHER_ID)!=541",
  "&abs(lab5_MC_GD_MOTHER_ID)!=541",
  "&abs(lab3_MC_GD_MOTHER_ID)<1000",
  "&abs(lab4_MC_GD_MOTHER_ID)<1000",
  "&abs(lab5_MC_GD_MOTHER_ID)<1000"]

    mc_truth_promptno = [
  "abs(lab2_TRUEID)==431", # Ds
  # "&&abs(lab1_TRUEID)==211" # pi
  # "&abs(lab2_MC_MOTHER_ID)!=531", # no Bs
  "&abs(lab2_MC_MOTHER_ID)!=511", # no B+
  "&abs(lab2_MC_MOTHER_ID)!=521",  # no B0
  "&abs(lab2_MC_MOTHER_ID)!=541", # no Bc
  "&abs(lab2_MC_MOTHER_ID)<1000", # no L's
  # "&abs(lab2_MC_GD_MOTHER_ID)!=531", # no Bs
  "&abs(lab2_MC_GD_MOTHER_ID)!=511", # no B+
  "&abs(lab2_MC_GD_MOTHER_ID)!=521", # no B0
  "&abs(lab2_MC_GD_MOTHER_ID)!=541", # no Bc
  "&abs(lab2_MC_GD_MOTHER_ID)<1000", # no L's
  "&abs(lab3_TRUEID)==321", # K
  "&abs(lab4_TRUEID)==321", # K
  "&abs(lab5_TRUEID)==211", # pi
  "&abs(lab3_MC_MOTHER_ID)==431", # Ds
  "&abs(lab4_MC_MOTHER_ID)==431", # Ds
  "&abs(lab5_MC_MOTHER_ID)==431", # Ds
  # "&abs(lab3_MC_GD_MOTHER_ID)!=531", # no Bs
  # "&abs(lab4_MC_GD_MOTHER_ID)!=531", 
  # "&abs(lab5_MC_GD_MOTHER_ID)!=531", 
  "&abs(lab3_MC_GD_MOTHER_ID)!=511", # no B+
  "&abs(lab4_MC_GD_MOTHER_ID)!=511",
  "&abs(lab5_MC_GD_MOTHER_ID)!=511",
  "&abs(lab3_MC_GD_MOTHER_ID)!=521",
  "&abs(lab4_MC_GD_MOTHER_ID)!=521",
  "&abs(lab5_MC_GD_MOTHER_ID)!=521",
  "&abs(lab3_MC_GD_MOTHER_ID)!=541",
  "&abs(lab4_MC_GD_MOTHER_ID)!=541",
  "&abs(lab5_MC_GD_MOTHER_ID)!=541",
  "&abs(lab3_MC_GD_MOTHER_ID)<1000",
  "&abs(lab4_MC_GD_MOTHER_ID)<1000",
  "&abs(lab5_MC_GD_MOTHER_ID)<1000"]

    if args["sweights"]:
        sweights = True
    else:
        sweights = False

    if args["input_params"]:
        input_pars = ipanema.Parameters.load(args["input_params"])
    else:
        input_pars = False
    
    branches = [args["mass_branch"]]
    year = args["year"]




    if args["mass_weight"]:
        mass_weight = args["mass_weight"]
        branches += [mass_weight]
    else:
        mass_weight = f"{branches[0]}/{branches[0]}"

    cut = False
    mode = args["mode"]
    if "prefit" in args["output_params"]:
      if "Promptno" in mode:
        cut = "(B_BKGCAT <= 30 | B_BKGCAT ==50)"
        branches += ["B_BKGCAT"]
      else:
        cut = " ".join(truth.replace("&", " & ") for truth in mc_truth)
        branches += ["lab2_TRUEID", "lab2_MC_MOTHER_ID", "lab2_MC_GD_MOTHER_ID", "lab3_TRUEID", "lab4_TRUEID", "lab5_TRUEID", "lab3_MC_MOTHER_ID", "lab4_MC_MOTHER_ID", "lab5_MC_MOTHER_ID", "lab3_MC_GD_MOTHER_ID", "lab4_MC_GD_MOTHER_ID", "lab5_MC_GD_MOTHER_ID"]


    sample = Sample.from_root(args["sample"], branches=branches)#, entrystop=100000)
    print("Branches:", branches)
    print(sample)

    pars, sw = mass_fitter(
        sample.df,
        mass_range=False,
        mass_branch=args["mass_branch"],
        mass_weight=mass_weight,
        trigger=args["trigger"],
        mode = mode,
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
        for w in ["fsigDs", "fexp"]: #TODO: fix
          _weight = sw
          _proxy += _weight[w]
          sample.df[f"{w[1:]}SW"] = _proxy
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
