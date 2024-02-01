# create_time_pars
#
#

__all__ = []
__author__ = ["Ramon Angel Ruiz Fernandez"]
__email__ = ["rruizfer@cern.ch"]


import ipanema

import numpy as np
import uncertainties
from uncertainties import ufloat

modes = ["Bs2DsPi", "Bd2JpsiKstar"]

offset = ["nobias", "biasDsPi", "biasJpsiPhi"]
years = ["2015", "2016", "2017", "2018"]

mu_DsPi = {
	    "2015": ufloat(-2.254e-3, 0.109e-3),
	    "2016": ufloat(-2.254e-3, 0.109e-3),
	    "201516": ufloat(-2.254e-3, 0.109e-3),
      "2017": ufloat(-3.047e-3, 0.112e-3),
      "2018":  ufloat(-2.394e-3, 0.107e-3)
    }

mu_JpsiPhi = {
	    "2015": ufloat(-3.90e-3, 0.10e-3),
	    "2016": ufloat(-5.12e-3, 0.04e-3),
	    "201516": ufloat(-4.96e-3, 0.11e-3), #From N2/N1 = (\sigma_1/\sigma_2)**2 ponderated
	    "2017": ufloat(-6.09e-3, 0.04e-3),
	    "2018": ufloat(-4.88e-3, 0.04e-3),
    }



time_params = {

            "Bs" : {
                    "DM" : ufloat(17.766, 0.0051),
                    "DG" : ufloat(0.085, 0.004),
                    "tau" : ufloat(1.520, 0.005),
                },
            "Bd" : {
                    "DM" : ufloat(0.5065, 0.0019),
                    "tau" : ufloat(1.519, 0.004),
                    }
}



for m in modes:
  for y in years:
    for off in offset:
      op = f"output/params/time_params/{y}/{m}/time_pars_{off}.json"
      pars = ipanema.Parameters()
      if "no" in off:
        pars.add(dict(name="mu", value=0., free=False))
			#TODO: Numbers for Bd? for the moment same as in Bs
      elif "DsPi" in off:
        pars.add(
					       dict(name="mu", value=mu_DsPi[y].n, stdev=mu_DsPi[y].s, free=False)
				)
      elif "JpsiPhi" in off:
        pars.add(
					      dict(name="mu", value=mu_JpsiPhi[y].n, stdev=mu_JpsiPhi[y].s, free=False)
				        )

      for k in time_params[m[0:2]].keys():
        pars.add(
                dict(name=k, value=time_params[f"{m[0:2]}"][k].n, 
										 stdev=time_params[f"{m[0:2]}"][k].s, 
					           min = 0.8*time_params[f"{m[0:2]}"][k].n,
					          max = 1.2*time_params[f"{m[0:2]}"][k].n,
					          free=True)
				        )
      if "Bd" in m:
        pars.add(
                dict(name="DG", value=0., free=False)
				        )
      pars.add(
					      dict(name="G", 
					          value = 1./pars["tau"].uvalue.n,
					           stdev = 1./pars["tau"].uvalue.s)
			)
      pars.dump(op)








# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
