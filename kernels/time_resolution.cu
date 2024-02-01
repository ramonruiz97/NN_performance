////////////////////////////////////////////////////////
////        Author: Ramón Ángel Ruiz Fernandez      ////
///////////////////////////////////////////////////////


#define USE_DOUBLE 1
#include <exposed/kernels.ocl>


//Gauss Normalized
WITHIN_KERNEL
ftype gauss_timeres(const ftype x,
                    const ftype mu, const ftype sigma,
                    const ftype xLL, const ftype xUL)
{
  ftype num = exp(-0.5*pow(mu-x,2)/rpow(sigma,2));
  ftype den = 0;
  den += erf((mu-xLL)/(sqrt(2.)*sigma));
  den -= erf((mu-xUL)/(sqrt(2.)*sigma));
  den *= sqrt(M_PI/2.)*sigma;

  return num/den;
}

WITHIN_KERNEL
ftype wrong_pv_component(const ftype x,
                         const ftype tau1, 
                         const ftype xmin, const ftype xmax) {

  ftype num = exp(-fabs(x) / tau1) /
               (2 - exp(-(xmax / tau1)) - exp(xmin / tau1)) / tau1;

  return num;
}



//Exponential 0 -> inf convolved with a gaussian -> Normalized
WITHIN_KERNEL
ftype expposconv(const ftype time,
                 const ftype mu, const ftype sigma, const ftype tau,
                const ftype xLL, const ftype xUL)
{
  ftype num = 0.;
  num += exp((pow(sigma, 2)-2*time*tau+2*mu*tau)/(2*pow(tau, 2)));
  num *= erfc((pow(sigma, 2)+(mu-time)*tau)/(sqrt(2.)*sigma*tau));
  num *= sqrt(M_PI/2.)*sigma;

  ftype den = 0;
  den += exp(xUL/tau)*erfc((pow(sigma,2)-xLL*tau+mu*tau)/(sqrt(2.)*sigma*tau)) - exp(xLL/tau)*erfc((pow(sigma,2)-xUL*tau+mu*tau)/(sqrt(2.)*sigma*tau));
  den *= exp((pow(sigma, 2)-2*(xLL+xUL-mu)*tau)/(2*pow(tau,2)));
  den += erf((xUL-mu)/(sqrt(2.)*sigma));
  den -= erf((xLL-mu)/(sqrt(2.)*sigma));
  den *= tau*sqrt(M_PI/2.)*sigma;


  return num/den;
}




//Exponential Absolute time -Inf -> inf convolved with a gaussian -> Normalized
WITHIN_KERNEL
ftype expabsconv(const ftype time,
                 const ftype mu, const ftype sigma, const ftype tau,
                 const ftype xLL, const ftype xUL)
{
  ftype num = 0.;
  num += exp(2.*time/tau)*erfc((pow(sigma,2)+(time-mu)*tau)/(sqrt(2.)*sigma*tau));
  num += exp(2.*mu/tau)*erfc((pow(sigma,2)+(-time+mu)*tau)/(sqrt(2.)*sigma*tau));
  num *= sqrt(M_PI/2.)*sigma*exp((pow(sigma,2)-2.*(time+mu)*tau)/(2.*pow(tau,2)));

  ftype den = 0;
  den -= exp((2*xLL+xUL)/tau)*erfc((pow(sigma, 2)+ (xLL - mu)*tau)/(sqrt(2.)*sigma*tau));
  den += exp((xLL+2*xUL)/tau)*erfc((pow(sigma, 2)+ (xUL - mu)*tau)/(sqrt(2.)*sigma*tau));
  den += exp((xUL+2*mu)/tau)*erfc((pow(sigma, 2)+ (-xLL + mu)*tau)/(sqrt(2.)*sigma*tau));
  den -= exp((xLL+2*mu)/tau)*erfc((pow(sigma, 2)+ (-xUL + mu)*tau)/(sqrt(2.)*sigma*tau));
  den *= sqrt(M_PI/2.)*sigma*tau*exp((pow(sigma, 2)-2*(xLL+xUL+mu)*tau)/(2.*pow(tau,2)));
  den += sqrt(2.*M_PI)*sigma*tau*(-erf((xLL-mu)/(sqrt(2.)*sigma)) + erf((xUL - mu)/(sqrt(2.)*sigma)));

  /* printf("'num'= %+.4f 'den'= %+.8f \n", num , den); */

  return num/den;
}

WITHIN_KERNEL
ftype time_fit(const ftype time, 
               const ftype fsig, 
               const ftype fg1, const ftype fg2, const ftype fg3,
               const ftype mu,
               const ftype sigma1, const ftype sigma2, const ftype sigma3,
               const ftype fexp,
               const ftype fbkg1, const ftype fbkg2,
               const ftype tau1, const ftype tau2,
               const ftype tLL, const ftype tUL)
{
  ftype signal = 0;
  signal += fg1*gauss_timeres(time, mu, sigma1, tLL, tUL);
  signal += fg2*gauss_timeres(time, mu, sigma2, tLL, tUL);
  signal += fg3*gauss_timeres(time, mu, sigma3, tLL, tUL);

  ftype bkg = 0;
  
  //WPV
  bkg += fbkg1*fg1*expabsconv(time, mu, sigma1, tau1, tLL, tUL);
  bkg += fbkg1*fg2*expabsconv(time, mu, sigma2, tau1, tLL, tUL);
  bkg += fbkg1*fg3*expabsconv(time, mu, sigma3, tau1, tLL, tUL);
  
  //RD
  bkg += fbkg2*fg1*expposconv(time, mu, sigma1, tau2, tLL, tUL);
  bkg += fbkg2*fg2*expposconv(time, mu, sigma2, tau2, tLL, tUL);
  bkg += fbkg2*fg3*expposconv(time, mu, sigma3, tau2, tLL, tUL);

  
  ftype num = (fsig*signal + fexp*bkg)/(fsig + fexp);
  /* printf("'num'=%.4f\n", num);  */

  return num;
}

WITHIN_KERNEL
ftype time_fit_unbined(const ftype time, const ftype sigmat,
               const ftype fsig, 
               const ftype fg1, const ftype fg2, const ftype fg3,
               const ftype mu,
               const ftype a1, const ftype a2, const ftype a3,
               const ftype b1, const ftype b2, const ftype b3,
               const ftype fexp,
               const ftype fbkg1, const ftype fbkg2,
               const ftype tau1, const ftype tau2,
               const ftype tLL, const ftype tUL)
{
  ftype signal = 0;
  const ftype sigma1 = a1 + b1*sigmat;
  const ftype sigma2 = a2 + b2*sigmat;
  const ftype sigma3 = a3 + b3*sigmat;
  signal += fg1*gauss_timeres(time, mu, sigma1, tLL, tUL);
  signal += fg2*gauss_timeres(time, mu, sigma2, tLL, tUL);
  signal += fg3*gauss_timeres(time, mu, sigma3, tLL, tUL);

  ftype bkg = 0;
  
  //WPV
  bkg += fbkg1*fg1*expabsconv(time, mu, sigma1, tau1, tLL, tUL);
  bkg += fbkg1*fg2*expabsconv(time, mu, sigma2, tau1, tLL, tUL);
  bkg += fbkg1*fg3*expabsconv(time, mu, sigma3, tau1, tLL, tUL);
  
  //RD
  bkg += fbkg2*fg1*expposconv(time, mu, sigma1, tau2, tLL, tUL);
  bkg += fbkg2*fg2*expposconv(time, mu, sigma2, tau2, tLL, tUL);
  bkg += fbkg2*fg3*expposconv(time, mu, sigma3, tau2, tLL, tUL);

  
  ftype num = (fsig*signal + fexp*bkg)/(fsig + fexp);
  /* printf("'num'=%.4f\n", num);  */

  return num;
}


WITHIN_KERNEL
ftype time_fit_Bd(const ftype time, 
               const ftype fsig, 
               const ftype fg1, const ftype fg2, const ftype fg3,
               const ftype mu,
               const ftype sigma1, const ftype sigma2, const ftype sigma3,
               const ftype fexp,
               const ftype fbkg1, const ftype fbkg2,
               const ftype tau1, const ftype tau2,
               const ftype tLL, const ftype tUL)
{
  ftype signal = 0;
  signal += fg1*gauss_timeres(time, mu, sigma1, tLL, tUL);
  signal += fg2*gauss_timeres(time, mu, sigma2, tLL, tUL);
  signal += fg3*gauss_timeres(time, mu, sigma3, tLL, tUL);

  ftype bkg = 0;
  
  //WPV
  bkg += fbkg1*wrong_pv_component(time, tau1, tLL, tUL);
  
  //RD
  bkg += fbkg2*fg1*expposconv(time, mu, sigma1, tau2, tLL, tUL);
  bkg += fbkg2*fg2*expposconv(time, mu, sigma2, tau2, tLL, tUL);
  bkg += fbkg2*fg3*expposconv(time, mu, sigma3, tau2, tLL, tUL);

  
  ftype num = (fsig*signal + fexp*bkg)/(fsig + fexp);

  return num;
}

KERNEL
void kernel_time_fit(GLOBAL_MEM ftype * prob, 
                     GLOBAL_MEM const ftype *time,
                     const ftype fsig, 
                     const ftype fg1,  const ftype fg2, const ftype fg3,  
                     const ftype mu,
                     const ftype sigma1, const ftype sigma2,  const ftype sigma3,
                     const ftype fexp,
                     const ftype fbkg1, const ftype fbkg2,
                     const ftype tau1, const ftype tau2,
                     const ftype tLL, const ftype tUL)
{
    const int idx = get_global_id(0);
    /* if (idx==0){ */
    /* printf("'fsig'= %+.4f  'fg' =%+.4f\n", fsig, fg); */
    /* printf("'mu'= %+.4f 'sigma1' =%+.4f 'sigma2' =%+.4f\n", mu, sigma1, sigma2); */ /* printf("'fexp'= %+.4f 'tau1' =%+.4f 'tau2' =%+.4f\n", fexp, tau1, tau2); */
    /* printf("'tLL'= %+.4f 'tUL' =%+.4f\n", tLL, tUL); */
    /* } */
    prob[idx] = time_fit(time[idx],  
                        fsig, 
                        fg1, fg2, fg3,
                        mu, 
                        sigma1, sigma2, sigma3,
                        fexp,
                        fbkg1, fbkg2,
                        tau1, tau2,
                        tLL, tUL);
}

KERNEL
void kernel_time_fit_Bd(GLOBAL_MEM ftype * prob, 
                     GLOBAL_MEM const ftype *time,
                     const ftype fsig, 
                     const ftype fg1,  const ftype fg2, const ftype fg3,  
                     const ftype mu,
                     const ftype sigma1, const ftype sigma2,  const ftype sigma3,
                     const ftype fexp,
                     const ftype fbkg1, const ftype fbkg2,
                     const ftype tau1, const ftype tau2,
                     const ftype tLL, const ftype tUL)
{
    const int idx = get_global_id(0);
    /* if (idx==0){ */
    /* printf("'fsig'= %+.4f  'fg' =%+.4f\n", fsig, fg); */
    /* printf("'mu'= %+.4f 'sigma1' =%+.4f 'sigma2' =%+.4f\n", mu, sigma1, sigma2); */ /* printf("'fexp'= %+.4f 'tau1' =%+.4f 'tau2' =%+.4f\n", fexp, tau1, tau2); */
    /* printf("'tLL'= %+.4f 'tUL' =%+.4f\n", tLL, tUL); */
    /* } */
    prob[idx] = time_fit_Bd(time[idx],  
                        fsig, 
                        fg1, fg2, fg3,
                        mu, 
                        sigma1, sigma2, sigma3,
                        fexp,
                        fbkg1, fbkg2,
                        tau1, tau2,
                        tLL, tUL);
}

KERNEL
void kernel_time_fit_unbined(GLOBAL_MEM ftype * prob, 
                     GLOBAL_MEM const ftype *time, 
                     GLOBAL_MEM const ftype *sigmat,
                     const ftype fsig, 
                     const ftype fg1,  const ftype fg2, const ftype fg3,  
                     const ftype mu,
                     const ftype a1, const ftype a2,  const ftype a3,
                     const ftype b1, const ftype b2,  const ftype b3,
                     const ftype fexp,
                     const ftype fbkg1, const ftype fbkg2,
                     const ftype tau1, const ftype tau2,
                     const ftype tLL, const ftype tUL)
{
    const int idx = get_global_id(0);
    
    /* if (idx==0){ */
    /* printf("'fsig'= %+.4f  'fg' =%+.4f\n", fsig, fg); */
    /* printf("'mu'= %+.4f 'sigma1' =%+.4f 'sigma2' =%+.4f\n", mu, sigma1, sigma2); */ /* printf("'fexp'= %+.4f 'tau1' =%+.4f 'tau2' =%+.4f\n", fexp, tau1, tau2); */
    /* printf("'tLL'= %+.4f 'tUL' =%+.4f\n", tLL, tUL); */
    /* } */
    prob[idx] = time_fit_unbined(time[idx], sigmat[idx],
                        fsig, 
                        fg1, fg2, fg3,
                        mu, 
                        a1, a2, a3,
                        b1, b2, b3,
                        fexp,
                        fbkg1, fbkg2,
                        tau1, tau2,
                        tLL, tUL);
}

