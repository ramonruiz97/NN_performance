//AUthor Marcos Romero Lamas
//       Ramón Ángel Ruiz Fernandez

#define USE_DOUBLE 1
#include <exposed/kernels.ocl>
#define SIGMA_THRESHOLD 5.0
#define DEBUG 1
#define DEBUG_EVT 4

/* #define NKNOTS 4 */
/* #define SPL_BINS 4 */
/* #define NTIMEBINS 5 */
#define NKNOTS 7
#define SPL_BINS 7
#define NTIMEBINS 8


/* const CONSTANT_MEM ftype KNOTS[4] = {0.3, 0.91, 1.96, 9.0}; */
const CONSTANT_MEM ftype KNOTS[7] = {0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 12.};

//TIME Acceptance kernels from phis-scq -> Author Marcos {{{
WITHIN_KERNEL
unsigned int getTimeBin(ftype const t)
{
  int _i = 0;
  int _n = NKNOTS-1;
  while(_i <= _n )
  {
    if( t < KNOTS[_i] ) {break;}
    _i++;
  }
  if ((0 == _i) & (DEBUG > 3)) {
    printf("WARNING: t=%.16f below first knot!\n",t);
  }
  return _i - 1;

}



WITHIN_KERNEL
ctype expconv(ftype t, ftype G, ftype omega, ftype sigma)
{
  const ftype sigma2 = sigma*sigma;
  const ftype omega2 = omega*omega;

  if( t > SIGMA_THRESHOLD*sigma )
  {
    ftype a = exp(-G*t+0.5*G*G*sigma2-0.5*omega2*sigma2);
    ftype b = omega*(t-G*sigma2);
    return C(a*cos(b),a*sin(b));
  }
  else
  {
    ctype z, fad;
    z   = C(-omega*sigma2/(sigma*sqrt(2.)), -(t-sigma2*G)/(sigma*sqrt(2.)));
    fad = cwofz(z);
    return cmul( C(fad.x,-fad.y), C(0.5*exp(-0.5*t*t/sigma2),0) );
  }
}

WITHIN_KERNEL
ftype getCoeff(GLOBAL_MEM const ftype *mat, int const r, int const c)
{
  return mat[4*r+c];
}




WITHIN_KERNEL
ftype time_efficiency(const ftype t, GLOBAL_MEM const ftype *coeffs,
    const ftype tLL, const ftype tUL)
{
  int bin   = getTimeBin(t);
  ftype c0 = getCoeff(coeffs,bin,0);
  ftype c1 = getCoeff(coeffs,bin,1);
  ftype c2 = getCoeff(coeffs,bin,2);
  ftype c3 = getCoeff(coeffs,bin,3);
  #if DEBUG
  if (DEBUG >= 3 && ( get_global_id(0) == DEBUG_EVT))
  {
    printf("\nTIME ACC           : t=%.8f\tbin=%d\tc=[%+f\t%+f\t%+f\t%+f]\tdta=%+.8f\n",
        t,bin,c0,c1,c2,c3, (c0 + t*(c1 + t*(c2 + t*c3))) );
  }
  #endif

  return (c0 + t*(c1 + t*(c2 + t*c3)));
}

WITHIN_KERNEL
ftype china_eff(const ftype t, 
                const ftype a, 
                const ftype b, 
                const ftype n)
{
  ftype den = 1. + rpow(a*t, n) - b;
  ftype corr = 1./den;
  return 1. - corr;
}


WITHIN_KERNEL ctype faddeeva( ctype z)
{
   ftype in_real = z.x;
   ftype in_imag = z.y;
   int n, nc, nu;
   ftype h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, x, yh, y;
   ftype Rx [33];
   ftype Ry [33];

   x = fabs(in_real);
   y = fabs(in_imag);

   if (y < YLIM && x < XLIM) {
      q = (1.0 - y / YLIM) * sqrt(1.0 - (x / XLIM) * (x / XLIM));
      h  = 1.0 / (3.2 * q);
      #ifdef CUDA
        nc = 7 + int(23.0 * q);
      #else
        nc = 7 + convert_int(23.0 * q);
      #endif

//       xl = pow(h, ftype(1 - nc));
      ftype h_inv = 1./h;
      xl = h_inv;
      for(int i = 1; i < nc-1; i++)
          xl *= h_inv;

      xh = y + 0.5 / h;
      yh = x;
      #ifdef CUDA
        nu = 10 + int(21.0 * q);
      #else
        nu = 10 + convert_int(21.0 * q);
      #endif
      Rx[nu] = 0.;
      Ry[nu] = 0.;
      for (n = nu; n > 0; n--){
         Tx = xh + n * Rx[n];
         Ty = yh - n * Ry[n];
         Tn = Tx*Tx + Ty*Ty;
         Rx[n-1] = 0.5 * Tx / Tn;
         Ry[n-1] = 0.5 * Ty / Tn;
         }
      Sx = 0.;
      Sy = 0.;
      for (n = nc; n>0; n--){
         Saux = Sx + xl;
         Sx = Rx[n-1] * Saux - Ry[n-1] * Sy;
         Sy = Rx[n-1] * Sy + Ry[n-1] * Saux;
         xl = h * xl;
      };
      Wx = ERRF_CONST * Sx;
      Wy = ERRF_CONST * Sy;
   }
   else {
      xh = y;
      yh = x;
      Rx[0] = 0.;
      Ry[0] = 0.;
      for (n = 9; n>0; n--){
         Tx = xh + n * Rx[0];
         Ty = yh - n * Ry[0];
         Tn = Tx * Tx + Ty * Ty;
         Rx[0] = 0.5 * Tx / Tn;
         Ry[0] = 0.5 * Ty / Tn;
      };
      Wx = ERRF_CONST * Rx[0];
      Wy = ERRF_CONST * Ry[0];
   }

   if (y == 0.) {
      Wx = exp(-x * x);
   }
   if (in_imag < 0.) {

      ftype exp_x2_y2 = exp(y * y - x * x);
      Wx =   2.0 * exp_x2_y2 * cos(2.0 * x * y) - Wx;
      Wy = - 2.0 * exp_x2_y2 * sin(2.0 * x * y) - Wy;
      if (in_real > 0.) {
         Wy = -Wy;
      }
   }
   else if (in_real < 0.) {
      Wy = -Wy;
   }

   return C(Wx,Wy);
}


WITHIN_KERNEL
ctype cErrF_2(ctype x)
{
  // ctype I = C(0.0,1.0);
  ctype z = cmul(I,x);
  ctype result = cmul( cexp(  cmul(C(-1,0),cmul(x,x))   ) , faddeeva(z) );

  //printf("z = %+.16f %+.16fi\n", z.x, z.y);
  //printf("fad = %+.16f %+.16fi\n", faddeeva(z).x, faddeeva(z).y);

  if (x.x > 20.0){// && fabs(x.y < 20.0)
    result = C(0.0,0);
  }
  if (x.x < -20.0){// && fabs(x.y < 20.0)
    result = C(2.0,0);
  }

  return result;
}


WITHIN_KERNEL
ctype old_cerfc(ctype z)
{
  if (z.y<0)
  {
    ctype ans = cErrF_2( C(-z.x, -z.y) );
    return C( 2.0-ans.x, -ans.y);
  }
  else{
    return cErrF_2(z);
  }
}


  WITHIN_KERNEL
ctype getK(const ctype z, const int n)
{
  ctype z2 = cmul(z,z);
  ctype z3 = cmul(z,z2);
  ctype z4 = cmul(z,z3);
  ctype z5 = cmul(z,z4);
  ctype z6 = cmul(z,z5);
  ctype w;

  if (n == 0)
  {
    w = cmul( C(2.0,0.0), z);
    return cdiv(C( 1.0,0.0), w );
  }
  else if (n == 1)
  {
    w = cmul( C(2.0,0.0), z2);
    return cdiv(C(1.0,0.0), w );
  }
  else if (n == 2)
  {
    w = cdiv( C(1.0,0.0), z2 );
    w = cadd( C(1.0,0.0), w );
    return cmul( cdiv(C(1.0,0.0),z) , w );
  }
  else if (n == 3)
  {
    w = cdiv( C(1.0,0.0), z2 );
    w = cadd( C(1.0,0.0), w );
    return cmul( cdiv(C(3.0,0.0),z2) , w );
  }
  // else if (n == 4) {
  //   return cdiv(C( 6.,0), z*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 5) {
  //   return cdiv(C(30.,0), (z*z)*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 6) {
  //   return cdiv(C(60.,0), z*(1.+3./(z*z)+6./(z*z*z*z)+6./(z*z*z*z*z*z))  );
  // }

  return C(0.,0.);
}



  WITHIN_KERNEL
ctype getM(ftype x, int n, ftype t, ftype sigma, ftype gamma, ftype omega)
{
  ctype conv_term, z;
  ctype I2 = C(-1,0);
  ctype I3 = C(0,-1);

  z = C(gamma*sigma/sqrt(2.0),-omega*sigma/sqrt(2.0));
  ctype arg1 = csub( cmul(z,z), cmul(C(2*x,0),z) );
  ctype arg2 = csub(z,C(x,0));
  //conv_term = 5.0*expconv(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));
  // warning there are improvement to do here!!!
  if (omega == 0){
    //conv_term = cmul( cexp(arg1), ipanema_erfc(arg2) );
    conv_term = cmul( cexp(arg1), old_cerfc(arg2) );
  }
  else{
    conv_term = cmul( cexp(arg1), old_cerfc(arg2) );
    //conv_term = 2.0*expconv_simon(t,gamma,omega,sigma);
    //conv_term = 2.0*exp(-gamma*t+0.5*gamma*gamma*sigma*sigma-0.5*omega*omega*sigma*sigma)*(cos(omega*(t-gamma*sigma*sigma)) + I*sin(omega*(t-gamma*sigma*sigma)));
  }
  //conv_term = 2.0*expconv_simon(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));

// #if DEBUG
//   if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) ){
//     printf("\nerfc*exp = %+.16f %+.16fi\n",  conv_term.x, conv_term.y);
//     // printf("erfc = %+.16f %+.16fi\n",  ipanema_erfc(arg2).x, ipanema_erfc(arg2).y );
//     // printf("cErrF_2 = %+.16f %+.16fi\n",  cErrF_2(arg2).x, cErrF_2(arg2).y );
//     // printf("exp  = %+.16f %+.16fi\n",  cexp(arg1).x, cexp(arg1).y );
//     // printf("z    = %+.16f %+.16fi     %+.16f %+.16f %+.16f        x = %+.16f\n",  z.x, z.y, gamma, omega, sigma, x);
//   }
// #endif

  if (n == 0)
  {
    ctype a = C(erf(x),0.);
    ctype b = conv_term;
    return csub(a,b);
  }
  else if (n == 1)
  {
    ctype a = C(sqrt(1.0/M_PI)*exp(-x*x),0.);
    ctype b = C(x,0);
    b = cmul(b,conv_term);
    return cmul(C(-2.0,0.0),cadd(a,b));
  }
  else if (n == 2)
  {
    ctype a = C(-2.*x*exp(-x*x)*sqrt(1./M_PI),0.);
    ctype b = C(2*x*x-1,0);
    b = cmul(b,conv_term);
    return cmul(C(2,0),csub(a,b));
  }
  else if (n == 3)
  {
    ctype a = C(-(2.*x*x-1.)*exp(-x*x)*sqrt(1./M_PI),0.);
    ctype b = C(x*(2*x*x-3),0);
    b = cmul(b,conv_term);
    return cmul(C(4,0),csub(a,b));
  }
  // else if (n == 4)
  // {
  //   return 4.*(exp(-x*x)*(6.*x+4.*x*x*x)*ctype(sqrt(1./M_PI),0.)-(3.-12.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 5)
  // {
  //   return 8.*(-(3.-12.*x*x+4.*x*x*x*x)*exp(-x*x)*ctype(sqrt(1./M_PI),0.)-x*(15.-20.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 6)
  // {
  //   return 8.*(-exp(-x*x)*(30.*x-40.*x*x*x+8.*x*x*x*x*x)*ctype(sqrt(1./M_PI),0.)-(-15.+90.*x*x-60.*x*x*x*x+8.*x*x*x*x*x*x)*conv_term);
  // }
  return C(0.,0.);
}

WITHIN_KERNEL ctype integral_rongin(const int k, const ftype mu,
                                    const ftype sigma, const ctype *M1,
                                    const ctype *M2, const ctype *K) {
  // if (get_global_id(0) == 0){
  //   printf("k -> %d  tLL->%f, tUL->%f, G->%f, DM->%f, mu->%f, sigma->%f\n",
  //   k, tLL, tUL, G, DM, mu, sigma);
  // }
  // ctype z = C(sigma*G/M_SQRT2, -sigma*DM/M_SQRT2);
  // ctype xLL = C((tLL-mu)/(M_SQRT2*sigma), 0);
  // ctype xUL = C((tUL-mu)/(M_SQRT2*sigma), 0);

  ctype ans = C(0, 0);
  ctype rex = C(0, 0);
  ctype morito = C(0, 0);
  ctype wifi = C(0, 0);

  // ctype arr_M[4] = {
  //   csub(getM(cre(xUL), 0, tUL, sigma, G, DM),
  //        getM(cre(xLL), 0, tLL, sigma, G, DM)),
  //   csub(getM(cre(xUL), 1, tUL, sigma, G, DM),
  //        getM(cre(xLL), 1, tLL, sigma, G, DM)),
  //   csub(getM(cre(xUL), 2, tUL, sigma, G, DM),
  //        getM(cre(xLL), 2, tLL, sigma, G, DM)),
  //   csub(getM(cre(xUL), 3, tUL, sigma, G, DM),
  //        getM(cre(xLL), 3, tLL, sigma, G, DM)),
  // };

  const int n_start = mu != 0 ? 0 : k;
  for (int n = n_start; n <= k; n++) {
    rex =
        C(binom(k, n) * rpow(M_SQRT2 * sigma, n) * rpow(mu, k - n) / rpow(2, n),
          0);
    morito = C(0, 0);
    for (int i = 0; i <= n; i++) {
      wifi = cmul(C(binom(n, i), 0), K[i]);
      // wifi = cmul(wifi, arr_M[n-i]);
      wifi = cmul(wifi, csub(M2[n - i], M1[n - i]));
      morito = cadd(morito, wifi);
    }
    ans = cadd(ans, cmul(rex, morito));
  }
  ans = cmul(C(sigma / M_SQRT2, 0), ans);
  return ans;
}


WITHIN_KERNEL void intgTimeAcceptanceOffset(ftype time_terms[4],
                                            const ftype sigma, const ftype G,
                                            const ftype DG, const ftype DM,
                                            GLOBAL_MEM const ftype *coeffs,
                                            const ftype mu, const ftype tLL,
                                            const ftype tUL) {

  // ftype tS = 0;
  // ftype tE = 0;
  ctype int_expm = C(0., 0.);
  ctype int_expp = C(0., 0.);
  ctype int_trig = C(0., 0.);
  ctype int_expm_aux, int_expp_aux, int_trig_aux;

  const int degree = 3;
  ctype Ik = C(0, 0);
  ctype ak = C(0, 0);
  // ctype z = C(sigma*G/M_SQRT2, -sigma*DM/M_SQRT2);

  ctype z_expm, K_expm[4], M_expm[SPL_BINS + 1][4];
  ctype z_expp, K_expp[4], M_expp[SPL_BINS + 1][4];
  ctype z_trig, K_trig[4], M_trig[SPL_BINS + 1][4];

  // compute z
  const ctype cte2 = C(sigma / (M_SQRT2), 0);
  z_expm = cmul(cte2, C(G - 0.5 * DG, 0));
  z_expp = cmul(cte2, C(G + 0.5 * DG, 0));
  z_trig = cmul(cte2, C(G, -DM));

  // Fill K and M
  ftype xS = 0.0;
  ftype tS = 0.0;
  for (int j = 0; j <= degree; ++j) {
    K_expp[j] = getK(z_expp, j);
    K_expm[j] = getK(z_expm, j);
    K_trig[j] = getK(z_trig, j);
    for (int bin = 0; bin < SPL_BINS + 1; ++bin) {

      tS = (bin == NKNOTS) ? tUL : KNOTS[bin];
      // if (bin == NKNOTS-1){ tS = KNOTS[bin+0]; tE = tUL; }
      // else{ tS = KNOTS[bin+0]; tE = KNOTS[bin+1]; }

      xS = (tS - mu) / (M_SQRT2 * sigma);
      M_expm[bin][j] = getM(xS, j, tS - mu, sigma, G - 0.5 * DG, 0.);
      M_expp[bin][j] = getM(xS, j, tS - mu, sigma, G + 0.5 * DG, 0.);
      M_trig[bin][j] = getM(xS, j, tS - mu, sigma, G, DM);
    }
  }

  for (int bin = 0; bin < NKNOTS; bin++) {
    // for each of the time bins, we need to compute the integral
    // whichs is N = ak * Ik. there is one of these for each of
    // the 3 time terms cosh, sinh and (cos, sin)
    int_expm_aux = C(0, 0);
    int_expp_aux = C(0, 0);
    int_trig_aux = C(0, 0);
    // if (bin == NKNOTS-1){ tS = KNOTS[bin+0]; tE = tUL; }
    // else{ tS = KNOTS[bin+0]; tE = KNOTS[bin+1]; }
    //
    // ctype xLL = C((tS-mu)/(M_SQRT2*sigma), 0);
    // ctype xUL = C((tE-mu)/(M_SQRT2*sigma), 0);

    for (int k = 0; k <= degree; k++) {
      ak = C(getCoeff(coeffs, bin, k), 0);
      // Ik = integral_rongin(k, tS, tE, G-0.5*DG, 0, mu, sigma);
      Ik = integral_rongin(k, mu, sigma, M_expm[bin], M_expm[bin + 1], K_expm);
      // if (get_global_id(0) == 0){
      //   printf("k -> %d  tLL->%f, tUL->%f, w->%f %+f*I, mu->%f, sigma->%f\n",
      //          k, KNOTS[bin+0], KNOTS[bin+1], cre(z_expm), cim(z_expm), mu,
      //          sigma);
      // }
      // if (get_global_id(0) == 0){
      //   printf("I_%d = %.6f %+.6f i\n", k, cre(Ik), cim(Ik));
      // }
      int_expm_aux = cadd(int_expm_aux, cmul(ak, Ik));
      // Ik = integral_rongin(k, tS, tE, G+0.5*DG, 0, mu, sigma);
      Ik = integral_rongin(k, mu, sigma, M_expp[bin], M_expp[bin + 1], K_expp);
      int_expp_aux = cadd(int_expp_aux, cmul(ak, Ik));
      // Ik = integral_rongin(k, tS, tE, G,       DM, mu, sigma);
      Ik = integral_rongin(k, mu, sigma, M_trig[bin], M_trig[bin + 1], K_trig);
      // if (get_global_id(0) == 0){
      //   printf("k -> %d  tLL->%f, tUL->%f, G->%f, DM->%f, mu->%f,
      //   sigma->%f\n",
      //          k, KNOTS[bin+0], KNOTS[bin+1], G, DM, mu, sigma);
      // }
      // if (get_global_id(0) == 0){
      //   printf("I_%d = %.6f %+.6f i\n", k, cre(Ik), cim(Ik));
      // }
      int_trig_aux = cadd(int_trig_aux, cmul(ak, Ik));
    }

    int_expm = cadd(int_expm, int_expm_aux);
    int_expp = cadd(int_expp, int_expp_aux);
    int_trig = cadd(int_trig, int_trig_aux);
  }
  // return the four integrals
  time_terms[0] = 0.5 * (int_expm.x + int_expp.x);
  time_terms[1] = 0.5 * (int_expm.x - int_expp.x);
  time_terms[2] = int_trig.x;
  time_terms[3] = int_trig.y;
}

WITHIN_KERNEL void
intgTimeAcceptance(ftype time_terms[4], 
                   const ftype delta_t, const ftype G,
                   const ftype DG, const ftype DM, 
                   GLOBAL_MEM const ftype *coeffs,
                   const ftype t0, 
                   const ftype tLL, const ftype tUL)
{
  // Some constants
  const ftype cte1 = 1.0/(sqrt(2.0)*delta_t);
  const ctype cte2 = C(delta_t/(sqrt(2.0)), 0);
  #if DEBUG
  if (DEBUG > 3)
  {
    /* printf("WARNING            : mu = %.4f\n", t0); */
    if (delta_t <= 0)
    {
      printf("ERROR               : delta_t = %.4f is not a valid value.\n", delta_t);
    }
  }
  #endif

  // Add tUL to knots list
  ftype x[NTIMEBINS] = {0.};
  ftype knots[NTIMEBINS] = {0.};
  knots[0] = tLL; x[0] = (knots[0] - t0)*cte1;
  for(int i = 1; i < NKNOTS; i++)
  {
    knots[i] = KNOTS[i];
    x[i] = (knots[i] - t0)*cte1;
  }
  knots[NKNOTS] = tUL;
  x[NKNOTS] = (knots[NKNOTS] - t0)*cte1;

  ftype S[SPL_BINS][4][4];
  for (int bin=0; bin < SPL_BINS; ++bin)
  {
    for (int i=0; i<4; ++i)
    {
      for (int j=0; j<4; ++j)
      {
        if(i+j < 4)
        {
          S[bin][i][j] = getCoeff(coeffs,bin,i+j) * factorial(i+j) / factorial(j) / factorial(i) / rpow(2.0,i+j);
        }
        else
        {
          S[bin][i][j] = 0.;
        }
      }
    }
  }

  ctype z_expm, K_expm[4], M_expm[SPL_BINS+1][4];
  ctype z_expp, K_expp[4], M_expp[SPL_BINS+1][4];
  ctype z_trig, K_trig[4], M_trig[SPL_BINS+1][4];

  z_expm = cmul( cte2 , C(G-0.5*DG,  0) );
  z_expp = cmul( cte2 , C(G+0.5*DG,  0) );
  z_trig = cmul( cte2 , C(       G,-DM) );

  // Fill Kn                 (only need to calculate this once per minimization)
  for (int j=0; j<4; ++j)
  {
    K_expp[j] = getK(z_expp,j);
    K_expm[j] = getK(z_expm,j);
    K_trig[j] = getK(z_trig,j);
    #if DEBUG
    if (DEBUG > 3 && (get_global_id(0) == DEBUG_EVT) )
    {
      printf("K_expp[%d](%+.14f%+.14f) = %+.14f%+.14f\n",  j,z_expp.x,z_expp.y,K_expp[j].x,K_expp[j].y);
      printf("K_expm[%d](%+.14f%+.14f) = %+.14f%+.14f\n",  j,z_expm.x,z_expm.y,K_expm[j].x,K_expm[j].y);
      printf("K_trig[%d](%+.14f%+.14f) = %+.14f%+.14f\n\n",j,z_trig.x,z_trig.y,K_trig[j].x,K_trig[j].y);
    }
    #endif
  }

  // Fill Mn
  for (int j=0; j<4; ++j)
  {
    for(int bin=0; bin < SPL_BINS+1; ++bin)
    {
      M_expm[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G-0.5*DG,0.);
      M_expp[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G+0.5*DG,0.);
      M_trig[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G       ,DM);
      if (bin>0){
        #if DEBUG
        if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
        {
          ctype aja = M_expp[bin][j];//-M_expp[bin-1][j];
          ctype eje = M_expm[bin][j];//-M_expm[bin-1][j];
          ctype iji = M_trig[bin][j];//-M_trig[bin-1][j];
          printf("bin=%d M_expp[%d] = %+.14f%+.14f\n",  bin,j,aja.x,aja.y);
          printf("bin=%d M_expm[%d] = %+.14f%+.14f\n",  bin,j,eje.x,eje.y);
          printf("bin=%d M_trig[%d] = %+.14f%+.14f\n\n",bin,j,iji.x,iji.y);
        }
        #endif
      }
    }
  }

  // Fill the delta factors to multiply by the integrals
  ftype delta_t_fact[4];
  for (int i=0; i<4; ++i)
  {
    delta_t_fact[i] = rpow(delta_t*sqrt(2.), i+1) / sqrt(2.);
  }

  // Integral calculation for cosh, expm, cos, sin terms
  ctype int_expm = C(0.,0.);
  ctype int_expp = C(0.,0.);
  ctype int_trig = C(0.,0.);
  ctype aux, int_expm_aux, int_expp_aux, int_trig_aux;

  for (int bin=0; bin < SPL_BINS; ++bin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        aux = C( S[bin][j][k]*delta_t_fact[j+k], 0 );

        int_expm_aux = csub(M_expm[bin+1][j],M_expm[bin][j]);
        int_expm_aux = cmul(int_expm_aux,K_expm[k]);
        int_expm_aux = cmul(int_expm_aux,aux);
        int_expm     = cadd( int_expm, int_expm_aux );

        int_expp_aux = csub(M_expp[bin+1][j],M_expp[bin][j]);
        int_expp_aux = cmul(int_expp_aux,K_expp[k]);
        int_expp_aux = cmul(int_expp_aux,aux);
        int_expp     = cadd( int_expp, int_expp_aux );

        int_trig_aux = csub(M_trig[bin+1][j],M_trig[bin][j]);
        int_trig_aux = cmul(int_trig_aux,K_trig[k]);
        int_trig_aux = cmul(int_trig_aux,aux);
        int_trig     = cadd( int_trig, int_trig_aux );

        #if DEBUG
        if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
        {
          printf("bin=%d int_trig[%d,%d] = %+.14f%+.14f\n",  bin,j,k,int_trig.x,int_trig.y);
        }
        #endif
      }
    }
  }

  // Fill itengral terms - 0:cosh, 1:sinh, 2:cos, 3:sin
  time_terms[0] = sqrt(0.5)*0.5*(int_expm.x+int_expp.x);
  time_terms[1] = sqrt(0.5)*0.5*(int_expm.x-int_expp.x);
  time_terms[2] = sqrt(0.5)*int_trig.x;
  time_terms[3] = sqrt(0.5)*int_trig.y;

  #if DEBUG
  if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
  {
    printf("\nNORMALIZATION      : ta=%.16f\ttb=%.16f\ttc=%.16f\ttd=%.16f\n",
        time_terms[0],time_terms[1],time_terms[2],time_terms[3]);
    printf("                   : sigma=%.16f\tgamma+=%.16f\tgamma-=%.16f\n",
        delta_t, G+0.5*DG, G-0.5*DG);
  }
  #endif
}


  // }}}


















WITHIN_KERNEL
ftype get_omega(const ftype eta, const ftype tag, 
                const ftype p0, const ftype p1, const ftype p2, 
                const ftype dp0, const ftype dp1, const ftype dp2, 
                const ftype eta_bar) {
  ftype omega = 0;
  omega += (p0 + tag * 0.5 * dp0);
  omega += (p1 + tag * 0.5 * dp1) * (eta - eta_bar);
  omega += (p2 + tag * 0.5 * dp2) * (eta - eta_bar) * (eta - eta_bar);

  // if (omega < 0.0) {
  //   return 0;
  // }
  return omega;
}

WITHIN_KERNEL
ftype get_domega(const ftype eta, 
                const ftype dp0, 
                const ftype dp1, 
                const ftype dp2, 
                const ftype eta_bar) {
  ftype domega = 0;
  domega += dp0;
  domega += dp1 * (eta - eta_bar);
  domega += dp2 * (eta - eta_bar);
  return domega;
}





WITHIN_KERNEL
void calibrated_mistag(GLOBAL_MEM const ftype *b_eta, 
                       GLOBAL_MEM const ftype *b_id,
                       ftype *out,
                       const ftype p0, const ftype p1, const ftype p2,
                       const ftype dp0, const ftype dp1, const ftype dp2,
                       const ftype eta_bar,
                       const int Nevt
                       ){
  int row = get_global_id(0);
  const ftype eta = b_eta[row];  //mistag
  const ftype id = b_id[row]; //id 

  const ftype q_true = id/fabs(id);
  const ftype om = get_omega(eta, q_true, p0, p1, p2, dp0, dp1, dp2, eta_bar);
  out[row] = om;
}



KERNEL
void os_calibration(
                    GLOBAL_MEM const ftype *data,
                    GLOBAL_MEM ftype *pdf, 
                    const ftype p0, const ftype dp0, 
                    const ftype p1, const ftype dp1, 
                    const ftype p2, const ftype dp2, 
                    const ftype eta_bar,  
                    const int Nevt) {

  const int row = get_global_id(0); //evt

  if (row >= Nevt) {
    return;
  }
  // #if DEBUG 
  // if (row == DEBUG_EVT && DEBUG==1) { 
  // printf("FLAVOUR TAGGING PARAMETERS:\n");
  // printf("'p0'= %+.4f  'p1' =%+.4f 'p2'=%+.4f\n", p0, p1, p2); 
  // printf("'dp0'= %+.4f  'dp1' =%+.4f 'dp2'=%+.4f\n", dp0, dp1, dp2); 
  // printf("eta_bar' = %+.4f  \n", eta_bar); 
  // } 
  // #endif 


  const ftype q = data[0+row*3]; //tag dec
  const ftype eta = data[1+row*3];  //mistag
  const ftype id = data[2+row*3]; //id 

  //B+ and MC we know original flavour:
  const ftype q_true = id / fabs(id);

  ftype a = 0; //China notation see ANAnote
  if (q_true==q){
    a = 1.; 
  }

  const ftype omega = get_omega(eta, q_true, p0, p1, p2, dp0, dp1, dp2, eta_bar);
  pdf[row] = (1-a)*omega + a*(1-omega); 
}

KERNEL
void ss_calibration(GLOBAL_MEM const ftype *data,
                    GLOBAL_MEM ftype *pdf, GLOBAL_MEM ftype *coeffs,
                    const ftype G, const ftype DG, const ftype DM,
                    const ftype p0, const ftype dp0, const ftype p1,
                    const ftype dp1, const ftype p2, const ftype dp2,
                    const ftype eta_bar, 
                    const ftype sigma_0, const ftype sigma_1, const ftype mu,
                    const ftype tLL, const ftype tUL,
                    const int Nevt) {

  const int row = get_global_id(0); //evt

  if (row >= Nevt) {
    return;
  }
  const ftype q = data[0+row*5]; //tag dec
  const ftype eta = data[1+row*5];  //mistag
  const ftype id = data[2+row*5]; //id 
  const ftype time = data[3+row*5]; //time branch
  const ftype sigmat = data[4+row*5]; //time error for res
 
/* #if DEBUG */
/*   if (row == DEBUG_EVT && DEBUG==1) { */
/*   printf("FLAVOUR TAGGING PARAMETERS:\n"); */
/*   printf("'p0'= %+.4f  'p1' =%+.4f 'p2'=%+.4f\n", p0, p1, p2); */
/*   printf("'dp0'= %+.4f  'dp1' =%+.4f 'dp2'=%+.4f\n", dp0, dp1, dp2); */
/*   printf("eta_bar' = %+.4f  \n", eta_bar); */
/*   printf("DATA PARAMETERS:\n"); */
/*   printf("'q'= %+.4f  'eta' =%+.4f 'id'=%+.4f\n", q, eta, id); */
/*   printf("'time'= %+.4f  'sigmat' =%+.4f\n", time, sigmat); */
/*   printf("TIME CONSTANTS:\n"); */
/*   printf("'G'= %+.4f  'DG' =%+.4f 'DM'=%+.4f\n", G, DG, DM); */
/*   printf("'tLL'= %+.4f  'tUL' =%+.4f\n", tLL, tUL); */
/*   printf("TIME Resolution:\n"); */
/*   printf("'sigma0'= %+.4f  'sigma1' =%+.4f\n", sigma_0, sigma_1); */
/*   printf("COEFFS             : %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]); */
/*   printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]); */
/*   printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]); */
/*   printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]); */
/*   } */
/* #endif */

  ftype t_offset = mu; 
  ftype delta_t = sigma_0 + sigma_1 * sigmat; //Apply calibration to sigma

  ctype exp_p = C(0, 0);
  ctype exp_m = C(0, 0);
  ctype exp_i = C(0, 0);
  
  exp_p = expconv(time - t_offset, G + 0.5 * DG, 0., delta_t); //Joga bonito
  exp_m = expconv(time - t_offset, G - 0.5 * DG, 0., delta_t);
  exp_i = expconv(time - t_offset, G, DM, delta_t);

  ftype ta = 0.5 * (exp_m.x + exp_p.x); //Conv part cosh
  ftype tc = exp_i.x;  //Conv part cos

  ftype dta = 1.;
  dta = time_efficiency(time, coeffs, tLL, tUL);


  //Norm (Thanks marcs)
  ftype integrals[4] = {0., 0., 0., 0.};
  /* intgTimeAcceptance(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL); */
  //well described time acc with offset
  intgTimeAcceptanceOffset(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL);
  
  //Only need ta and tc (cosh and cos)
  const ftype int_ta = integrals[0];
  const ftype int_tc = integrals[2];
  

  const ftype q_end = id / fabs(id);
  ftype q_mix = 0.;

  if (q_end*q > 0){
    q_mix = 1.;
  }
  else if (q_end*q < 0){
    q_mix = -1.;
  }
  //should not depend on tag: -> to check
  const ftype om_b = get_omega(eta, +1., p0, p1, p2, dp0, dp1, dp2, eta_bar); 
  const ftype om_bbar = get_omega(eta, -1., p0, p1, p2, dp0, dp1, dp2, eta_bar); 
  const ftype om = (om_b + om_bbar)/2;
  const ftype dom = om_b - om_bbar;
  /* const ftype dom = get_domega(eta, dp0, dp1, dp2, eta_bar);  */
  /* const ftype dom = get_omega(eta, 0., 0., 0., 0., dp0, dp1, dp2, eta_bar);  */

  const ftype num = dta * ((1.-q*dom)*ta + q_mix * (1 - 2 * om) * tc);
  //Integral also in q and q_f summing over all posibilities:
  const ftype den = int_ta; 
  pdf[row] = num / den;

}

KERNEL
void ss_calibration_wouter(GLOBAL_MEM const ftype *data,
                    GLOBAL_MEM ftype *pdf, GLOBAL_MEM ftype *coeffs,
                    const ftype G, const ftype DG, const ftype DM,
                    const ftype f1, const ftype f2,
                    const ftype sigma_0, const ftype sigma_1, const ftype mu,
                    const ftype tLL, const ftype tUL,
                    const int Nevt) {

  const int row = get_global_id(0); //evt

  if (row >= Nevt) {
    return;
  }
  const ftype q = data[0+row*5]; //tag dec
  const ftype eta = data[1+row*5];  //mistag
  const ftype id = data[2+row*5]; //id 
  const ftype time = data[3+row*5]; //time branch
  const ftype sigmat = data[4+row*5]; //time error for res
 

  ftype t_offset = mu; 
  ftype delta_t = sigma_0 + sigma_1 * sigmat; //Apply calibration to sigma

  ctype exp_p = C(0, 0);
  ctype exp_m = C(0, 0);
  ctype exp_i = C(0, 0);
  
  exp_p = expconv(time - t_offset, G + 0.5 * DG, 0., delta_t); //Joga bonito
  exp_m = expconv(time - t_offset, G - 0.5 * DG, 0., delta_t);
  exp_i = expconv(time - t_offset, G, DM, delta_t);

  ftype ta = 0.5 * (exp_m.x + exp_p.x); //Conv part cosh
  ftype tc = exp_i.x;  //Conv part cos

  ftype dta = 1.;
  dta = time_efficiency(time, coeffs, tLL, tUL);


  //Norm (Thanks marcs)
  ftype integrals[4] = {0., 0., 0., 0.};
  /* intgTimeAcceptance(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL); */
  //well described time acc with offset
  intgTimeAcceptanceOffset(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL);
  
  //Only need ta and tc (cosh and cos)
  const ftype int_ta = integrals[0];
  const ftype int_tc = integrals[2];
  

  const ftype q_end = id / fabs(id);
  ftype q_mix = 0.;

  if (q_end*q > 0){
    q_mix = 1.;
  }
  else if (q_end*q < 0){
    q_mix = -1.;
  }
  //should not depend on tag: -> to check
  const ftype om_b = f1*eta + 0.5*(1 - f1);
  const ftype om_bbar = f2*eta + 0.5*(1 - f2);
  const ftype om = (om_b + om_bbar)/2;
  const ftype dom = (om_b - om_bbar)/2;

  const ftype num = dta * ((1.-q*dom)*ta + q_mix * (1 - 2 * om) * tc);
  //Integral also in q and q_f summing over all posibilities:
  const ftype den = int_ta; 
  pdf[row] = num / den;

}
/* KERNEL */
/* void ss_calibration_integral_2(GLOBAL_MEM const ftype *data, */
/*                     GLOBAL_MEM ftype *pdf, GLOBAL_MEM ftype *coeffs, */
/*                     const ftype G, const ftype DG, const ftype DM, */
/*                     const ftype p0, const ftype dp0, const ftype p1, */
/*                     const ftype dp1, const ftype p2, const ftype dp2, */
/*                     const ftype eta_bar,  */
/*                     const ftype sigma_0, const ftype sigma_1, const ftype mu, */
/*                     const ftype tLL, const ftype tUL, */
/*                     const int Nevt) { */
/**/
/*   const int row = get_global_id(0); //evt */
/**/
/*   if (row >= Nevt) { */
/*     return; */
/*   } */
/*   const ftype q = data[0+row*5]; //tag dec */
/*   const ftype eta = data[1+row*5];  //mistag */
/*   const ftype id = data[2+row*5]; //id  */
/*   const ftype time = data[3+row*5]; //time branch */
/*   const ftype sigmat = data[4+row*5]; //time error for res */
/*   */
/**/
/*   ftype t_offset = mu;  */
/*   ftype delta_t = sigma_0 + sigma_1 * sigmat; //Apply calibration to sigma */
/**/
/*   ctype exp_p = C(0, 0); */
/*   ctype exp_m = C(0, 0); */
/*   ctype exp_i = C(0, 0); */
/*    */
/*   exp_p = expconv(time - t_offset, G + 0.5 * DG, 0., delta_t); //Joga bonito */
/*   exp_m = expconv(time - t_offset, G - 0.5 * DG, 0., delta_t); */
/*   exp_i = expconv(time - t_offset, G, DM, delta_t); */
/**/
/*   ftype ta = 0.5 * (exp_m.x + exp_p.x); //Conv part cosh */
/*   ftype tc = exp_i.x;  //Conv part cos */
/**/
/*   ftype dta = 1.; */
/*   dta = time_efficiency(time, coeffs, tLL, tUL); */
/**/
/**/
/*   //Norm (Thanks marcs) */
/*   ftype integrals[4] = {0., 0., 0., 0.}; */
/*   //well described time acc with offset */
/*   intgTimeAcceptanceOffset(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL); */
/*    */
/*   //Only need ta and tc (cosh and cos) */
/*   const ftype int_ta = integrals[0]; */
/*   const ftype int_tc = integrals[2]; */
/**/
/*   //We dont know the flavor :( */
/*   const ftype q_end = id / fabs(id); */
/*   ftype q_mix = 0.; */
/**/
/*   if (q_end*q > 0){ */
/*     q_mix = 1.; */
/*   } */
/*   else if (q_end*q < 0){ */
/*     q_mix = -1.; */
/*   } */
/*   //should not depend on tag: -> to check */
/*   const ftype om = get_omega(eta, 0., p0, p1, p2, 0., 0., 0., eta_bar);  */
/*   const ftype dom = get_domega(eta, dp0, dp1, dp2, eta_bar);  */
/**/
/*   const ftype den = int_ta; */
/*   pdf[row] = den; */

/* } */

/* KERNEL */
/* void ss_calibration_integral(GLOBAL_MEM const ftype *data, */
/*                     GLOBAL_MEM ftype *pdf, GLOBAL_MEM ftype *coeffs, */
/*                     const ftype G, const ftype DG, const ftype DM, */
/*                     const ftype p0, const ftype dp0, const ftype p1, */
/*                     const ftype dp1, const ftype p2, const ftype dp2, */
/*                     const ftype eta_bar,  */
/*                     const ftype sigma_0, const ftype sigma_1, const ftype mu, */
/*                     const ftype tLL, const ftype tUL, */
/*                     const ftype q_ss, const ftype q_f, */
/*                     const int Nevt) { */
/**/
/*   const int row = get_global_id(0); //evt */
/**/
/*   if (row >= Nevt) { */
/*     return; */
/*   } */
/*   const ftype q = data[0+row*5]; //tag dec */
/*   const ftype eta = data[1+row*5];  //mistag */
/*   const ftype id = data[2+row*5]; //id  */
/*   const ftype time = data[3+row*5]; //time branch */
/*   const ftype sigmat = data[4+row*5]; //time error for res */
/*   */
/**/
/*   ftype t_offset = mu;  */
/*   ftype delta_t = sigma_0 + sigma_1 * sigmat; //Apply calibration to sigma */
/**/
/*   ctype exp_p = C(0, 0); */
/*   ctype exp_m = C(0, 0); */
/*   ctype exp_i = C(0, 0); */
/*    */
/*   exp_p = expconv(time - t_offset, G + 0.5 * DG, 0., delta_t); //Joga bonito */
/*   exp_m = expconv(time - t_offset, G - 0.5 * DG, 0., delta_t); */
/*   exp_i = expconv(time - t_offset, G, DM, delta_t); */
/**/
/*   ftype ta = 0.5 * (exp_m.x + exp_p.x); //Conv part cosh */
/*   ftype tc = exp_i.x;  //Conv part cos */
/**/
/*   ftype dta = 1.; */
/*   dta = time_efficiency(time, coeffs, tLL, tUL); */
/**/
/**/
/*   //Norm (Thanks marcs) */
/*   ftype integrals[4] = {0., 0., 0., 0.}; */
/*   intgTimeAcceptanceOffset(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL); */
/*    */
/*   //Only need ta and tc (cosh and cos) */
/*   const ftype int_ta = integrals[0]; */
/*   const ftype int_tc = integrals[2]; */
/**/
/*   //We dont know the flavor :( */
/*   const ftype q_end = id / fabs(id); */
/*   ftype q_mix = 0.; */
/**/
/*   if (q_end*q > 0){ */
/*     q_mix = 1.; */
/*   } */
/*   else if (q_end*q < 0){ */
/*     q_mix = -1.; */
/*   } */
/*   const ftype om = get_omega(eta, 0., p0, p1, p2, 0., 0., 0., eta_bar);  */
/*   const ftype dom = get_domega(eta, dp0, dp1, dp2, eta_bar);  */
/**/
/*   const ftype den = ((1.-q_ss*dom)*int_ta + q_ss*q_f * (1 - 2 * om) * int_tc); */
/*   pdf[row] = den; */
/**/
/* } */


WITHIN_KERNEL
ftype integral_china(
                    const ftype sigmat,
                    const ftype id,
                    const ftype q,
                    const ftype eta,
                    const ftype a, const ftype b, const ftype n, //eff parameters
                    const ftype G, const ftype DG, const ftype DM,
                    const ftype p0, const ftype dp0, const ftype p1,
                    const ftype dp1, const ftype p2, const ftype dp2,
                    const ftype eta_bar, 
                    const ftype sigma_0, const ftype sigma_1,
                    const ftype tLL, const ftype tUL)
                    {

  ftype diff = 10.;
  ftype nevals = 1000.;
  int it = 0;
  ftype fs[2] = {0., 0.};
  ftype f;
  while (diff>1.e-4){
    ftype h = (tUL - tLL)/nevals;
    /* printf("it: %d, it previa: %.8f, it nueva: %.8f\n", it, fs[0], fs[1]); */
    if (it >0){
      fs[0] = f;
    }
    f = 0.;
    for(int i = 0; i < nevals; i++)
      {
      ftype time = tLL + i*h;
      /* printf("time: %.4f", time); */
      ftype delta_t = sigma_0 + sigma_1 * sigmat; //Apply calibration to sigma
      ctype exp_p = C(0, 0);
      ctype exp_m = C(0, 0);
      ctype exp_i = C(0, 0);
      exp_p = expconv(time , G + 0.5 * DG, 0., delta_t); //Joga bonito
      exp_m = expconv(time , G - 0.5 * DG, 0., delta_t);
      exp_i = expconv(time, G, DM, delta_t);
      ftype ta = 0.5 * (exp_m.x + exp_p.x); //Conv part cosh
      ftype tc = exp_i.x;  //Conv part cos
      ftype dta = china_eff(time, a, b, n);
      ftype q_end = id / fabs(id);
      ftype q_mix;
      if (q_end*q > 0){
        q_mix = 1.;
       }
      else if (q_end*q < 0){
        q_mix = -1.;
      }
      ftype om = p0 + p1*(eta-eta_bar);
      f += dta * (ta + q_mix * (1 - 2 * om) * tc)*(time-(time-h));  //Riemman integral 
    } //End of loop
    fs[1] = f;
    if (it >20 ){
      diff = 0.;
    }
    if (it>0){
    diff = fabs(fs[1]-fs[0]);
    /* printf("diff :%.e\n", diff); */
    }
    it += 1;
    nevals += 300;
  } //end while
  /* printf("Iteration of integrals: %d\n", it); */
  /* printf("Integral per event value: %+.8f\n", f); */
  return f;
}


//Not normalized :( -> eff: 1- 1./(1+(at)**n -b)
KERNEL
void ss_calibration_china(
                    GLOBAL_MEM const ftype *time_arr,
                    GLOBAL_MEM const ftype *sigmat_arr,
                    GLOBAL_MEM const ftype *id_arr,
                    GLOBAL_MEM const ftype *q_arr,
                    GLOBAL_MEM const ftype *eta_arr,
                    GLOBAL_MEM ftype *pdf, 
                    const ftype a, const ftype b, const ftype n, //eff parameters
                    const ftype G, const ftype DG, const ftype DM,
                    const ftype p0, const ftype dp0, const ftype p1,
                    const ftype dp1, const ftype p2, const ftype dp2,
                    const ftype eta_bar, 
                    const ftype sigma_0, const ftype sigma_1,
                    const ftype tLL, const ftype tUL,
                    const int Nevt) {

  const int row = get_global_id(0); //evt

  if (row >= Nevt) {
    return;
  }
  const ftype q = q_arr[row]; //tag dec
  const ftype eta = eta_arr[row];  //mistag
  const ftype id = id_arr[row]; //id 
  const ftype time = time_arr[row]; //time branch
  const ftype sigmat = sigmat_arr[row]; //time error for res
 
/* #if DEBUG */
/*   if (row == 0 && DEBUG==1) { */
/*   printf("eta_bar' = %+.4f  \n", eta_bar); */
/*   printf("'q'= %+.4f  'eta' =%+.4f 'id'=%+.4f\n", q, eta, id); */
/*   printf("'time'= %+.4f  'sigmat' =%+.4f\n", time, sigmat); */
/*   printf("'a'= %+.4f  'b' =%+.4f 'n' =%+.4f\n", a, b, n); */
/*   } */
/* #endif */

  ftype t_offset = 0.; 
  ftype delta_t = sigma_0 + sigma_1 * sigmat; //Apply calibration to sigma

  ctype exp_p = C(0, 0);
  ctype exp_m = C(0, 0);
  ctype exp_i = C(0, 0);
  
  exp_p = expconv(time - t_offset, G + 0.5 * DG, 0., delta_t); //Joga bonito
  exp_m = expconv(time - t_offset, G - 0.5 * DG, 0., delta_t);
  exp_i = expconv(time - t_offset, G, DM, delta_t);

  ftype ta = 0.5 * (exp_m.x + exp_p.x); //Conv part cosh
  ftype tc = exp_i.x;  //Conv part cos

  ftype dta = 1.;
  dta = china_eff(time, a, b, n);
  /* if (row == DEBUG_EVT && DEBUG==1) { */
  /* printf("dta' =%+.4f\n", dta); */
  /* } */

  //We dont know the flavor :(
  const ftype q_end = id / fabs(id);
  ftype q_mix = 0.;

  if (q_end*q > 0){
    q_mix = 1.;
  }
  else if (q_end*q < 0){
    q_mix = -1.;
  }
  //should not depend on tag: -> to check
  /* const ftype om = get_omega(eta, 1., p0, p1, p2, dp0, dp1, dp2, eta_bar);  */
  const ftype om = p0 + p1*(eta-eta_bar);

  const ftype num = dta * (ta + q_mix * (1 - 2 * om) * tc);
  const ftype den = integral_china(sigmat, id, q, eta, 
                                    a, b, n, G, DG, DM,
                                    p0, dp0, p1, dp1, p2, dp2,
                                    eta_bar, sigma_0, sigma_1,
                                    tLL, tUL);

  /* printf("Numeric integral per event: %.4f\n", den); */
  /* if (row == 0 && DEBUG==1) { */
  /*   printf("'q0'= %+.4f  'q1' =%+.4f ' sigmat = %.4f'\n", sigma_0, sigma_1, sigmat); */
  /*   printf("Numeric integral per event: %.4f\n", den); */
  /* } */

  pdf[row] = num/den; //Not normalized 

}


KERNEL
void plot_ss(
                  GLOBAL_MEM const ftype *time_arr, GLOBAL_MEM const ftype *sigmat_arr,
                  GLOBAL_MEM const ftype *q_arr, GLOBAL_MEM const ftype *id_arr,
                  GLOBAL_MEM ftype *pdf, 
                  GLOBAL_MEM ftype *coeffs,
                  const ftype G, const ftype DG, const ftype DM,
                  const ftype omega,
                  const ftype sigma_0, const ftype sigma_1,
                   const ftype tLL, const ftype tUL,
                   const int Nevt) {

  const int row = get_global_id(0); //evt

  if (row >= Nevt) {
    return;
  }
  const ftype time = time_arr[row];
  const ftype sigmat = sigmat_arr[row];
  const ftype q = q_arr[row];
  const ftype id = id_arr[row];
 
/* #if DEBUG */
/*   if (row == DEBUG_EVT && DEBUG==1) { */
/*   printf("FLAVOUR TAGGING PARAMETERS:\n"); */
/*   printf("DATA PARAMETERS:\n"); */
/*   printf("'q'= %+.4f  'id'=%+.4f\n", q, id); */
/*   printf("'time'= %+.4f  'sigmat' =%+.4f\n", time, sigmat); */
/*   printf("TIME CONSTANTS:\n"); */
/*   printf("'G'= %+.4f  'DG' =%+.4f 'DM'=%+.4f\n", G, DG, DM); */
/*   printf("'omega'= %+.4f\n", omega); */
/*   printf("'tLL'= %+.4f  'tUL' =%+.4f\n", tLL, tUL); */
/*   printf("TIME Resolution:\n"); */
/*   printf("'sigma0'= %+.4f  'sigma1' =%+.4f\n", sigma_0, sigma_1); */
/*   printf("COEFFS             : %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]); */
/*   printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]); */
/*   printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]); */
/*   printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n", */
/*   coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]); */
/*   } */
/* #endif */

  ftype t_offset = 0.; 
  ftype delta_t = sigma_0 + sigma_1 * sigmat; //Apply calibration to sigma

  ctype exp_p = C(0, 0);
  ctype exp_m = C(0, 0);
  ctype exp_i = C(0, 0);
  
  exp_p = expconv(time - t_offset, G + 0.5 * DG, 0., delta_t); //Joga bonito
  exp_m = expconv(time - t_offset, G - 0.5 * DG, 0., delta_t);
  exp_i = expconv(time - t_offset, G, DM, delta_t);

  ftype ta = 0.5 * (exp_m.x + exp_p.x); //Conv part cosh
  ftype tc = exp_i.x;  //Conv part cos

  ftype dta = 1.;
  dta = time_efficiency(time, coeffs, tLL, tUL);


  //Norm (Thanks marcs)
  ftype integrals[4] = {0., 0., 0., 0.};
  intgTimeAcceptance(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL);
  
  //Only need ta and tc (cosh and cos)
  const ftype int_ta = integrals[0];
  const ftype int_tc = integrals[2];

  //We dont know the flavor :(
  const ftype q_end = id / fabs(id);
  ftype q_mix = 0.;

  if (q_end*q > 0){
    q_mix = 1.;
  }
  else if (q_end*q < 0){
    q_mix = -1.;
  }
  //should not depend on tag: -> to check
  /* const ftype om = get_omega(eta, 1., p0, p1, p2, dp0, dp1, dp2, eta_bar);  */

  const ftype num = dta * (ta + q_mix * (1 - 2 * omega) * tc);
  const ftype den = (int_ta + q_mix * (1 - 2 * omega) * int_tc);
  pdf[row] = num / den;

}

KERNEL
void plot_os(GLOBAL_MEM const ftype *q_arr, GLOBAL_MEM const ftype *id_arr,
            GLOBAL_MEM ftype *pdf, 
            const ftype omega,
            const int Nevt) {

  const int row = get_global_id(0); //evt

  if (row >= Nevt) {
    return;
  }
  const ftype q = q_arr[row]; //tag dec
  const ftype id = id_arr[row]; //id

  //B+ and MC we know original flavour:
  const ftype q_true = id / fabs(id);

  ftype a = 0; //China notation see ANAnote
  if (q_true==q){
    a = 1.; 
  }

  pdf[row] = (1-a)*omega + a*(1-omega); 

}
