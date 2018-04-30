#include "cunfft_util.h"
#include "cunfft.h"
#include<string.h>

#define NFFT_PRECISION_DOUBLE
#include <nfft3mp.h>

/**
* NDFT computation for different dimensions.
* @param plan cunfft data structur
* @param f result of NDFT transformation
*/
double ndft(cunfft_plan *plan, gpuComplex *f);
double nfft(cunfft_plan *plan, gpuComplex *f);

/**
* NDFT adjoint computation for different dimensions.
* @param plan cunfft data structur
* @param f_hat result of NDFT adjoint transformation
*/
double ndft_adjoint(cunfft_plan *plan, gpuComplex* f_hat);
double nfft_adjoint(cunfft_plan *plan, gpuComplex* f_hat);

