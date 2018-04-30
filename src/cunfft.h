/*
 * Copyright (c) 2012 Susanne Kunis, Stefan Kunis
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * $Id: cunfft1.h 2012-06-01 11:06:00Z sukunis $
 *
 */

/**
 * @file cunfft1.h
 * @brief CUDA calling functions header file
 */

#ifndef CUNFFT1_H_
#define CUNFFT1_H_

#include "cunfft_typedefs.h"

/**
* Warm up function for device. First touch is much slower.
*/
void warmUp_createPlan(uint_t *n);
//------------------------------------------------------------------------------
// 		MEMORY TRANSFER FUNCTION
//------------------------------------------------------------------------------
/**
 * Transfer data for cunfft transformation to device memory.
 * Compute k_data and copy k_data and fast gaussian gridding data to constant
 * device memory and also * \f$\hat{f}\f$ and x to global device memory.
 * Write copy time to timer struct if MEASURED_TIMES is defined.
 *
 * @param plan data object
 */
void copyDataToDevice(cunfft_plan *plan);

/**
 * Transfer data for cunfft adjoint transformation to device memory.
 * Compute k_data and copy k_data and fast gaussian gridding data to constant
 * device memory and also * \f$\hat{f}\f$ and x to global device memory.
 * Write copy time to timer struct if MEASURED_TIMES is defined.
 *
 * @param plan data object
 */
void copyDataToDeviceAd(cunfft_plan *plan);

/**
 * Transfer data for cunfft transformation from device memory to host memory.
 * Copy result f from global device memory back to the host system. Write copy
 * time to timer struct if MEASURED_TIMES is defined.
 *
 * @param plan data object
 */
void copyDataToHost(cunfft_plan *plan);

/**
 * Transfer data for cunfft adjoint transformation from device memory to host memory.
 * Copy result f from global device memory back to the host system. Write copy
 * time to timer struct if MEASURED_TIMES is defined.
 *
 * @param plan data object
 */
void copyDataToHostAd(cunfft_plan *plan);

/**
 * Transfer data of Fourier Coefficients \f$\hat{f}\f$ from graphic card memory back to host.
 *
 * @param plan cunfft data object
 * @param arr return \f$ \hat{f}\f$
 */
void copy_f_hat_ToHost(cunfft_plan *plan,gpuComplex *arr);

/**
 * Transfer data of samples f from graphic card memory back to host.
 *
 * @param plan cunfft data object
 * @param return f
 */
void copy_f_ToHost(cunfft_plan *plan, gpuComplex *arr);

/**
 * Transfer data of oversampled vector g from graphic card memory back to host.
 *
 * @param plan cunfft data object
 */
void copy_g_ToHost(cunfft_plan *plan);

/**
 * Transfer data of nodes x from graphic card memory back to host.
 *
 * @param plan cunfft data object
 * @param arr return x
 */
void copy_x_toHost(cunfft_plan *plan, dTyp *arr);

//------------------------------------------------------------------------------
// 		CUNFFT
//------------------------------------------------------------------------------

/**
 * Call CUNFFT transformation for given dimension.
 *
 * @param plan data object
 */
void cunfft_transform(cunfft_plan *plan);

/**
 * Call CUNFFT adjoint transformation for given dimension.
 *
 * @param plan data object
 */
void cunfft_adjoint(cunfft_plan *plan);

/**
 * Wrapper: Create plan on device for CUFFT computation.
 *
 * @param dim dimension
 * @param n length of fft
 * @param h handle object for plan
 */
void cunfft_initPlan(int dim,uint_t *n,cufftHandle *h);

/** Wrapper: Destroy CUFFT plan on device.
 *
 * @param p data object cunfft
 */
void cunfft_freePlan(cunfft_plan *p);

/** Allocate data on device and init with 0.
 *
 * @param plan data object cunfft
 * */
void cunfft_initDevicePara(cunfft_plan *plan);

/**
 * Initialisation of data object CUNFFT regarding given parameters and default
 * values for flags and sigma.
 *  *
 * @param plan data object CUNFFT
 * @param dim dimension
 * @param N multi bandwith
 * @param M_total total numbers of samples
 */
void cunfft_init(cunfft_plan *plan,int dim, uint_t *N, uint_t M_total);

/**
 * Initialisation of data object CUNFFT regarding given parameters.
 *  *
 * @param plan data object CUNFFT
 * @param dim dimension
 * @param N multi bandwith
 * @param M_total total numbers of samples
 * @param n fft length
 * @param flags
 */
void cunfft_initGuru(cunfft_plan *plan,int dim, uint_t *N, uint_t M_total,
		uint_t *n, unsigned flags);

/**
 * Allocation and initialisation of data object CUNFFT regarding given parameters.
 * Precomputation for fast gaussian gridding.
 *
 * @param plan data object CUNFFT
 * */
void cunfft_initHelp(cunfft_plan *plan);

/**
 * Reset result arrays of CUNFFT f and g on host (set to 0)
 * and reset data on device.
 *
 * @param plan data object
 */
void cunfft_reinit(cunfft_plan *plan);


/**
 * Reset result arrays of CUNFFT adjoint \f$\hat{f}\f$ and g on host (set to 0)
 * and reset data on device.
 *
 * @param plan data object
 */
void cunfft_reinitAd(cunfft_plan *plan);

/**
 * Free data object
 * @param plan data object
 */
void cunfft_finalize(cunfft_plan *plan);

/**
 * Free device memory.
 * @param plan data object
 */
void cunfft_finalizeDevice(cunfft_plan *plan);

/**
 * Reset timers to 0.0.
 * @param times
 */
void resetCUNFFTTimes(NFFTTimeSpec *times);

//------------------------------------------------------------------------------
// 		CUNDFT
//------------------------------------------------------------------------------

/**
 * NDFT 1D,2d,3d computation on the graphic card.
 * @param plan data object
 */
void cundft_transform(cunfft_plan *plan);

/**
 * NDFT adjoint 1D,2d,3d computation on the graphic card.
 * @param plan data object
 */
void cundft_adjoint(cunfft_plan *plan);


//------------------------------------------------------------------------------
// 		FG_PSI COMPUTATION FUNCTION
//------------------------------------------------------------------------------
void nfft_def_fg_exp_l(cunfft_plan *plan);
/**
 * Computes FILTER_SIZE many fg_exp_l-factors for fast gaussian
 * gridding
 * @param fg_exp_l output array for computed factors
 * @param b shape parameter of window function
 * @param offset write factors on fg_exp_l[offset],...,fg_exp_l[offset+FILTER_SIZE-1]
 */
void nfft_init_fg_exp_l(double *fg_exp_l,const double b,const int offset);

//------------------------------------------------------------------------------
// 		OTHER FUNCTIONS
//------------------------------------------------------------------------------
/**
 * Set  kdata struct members.
 * @param N multi bandwith
 * @param n length of fft
 * @param b shape parameter of window function
 * @param sigma oversampling factor
 * @param dim dimension
 * @param k_cpu output struct
 */
void setProperties(uint_t *N, uint_t *n, double *b, double *sigma,int dim,kdata *k_cpu);

/**
* Reset cuda device.
*/
void resetDevice();

#endif /* CUNFFT1_H_ */
