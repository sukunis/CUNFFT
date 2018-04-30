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
 * $Id: cunfft_typedefs.h 2012-06-01 11:06:00Z sukunis $
 *
 */

/**
 * @file cunfft_typedefs.h
 * @brief typedefs CUNFFT
 */

#ifndef CUNFFT_TYPEDEFS_H_
#define CUNFFT_TYPEDEFS_H_

#include <complex.h> // include complex.h before fftw3.h to avoid cast problems
#include <cuda.h>
#include <cufft.h>

#include "config.h"
#include "timeMeasurement.h"

/** Size of filter window */
#define FILTER_SIZE ((2*CUT_OFF)+2)

/** Maximum of dimension for which transformation is implemented*/
#define MAX_DIM 3

/** Maximum of CUT-OFF Parameter*/
#define MAX_FILTER_RADIUS 50

/** FLAG defines for create CUFFT Plan */
#define CUFFT_INIT      (1U<<0)
/** FLAG define for create plan for forward and backward transformation*/
#define CUFFT_BOTH		(1U<<1)
/** FLAG define for create plan for forward transformation*/
#define CUFFT_FORW		(1U<<2)
/** FLAG define for create plan for backward transformation*/
#define CUFFT_BACKW		(1U<<3)


//------------------------------------------------------------------------------
// 		DEFINITION FOR DATATYPES
//------------------------------------------------------------------------------
#ifdef CUNFFT_DOUBLE_PRECISION
/** If CUNFFT_DOUBLE_PRECISION is defined: specified complex type is double complex */
typedef cuDoubleComplex gpuComplex;
/** If CUNFFT_DOUBLE_PRECISION is defined: specified datatyp is double*/
typedef double dTyp;
#else
/** If CUNFFT_DOUBLE_PRECISION is undefined: specified complex type is float complex */
typedef cuFloatComplex gpuComplex;
/** If CUNFFT_DOUBLE_PRECISION is defined: specified datatyp is float*/
typedef float dTyp;
#endif

#ifdef LARGE_INPUT
/** If LARGE_INPUT is defined: Integral type large enough to contain a stride
 * (what INT should have been in the first place) */
typedef ptrdiff_t uint_t;
/** If LARGE_INPUT is defined: Integral type large enough to contain a stride
 * (what INT should have been in the first place) */
typedef ptrdiff_t sint_t;
/** Print on stdout for integral type*/
#define PRINT_FORMAT "%ld"
#else
/** If LARGE_INPUT is undefined: INT*/
typedef int uint_t;
/** If LARGE_INPUT is undefined: INT*/
typedef int sint_t;
/** Print on stdout for INT type*/
#define PRINT_FORMAT "%d"
#endif

#define PRINT(V,F)  printf(#V " = " F, V )

//------------------------------------------------------------------------------
// 		DATA OBJECTS
//------------------------------------------------------------------------------
/**
 * @struct kdata For transformation properties to transfer to constant memory
 * on the device.
 */
typedef struct /*__align__(128)*/{
	/** Multi bandwith (number of Fourier coeff.)*/
	uint_t N[MAX_DIM]; //MAX_DIM*8byte
	/** FFTW length, equal to \f$\sigma N\f$, default is the power of 2*/
	uint_t n[MAX_DIM]; //MAX_DIM*8byte
	/** Shape parameter of the gaussian window function:
	 * \f$b_i=\frac{2\cdot \sigma_{i}}{2\cdot \sigma_{i} -1}\cdot \frac{m}{\pi} \f$ */
	double b[MAX_DIM]; //MAX_DIM*8byte
	/** Precomputation \f$ \sqrt{\pi b}\f$*/
	double bsqrt[MAX_DIM];//sqrt(PI*b) //MAX_DIM*8byte
	/** Precomputation \f$ \frac{\pi^2b}{n^2}\f$*/
	double fac[MAX_DIM]; //PI^2*b/n^2 //MAX_DIM*8byte
	/** Oversampling factor to reduce aliasing artefacts*/
	double sigma[MAX_DIM]; //n[i]/N[i]
}kdata;


/**
 * represent the data structure of cunfft
 */
typedef struct
{
	/** total number of Fourier coeff */
	uint_t N_total;
	/** total number of samples */
	uint_t M_total;
	/** dimension, rank */
	int d;
	/** multi bandwidth  (number of Fourier coefficients */
	uint_t *N;

	/** pointer to nodes in time/spatial domain,clustered arround 0,
	 * size is dM doubles*/
	dTyp *x;

	/** pointer to vector of Fourier coefficients, size is N_total */
	gpuComplex *f_hat;

	/** pointer to vector of samples, size is M_total*/
	gpuComplex *f;

	/** oversampling factor to reduce aliasing artefacts */
	double *sigma;

	/** FFTW length, equal to sigma*N, default is the power of 2*/
	uint_t *n;

	/** total size of FFTW (n0xn1) */
	uint_t n_total;

	/** cut-off parameter of window function, default value are:
	 *   (GAUSSIAN)
	 */
	int m;

	/** Shape parameter of the window function*/
	double *b;

	/** pointer to oversampled vector of samples, size is n_total */
	gpuComplex *g;

	/** pointer to zero-padded vector of Fourier coeff., size is n_total*/
	gpuComplex *g_hat;

	/** array of precomputed psi for fast gaussian gridding**/
	double *fg_exp_l_cpu;

	//-------------- GPU parameters ------------------------
	gpuComplex *g_gpu;/**< pointer to oversampled vector of samples on device*/
	gpuComplex *fhat_gpu;/**< pointer to vector of Fourier Coeff. on device*/
	dTyp *x_gpu;/**< pointer to nodes on device*/
	gpuComplex *f_gpu;/**<pointer to vector of samples on device*/

	cufftHandle fft_forward;/**< handle for cufftw forward transformation*/
	cufftHandle fft_backward;/**< handle for cufftw backward tansformation*/

	NFFTTimeSpec CUNFFTTimes; /**< Measured time for each step if MEASURE_TIME is set.*/

	unsigned flags;/**< Flags for precomputation and FFTW usage, default setting is: TODO*/

} cunfft_plan;



#endif  /* CUNFFT_TYPEDEFS_H_ */
