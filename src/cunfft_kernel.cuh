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
 *
 * $Id: kernel.cuh 2012-06-13 11:36:00Z sukunis $
 */

/**
 * @file cunfft1_kernel.cuh
 * @brief CUDA functions: kernel and kernel wrapper header file
 */

#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "cunfft_util.h"
#include "cunfft.h"

//#include <helper_functions.h>
//#include <helper_cuda.h>
#include "cuComplex.h"
#include <cuda_profiler_api.h>//for focused profiling

/** activate unroll loops in kernel with templates*/
#define UNROLL_INNER 0
#define TEXTURE 1





//------------------------------------------------------------------------------
//					KERNEL WRAPPER
//------------------------------------------------------------------------------
/**
 * Wrapper: Call launch configuration computation and device kernel of ROC computation
 * 1D,2D,3D.
 *
 * @param plan data object cunfft_plan
 * @return runtime of kernel function
 */
double cuKernelRolloffCorrection(cunfft_plan *plan);
/**
 * Wrapper: Call launch configuration computation and device kernel of ROC adjoint
 * computation 1D,2D,3D.
 *
 * @param plan data object cunfft_plan
 * @return runtime of kernel function
 */
double cuKernelRollOffCorrection_adjoint(cunfft_plan *plan);
/**
 * Wrapper: Call launch configuration computation and device kernel of convolution step
 * 1D, 2D, 3D.
 *
 * @param plan data object cunfft_plan
 * @return runtime of kernel function
 */
double cuConvolution(cunfft_plan *plan);

/**
 * Wrapper: Call launch configuration computation and device kernel of adjoint
 * convolution step 1D,2D,3D. (like "Fast Electrostatic halftoning" (Weikert et all))
 *
 * @param plan data object cunfft_plan
 * @return runtime of kernel function
 */
double cuConvolution_adjoint(cunfft_plan *plan);

/**
  * Wrapper for FFTW plan configuration step on the device.
  *
  * @param dim dimension
  * @param n size of g
  * @param h cufft handle object
  */
void cuFFT_createPlan(int dim,uint_t *n,cufftHandle *h);

/**
  * Wrapper: Destroy plan of FFTW.
  *
  * @param p data object cunfft_plan
  */
void cuFFT_DestroyPlan(cunfft_plan *p);

/**
 *  Wrapper for FFT computation step on the device.
 *
 * @param d_g fft input
 * @param h handle of cufft plan
 * @return runtime of kernel function
 */
double cuFFT(gpuComplex *d_g,cufftHandle h);

/**
 *  Wrapper for adjoint FFT computation step on the device.
 *
 * @param d_g fft input
 * @param h handle of cufft plan
 * @return runtime of kernel function
 */
double cuFFT_adjoint(gpuComplex *d_g,cufftHandle h);

/**
 * Wrapper: Call launch configuration computation and device kernel of NDFT 1D,2D,3D.
 *
 * @param plan data object
 */
void cuNDFT(cunfft_plan *plan);

/**
 * Wrapper: Call launch configuration computation and device kernel of NDFT adjoint 1D,2D,3D.
 *
 * @param plan data object
 */
void cuNDFT_adjoint(cunfft_plan *plan);

//------------------------------------------------------------------------------
// 		TEXTURE FUNCTIONS
//------------------------------------------------------------------------------
/**
  * Allocate texture object texData as linear memory texture.
  *
  * @param h_in source
  * @param width size of h_in
  */
void allocateTexture1DFetch(gpuComplex *h_in,int width);
/**
  * Unbind texture object and free memory.
  */
void freeTextureFetch();

//------------------------------------------------------------------------------
// 		KERNEL FUNCTIONS
//------------------------------------------------------------------------------
/**
  * Get complex element from texture object texData
  *
  * @param i element number
  */
inline __device__ gpuComplex tex2DComplex(uint_t i);


/**
  * \brief Kernel of ROC 1D computation.
  *
  * @param in  vector of Fourier coefficients \f$\hat{f}\f$ of size \f$N_{total}\f$
  * @param out (result) oversampled vector g of samples of size \f$n_{total}\f$
  * @param off_n offset of g: \f$n-\frac{N}2\f$
  * @param off_N offset of \f$\hat{f}\f$: \f$\frac{N}2\f$
  */
__global__ void kernelRollOf_1d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n,uint_t off_N);

/**
  * \brief Kernel of ROC 2D computation.
  *
  * @param in vector of Fourier coefficients \f$\hat{f}\f$ of size \f$N_{total}\f$
  * @param out (result) oversampled vector g of samples of size \f$n_{total}\f$
  * @param off_n0 offset of g: \f$n_0-\frac{N_0}2\f$,
  * @param off_n1 offset of g: \f$n_1-\frac{N_1}2\f$,
  * @param off_N0 offset of \f$\hat{f}\f$: \f$\frac{N_0}2\f$
  * @param off_N1 offset of \f$\hat{f}\f$: \f$\frac{N_1}2\f$
  */
__global__ void kernelRollOf_2d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1,uint_t off_N0, uint_t off_N1);
/**
  * \brief Kernel of ROC 3D computation.
  *
  * @param in vector of Fourier coefficients \f$\hat{f}\f$ of size \f$N_{total}\f$
  * @param out (result) oversampled vector g of samples of size \f$n_{total}\f$
  * @param off_n0 offset of g: \f$n_0-\frac{N_0}2\f$,
  * @param off_n1 offset of g: \f$n_1-\frac{N_1}2\f$,
  * @param off_n2 offset of g: \f$n_2-\frac{N_2}2\f$,
  * @param off_N0 offset of \f$\hat{f}\f$:  \f$\frac{N_0}2\f$
  * @param off_N1 offset of \f$\hat{f}\f$:  \f$\frac{N_1}2\f$
  * @param off_N2 offset of \f$\hat{f}\f$:  \f$\frac{N_2}2\f$
  */
__global__ void kernelRollOf_3d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1, uint_t off_n2,
		uint_t off_N0, uint_t off_N1, uint_t off_N2);

/**
  * \brief Kernel of ROC adjoint 1D computation.
  *
  * @param in oversampled vector g of samples of size \f$n_{total}\f$
  * @param out (result) vector of Fourier coefficients \f$\hat{f}\f$ of size \f$N_{total}\f$
  * @param off_n offset of g: \f$n-\frac{N}2\f$
  * @param off_N offset of \f$\hat{f}\f$: \f$\frac{N}2\f$
  */
__global__ void kernelRollOf_adjoint_1d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n, uint_t off_N);

/**
  * Kernel of ROC 2D adjoint computation.
  * @param in oversampled vector g of samples of size \f$n_{total}\f$
  * @param out (result) vector of Fourier coefficients \f$\hat{f}\f$ of size \f$N_{total}\f$
  * @param off_n0 offset of g: \f$n_0-\frac{N_0}2\f$,
  * @param off_n1 offset of g: \f$n_1-\frac{N_1}2\f$,
  * @param off_N0 offset of \f$\hat{f}\f$: \f$\frac{N_0}2\f$
  * @param off_N1 offset of \f$\hat{f}\f$: \f$\frac{N_1}2\f$
  */
__global__ void kernelRollOf_adjoint_2d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1,uint_t off_N0,uint_t off_N1 );

/**
  * Kernel of ROC adjoint 3D computation.
  *
  * @param in oversampled vector g of samples of size \f$n_{total}\f$
  * @param out (result) vector Fourier coefficients \f$\hat{f}\f$ of size \f$N_{total}\f$
  * @param off_n0 offset of g: \f$n_0-\frac{N_0}2\f$,
  * @param off_n1 offset of g: \f$n_1-\frac{N_1}2\f$,
  * @param off_n2 offset of g: \f$n_2-\frac{N_2}2\f$,
  * @param off_N0 offset of \f$\hat{f}\f$:  \f$\frac{N_0}2\f$
  * @param off_N1 offset of \f$\hat{f}\f$:  \f$\frac{N_1}2\f$
  * @param off_N2 offset of \f$\hat{f}\f$:  \f$\frac{N_2}2\f$
  */
__global__ void kernelRollOf_adjoint_3d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1, uint_t off_n2,
		uint_t off_N0, uint_t off_N1, uint_t off_N2);


//--------------------------------- CONVOLUTION --------------------------------
/**
  * \brief Kernel for convolution 1D with gaussian window. Oversampled vector g of
  * samples is given as texture.
  *
  * @param x_in vector of nodes in time/spatial domain clustered arround 0.
  * 		Size is \f$dM_{total}\f$
  * @param f vector of samples. Size is \f$M_{total}\f$
  * @param width size of f
  */
#ifdef TEXTURE
__global__ void kernelConvWithGauss_1d(const dTyp *x_in,gpuComplex *f,
		uint_t width);
#else
__global__ void kernelConvWithGauss_1d(const gpuComplex *g,const dTyp *x_in,gpuComplex *f,
		uint_t width);
#endif

/**
  * Kernel for convolution 2D with gaussian window.Oversampled vector g of
  * samples is given as texture. One thread compute the filter for one x.
  *
  * @param x_in vector of nodes in time/spatial domain clustered arround 0.
  * 	Size is \f$dM_{total}\f$
  * @param f vector of samples. Size is \f$M_{total}\f$.
  * @param width size of f
  */
#ifdef TEXTURE
__global__ void kernelConvWithGauss_2d(const dTyp *x_in,gpuComplex *f,
		uint_t width);
#else
__global__ void kernelConvWithGauss_2d(const gpuComplex *g,const dTyp *x_in,gpuComplex *f,
		uint_t width);
#endif
/**
  * \brief Kernel for convolution 3D with gaussian window. Oversampled vector g of
  * samples is given as texture.
  *
  * @param x_in vector of nodes in time/spatial domain clustered arround 0.
  * 	Size is \f$dM_{total}\f$
  * @param f vector of samples. Size is \f$M_{total}\f$.
  * @param width size of f
  * @param planeSize size of plane of g (n2*n1)
  */
#ifdef TEXTURE
__global__ void kernelConvWithGauss_3d(const dTyp *x_in,gpuComplex *f,
		uint_t width, uint_t planeSize);
#else
__global__ void kernelConvWithGauss_3d(const gpuComplex *g,const dTyp *x_in,gpuComplex *f,
		uint_t width, uint_t planeSize);
#endif

#ifdef COM_FG_PSI
/**
  * Kernel of adjoint convolution 1D with gaussian window and fast gaussian gridding.
  *
  * @param g oversampled vector of samples. Size is \f$n_{total}\f$.
  * @param x_in nodes in time/spatial domain clustered arround 0.
  * 	Size is \f$dM_{total}\f$
  * @param m cut-off parameter for gaussian window
  * @param f vector of samples. Size is \f$M_{total}\f$.
  * @param width M_total
  */
__global__ void kernelConvWithGauss_adjoint_1d( gpuComplex *g, dTyp *x_in,
		const gpuComplex* __restrict f, uint_t width);

/**
  * Kernel of adjoint convolution 2D with gaussian window and fast gaussian gridding.
  *
  * @param g oversampled vector of samples. Size is \f$n_{total}\f$.
  * @param x_in nodes in time/spatial domain clustered arround 0.
  * 	Size is \f$dM_{total}\f$
  * @param width \f$M_{total}\f$
  *
  * Given as Texture:
  * @param f vector of samples. Size is \f$M_{total}\f$.
  */
__global__ void kernelConvWithGauss_adjoint_2d( gpuComplex *g, dTyp *x_in,
		uint_t width);

/**
  * Kernel of adjoint convolution 3D with gaussian window for fast gaussian gridding.
  *
  * @param g oversampled vector of samples. Size is \f$n_{total}\f$.
  * @param x_in nodes in time/spatial domain clustered arround 0.
  * 	Size is \f$dM_{total}\f$
  * @param width \f$M_{total}\f$
  *
  * Given as Texture:
  * @param f vector of samples. Size is \f$M_{total}\f$.
  */
__global__ void kernelConvWithGauss_adjoint_3d( gpuComplex *g, dTyp *x_in,
		uint_t width);

#else
// CONVOLUTION ADJOINT
/**
  * Kernel of adjoint convolution 1D with gaussian window.
  *
  * @param f vector of samples. Size is \f$M_{total}\f$.
  * @param g result oversampled vector of samples. Size is \f$n_{total}\f$.
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$
  * @param width \f$M_{total}\f$
  */
__global__ void kernelConvWithGauss_adjoint_1d(gpuComplex *f,gpuComplex *g,
		dTyp *x_in,uint_t width);

/**
  * Kernel of adjoint convolution 2D with gaussian window.
  *
  * @param f vector of samples. Size is \f$M_{total}\f$.
  * @param g result oversampled vector of samples. Size is \f$n_{total}\f$.
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$
  * @param width \f$M_{total}\f$
  */
__global__ void kernelConvWithGauss_adjoint_2d(gpuComplex *f,gpuComplex *g,
		dTyp *x_in,uint_t width);

/**
  * Kernel of adjoint convolution 3D with gaussian window.
  *
  * @param f vector of samples. Size is \f$M_{total}\f$.
  * @param g result oversampled vector of samples. Size is \f$n_{total}\f$.
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$
  * @param width \f$M_{total}\f$
  */
__global__ void kernelConvWithGauss_adjoint_3d(gpuComplex *f,gpuComplex *g,
		dTyp *x_in,uint_t width);

#endif

//------------  NDFT -----------------------------------------------------------
/**
  * Kernel for NDFT transformation 1D.
  *
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
  * @param f_hat vector of Fourier coefficients. Size is \f$N_{total}\f$.
  * @param f result vector of samples. Size is \f$M_{total}\f$.
  * @param M_total size of f
  * @param N_total size of \f$\hat{f}\f$
  */
__global__ void kernelNDFT_1d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total);

/**
 * Helper function for kernelNDFT_2d and kernelNDFT_3d.
 * Computes NDFT for specific node j.
 *
 * @param j specific node
 * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
 * @param f_hat vector of Fourier coefficients. Size is \f$N_{total}\f$.
 * @param N_total size of \f$\hat{f}\f$
 * @param kd temp data array (size=dim)
 * @param x temp data array (size=dim)
 * @param Omega temp data array (size=dim)
 * @param d dimension
 */
__device__ gpuComplex NDFT_forElem(uint_t j,dTyp *x_in,gpuComplex *f_hat,uint_t N_total,
		sint_t *kd,double *x,double *Omega, int d);

/**
  * Kernel for NDFT transformation 2D.
  *
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
  * @param f_hat vector of Fourier coefficients. Size is \f$N_{total}\f$.
  * @param f result vector of samples. Size is \f$M_{total}\f$.
  * @param M_total size of f
  * @param N_total size of \f$\hat{f}\f$
  */
__global__ void kernelNDFT_2d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total);

/**
  * Kernel for NDFT transformation 3D.
  *
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
  * @param f_hat vector of Fourier coefficients. Size is \f$N_{total}\f$.
  * @param f result vector of samples. Size is \f$M_{total}\f$.
  * @param M_total size of f
  * @param N_total size of \f$\hat{f}\f$
  */
__global__ void kernelNDFT_3d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total);


//-------------- NDFT ADJOINT --------------------------------------------------

/**
  * Kernel for NDFT adjoint transformation 1D.
  *
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
  * @param f_hat vector of Fourier coefficients. Size is \f$N_{total}\f$.
  * @param f result vector of samples. Size is \f$M_{total}\f$.
  * @param M_total size of f
  * @param N_total size of \f$\hat{f}\f$
  */
__global__ void kernelNDFT_adjoint_1d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total);

/**
 * Helper function for kernelNDFT_adjoint_2d and kernelNDFT_adjoint_3d.
 * Computes NDFT for specific node j.
 *
 * @param k_L specific node
 * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
 * @param f
 * @param M_total size of f
 * @param k temp data array (size=dim)
 * @param d dimension
 */
__device__ gpuComplex NDFT_adjoint_forElem(uint_t k_L,dTyp *x_in,gpuComplex *f,
		uint_t M_total,sint_t *k, int d);

/**
  * Kernel for NDFT adjoint transformation 2D.
  *
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
  * @param f_hat vector of Fourier coefficients. Size is \f$N_{total}\f$.
  * @param f result vector of samples. Size is \f$M_{total}\f$.
  * @param M_total size of f
  * @param N_total size of \f$\hat{f}\f$
  */
__global__ void kernelNDFT_adjoint_2d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total);

/**
  * Kernel for NDFT adjoint transformation 3D.
  *
  * @param x_in nodes in time/spatial domain clustered arround 0. Size is \f$dM_{total}\f$.
  * @param f_hat vector of Fourier coefficients. Size is \f$N_{total}\f$.
  * @param f result vector of samples. Size is \f$M_{total}\f$.
  * @param M_total size of f
  * @param N_total size of \f$\hat{f}\f$
  */
__global__ void kernelNDFT_adjoint_3d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total);


//------------------------------------------------------------------------------
// 		HELPER FUNCTION CONSTANT MEMORY
//------------------------------------------------------------------------------

/**
 * Copies input of kind kdata with a certain size to constant memory on the
 * device (k_gpu).
 *
 * @param k_cpu data structur to copy
 */
void copyProperties(kdata *k_cpu);

/**
 *  Copies input of kind double to constant memory on the device (fg_exp_l_gpu).
 *
 *  @param fg_exp_l_cpu data for copy
 *  @param size_fg size of data fg
 */
void setPREexp(double *fg_exp_l_cpu,int size_fg);

//------------------------------------------------------------------------------
// 		GRID CONFIGURATION
//------------------------------------------------------------------------------

/**
 * Computes launch configuration 2D and 3D for ROC step.
 *
 * @param blockDim result block dimension
 * @param gridDim result grid dimension
 * @param size1 width
 * @param size2 heigth
 * @return 0 if size beyond the maximum computation grid, else 1.
 */
int getComputationGridParams_Matrix(dim3 *blockDim, dim3 *gridDim, uint_t size1,
		uint_t size2);

/**
 * Computes launch configuration for array handling.
 *
 * @param blockDim result block dimension
 * @param gridDim result grid dimension
 * @param sizeData width
 * @return 0 if size beyond the maximum computation grid, else 1.
 */
int getComputationGridParams_Array(dim3 *blockDim, dim3 *gridDim,
		uint_t sizeData);

//------------------------------------------------------------------------------
// 		COMPLEX DATA TYPE ON DEVICE
//------------------------------------------------------------------------------

/** Set given complex number to 0*/
__device__ static __forceinline__ void resetComplex(gpuComplex *a)
{
#ifdef CUNFFT_DOUBLE_PRECISION
	a->x=0.0;
	a->y=0.0;
#else
	a->x=0.0f;
	a->y=0.0f;
#endif
}

/** return a zero set complex number*/
__device__ static __forceinline__ gpuComplex newComplexZero(void){
#ifdef CUNFFT_DOUBLE_PRECISION
	return make_cuDoubleComplex(0.0,0.0);
#else
	return make_cuFloatComplex(0.f,0.f);
#endif
}

/** Create new complex number with given value for im and re */
__device__ static __forceinline__ gpuComplex newComplex(dTyp re, dTyp im){
	gpuComplex a=newComplexZero();
	a.x = re;
	a.y=im;
	return a;
}

/** Multiply given complex number with given scalar*/
__device__ static __forceinline__ gpuComplex complexMul_scalar(gpuComplex f, dTyp a){
	f.x = f.x*a;
	f.y = f.y*a;
	return f;
}

/** Multiply given complex numbers with each other. Return result.*/
__device__ static __forceinline__ gpuComplex complexMul(gpuComplex a, gpuComplex b)
{

	gpuComplex res;
	res.x = a.x*b.x - a.y*b.y;
	res.y = a.x*b.y + a.y*b.x;
	return res;
}

/** Add given complex number with each other. Return result.*/
__device__ static __forceinline__ gpuComplex complexAdd(gpuComplex a, gpuComplex b)
{
	a.x += b.x;
	a.y += b.y;
	return a;
}


#endif /* KERNEL_CUH_ */
