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
 * $Id: kernel.cu 2012-06-13 11:36:00Z sukunis $
 */

/**
 * @file cunfft_kernel.cu
 * @brief CUDA functions: kernel and kernel wrapper source file
 */
#include <assert.h>
#include <cuda_runtime.h>

#include "cunfft_kernel.cuh"



//------------------------------------------------------------------------------
//		DEFINES AND PARAMETER DEFINITIONS
//------------------------------------------------------------------------------

#ifdef CUNFFT_DOUBLE_PRECISION
	/** Define Atomic add for double precision*/
	#define ATOMIC_ADD(adr, val) atomicAddDouble(adr,val)
	/** texture for double Complex data*/
	typedef int4 TEXTURE_TYPE;
	/** Fetch double complex data from texture*/
	#define TEX2COMPLEX(pos) ( {\
		int4 g=tex1Dfetch(texData,pos);\
		newComplex(__hiloint2double(g.y,g.x),__hiloint2double(g.w,g.z));\
	})


#else
	/** Define Atomic add for float precision*/
	#define ATOMIC_ADD(adr,val) atomicAddFloat(adr,val)
	/** Texture for float Complex data*/
	typedef float2 TEXTURE_TYPE;
	/** Fetch float complex data from texture*/
	#define TEX2COMPLEX(pos) tex1Dfetch(texData,pos)
#endif
#define TEX2COMPLEX2(pos) g[pos]


/** Texture data structur */
texture<TEXTURE_TYPE, 1, cudaReadModeElementType> texData;

/** device data structure to bind to texture*/
gpuComplex *d_dCData;

/** Data struct on device constant memory*/
__device__ __constant__ kdata k_gpu;
/** Size of precomputations for fast gaussian gridding*/
#define FG_EXP_LENGTH MAX_DIM*FILTER_SIZE//GET_FILTER_LENGTH(MAX_FILTER_RADIUS)
/** Data structure for precomputations save on constant memory for fast
 * gaussian gridding*/
__device__ __constant__ double fg_exp_l_gpu[FG_EXP_LENGTH];

/** return position of element in 2D in 1d array structure*/
#define POS2D(i1,i0,width1) ((i1)+(i0)*(width1))
/** return position of element in 2D in 1d array structure*/
#define POS2D_C(x,colOff) ((x)+(colOff))
/** return position of element in 3D in 1d array structure*/
#define POS3D(i2,i1,i0,width1,width2) (POS2D(i2,POS2D(i1,i0,width1),width2))
/** return position of element in 3D in 1d array structure*/
#define POS3D_C(x,colOff,planeOff) (POS2D_C((x),(colOff))+(planeOff))



//------------------------------------------------------------------------------
//			ATOMIC FUNCTIONS
//------------------------------------------------------------------------------
/**
 * Atomic function for realise a+=val for double values on device.
 * Used to prevent race conditions which are common problems in mulithreaded
 * applications.
 * @param address address of value on witch result should be stored
 * @param val value for add to address
 *
 * @return
 */
__device__ __forceinline__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/**
 * Atomic function for realise a+=val for floating values on device.
 * Used to prevent race conditions which are common problems in mulithreaded
 * applications.
 * @param address address of value on witch result should be stored
 * @param val value for add to address
 */
__device__ __forceinline__ float atomicAddFloat(float* address, float val)
{
	int oldval, newval, readback;

	oldval = __float_as_int(*address);
	newval = __float_as_int(__int_as_float(oldval) + val);
	while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval)
	{
		oldval = readback;
		newval = __float_as_int(__int_as_float(oldval) + val);
	}
	return __int_as_float(oldval);
}


//------------------------------------------------------------------------------
// 		HELPER FUNCTION CONSTANT MEMORY
//------------------------------------------------------------------------------

void copyProperties(kdata *k_cpu)
{
	cudaVerify(cudaMemcpyToSymbol(k_gpu,k_cpu,(sizeof(uint_t)*2+sizeof(double)*4)*MAX_DIM,
			0,cudaMemcpyHostToDevice));
}

void setPREexp(double *fg_exp_l_cpu,int size_fg)
{
	cudaVerify(cudaMemcpyToSymbol(fg_exp_l_gpu,fg_exp_l_cpu,size_fg));
}


//------------------------------------------------------------------------------
// 		CONV: DEVICE HELPER FUNCTIONS
//------------------------------------------------------------------------------


/**
 * \brief Computes \f$\varphi\f$ for convolution step with gaussian window.
 *
 * Change \f$ \varphi = \frac{e^{\frac{-x^2}{b}}}{\sqrt{\pi\cdot b}},\quad
 * b=\frac{2\cdot\sigma}{2 \cdot\sigma-1} \cdot \frac{m}{\pi}\f$ to
 * \f$ \varphi=e^{-x^2\cdot b' \cdot \pi}\cdot\sqrt{b'}, \quad
 * b'= \frac{2\cdot\sigma-1}{2\cdot\sigma\cdot m}\f$
 */
__device__ __forceinline__ double getPHI(dTyp x, double sigma)
{
	double b=(2.0*sigma-1.0)/(2.0*sigma*CUT_OFF);
	return (exp(-(x*x)*b*M_PI) * sqrt(b));
}



/** Computes upper bound of gaussian window for given xj regarding to fftw result g.
 * @param n size of fftw result g in one direction
 * @param xj position in spatial domain
 * @return index for upper bound in given direction
 * */
__device__ __forceinline__ uint_t cunfft_u(const uint_t n, dTyp xj)
{
#ifdef LARGE_INPUT
	return llrint(xj*n)-signbit(xj)-CUT_OFF;
#else
	return lrint(xj*n)-signbit(xj)-CUT_OFF;
#endif
}



//------------------------------------------------------------------------------
// 		ROC
//------------------------------------------------------------------------------

/** Computes 1-periodic shifted window function \f$\tilde\varphi\f$ for 2D.
 * Note :\f$e^{-(\frac{\pi \cdot x}{n})^2\cdot b} \Rightarrow
 * e^{\pi^2 \cdot \frac{b}n \cdot x^2}\f$
 * @param k1 index of first dimension
 * @param fac0 \f$\frac{\pi^2\cdot b_0}{n_0^2}\f$ of first dimension
 * @param k2 index of second dimension
 * @param fac1 \f$\frac{\pi^2\cdot b_1}{n_1^2}\f$ of second dimension
 */
__device__ inline dTyp getPhiHutInv2(sint_t k1,double fac0,sint_t k2, double fac1)
{
#ifdef CUNFFT_DOUBLE_PRECISION
	return (double)exp(fac0*k1*k1+fac1*k2*k2);
#else
	return expf(fac0*k1*k1+fac1*k2*k2);
#endif
}

/** Computes 1-periodic shifted window function \f$\tilde\varphi\f$ for 3D.
 * Note :\f$e^{-(\frac{\pi \cdot x}{n})^2\cdot b} \Rightarrow
 * e^{\pi^2 \cdot \frac{b}n \cdot x^2}\f$
 * @param k1 index of first dimension
 * @param fac0 \f$\frac{\pi^2\cdot b_0}{n_0^2}\f$ of first dimension
 * @param k2 index of second dimension
 * @param fac1 \f$\frac{\pi^2\cdot b_1}{n_1^2}\f$ of second dimension
 * @param k3 index of third dimension
 * @param fac2 \f$\frac{\pi^2\cdot b_2}{n_2^2}\f$ of third dimension
 */
__device__ inline dTyp getPhiHutInv3(sint_t k1,double fac0,sint_t k2, double fac1,
		sint_t k3, double fac2)
{
#ifdef CUNFFT_DOUBLE_PRECISION
	return (double)exp(fac0*k1*k1+fac1*k2*k2+fac2*k3*k3);
#else
	return expf(fac0*k1*k1+fac1*k2*k2+fac2*k3*k3);
#endif
}




//------------------------------------------------------------------------------
// 		ROC: 1D KERNEL
//------------------------------------------------------------------------------
/**
 * Compute ROC 1D on the graphic card. Divide input array of size N in two
 * subarrays of size \f$\frac{N}2\f$. Swap arrays and zerro padding of size N
 * between the arrays. Every thread shift one element of each subarray.
 * A1 from left to right, A2 from right to left.
 * (A1xxx|A2yyy) -> (A2yyy|00000000|A1xxx)
 */
__global__ void kernelRollOf_1d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n,uint_t off_N)
{
	// for statement necessary if size exits launch configuration
	for(uint_t i0=blockIdx.x * blockDim.x + threadIdx.x; i0<off_N;
			i0+=blockDim.x*gridDim.x){

		gpuComplex A1=in[i0];
		gpuComplex A2=in[i0+off_N];

		out[i0+off_n] =  complexMul_scalar(A1,
				exp(((i0-off_N)*(i0-off_N)*k_gpu.fac[0])));//A1
		out[i0] = complexMul_scalar(A2,exp((i0*i0*k_gpu.fac[0])));//A2
	}
}

/**
 * Compute Adjoint ROC 1D on the graphic card. Divide input array of size N in
 * three subarrays, two of size \f$\frac{N}2\f$, one of size N.
 * Every thread shift one element of each subarray of size \f$\frac{N}2\f$. Ignore array
 * of size N.
 * A1 from left to right, A2 from right to left.
 * (A1xxx|........|A2yyy)-> (A2yyy|A1xxx)
 */
__global__ void kernelRollOf_adjoint_1d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n, uint_t off_N )
{
	for(uint_t i0=blockIdx.x * blockDim.x + threadIdx.x; i0<off_N;
				i0+=blockDim.x*gridDim.x){
		gpuComplex A1=in[i0];
		gpuComplex A2=in[i0+off_n];

		out[i0+off_N] =  complexMul_scalar(A1,exp(i0*i0*k_gpu.fac[0]));//A1
		out[i0] = complexMul_scalar(A2,exp((i0-off_N)*(i0-off_N)*k_gpu.fac[0]));//A2
	}
}



//------------------------------------------------------------------------------
// 		ROC: 2D KERNEL
//------------------------------------------------------------------------------
/**
 * Compute ROC 2D on the graphic card. Divide Input f_hat in 4 submatrices.
 * Every thread shift one element of each submatrix and write it to g.
 */
__global__ void kernelRollOf_2d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1,uint_t off_N0, uint_t off_N1)
{
	for(uint_t i0=blockIdx.y * blockDim.y + threadIdx.y; i0< off_N0;
			i0+=blockDim.y*gridDim.y){
		for(uint_t i1=blockIdx.x * blockDim.x + threadIdx.x; i1<off_N1;
				i1+=blockDim.x*gridDim.x){


			gpuComplex A1=in[POS2D(i1,i0,k_gpu.N[1])];
			gpuComplex A2=in[POS2D(i1+off_N1,i0,k_gpu.N[1])];

			gpuComplex A3=in[POS2D(i1,i0+off_N0,k_gpu.N[1])];
			gpuComplex A4=in[POS2D(i1+off_N1,i0+off_N0,k_gpu.N[1])];

			out[POS2D(i1,i0+off_n0,k_gpu.n[1])] = complexMul_scalar(A2,
					getPhiHutInv2(i0-off_N0,k_gpu.fac[0],i1,k_gpu.fac[1]));
			out[POS2D(i1+off_n1,i0+off_n0,k_gpu.n[1])] = complexMul_scalar(A1,
					getPhiHutInv2(i0-off_N0,k_gpu.fac[0],i1-off_N1,k_gpu.fac[1]));
			out[POS2D(i1,i0,k_gpu.n[1])] = complexMul_scalar(A4,
					getPhiHutInv2(i0,k_gpu.fac[0],i1,k_gpu.fac[1]));
			out[POS2D(i1+off_n1,i0,k_gpu.n[1])] = complexMul_scalar(A3,
					getPhiHutInv2(i0,k_gpu.fac[0],i1-off_N1,k_gpu.fac[1]));

		}
	}
}

/**
 * Compute Adjoint ROC 2D on the graphic card. Turning back operation to ROC 2D.
 * Every thread shift one element of each submatrix and write it to f_hat.
*/
__global__ void kernelRollOf_adjoint_2d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1, uint_t off_N0, uint_t off_N1)
{
	for(uint_t i0=blockIdx.y * blockDim.y + threadIdx.y; i0< off_N0;
			i0+=blockDim.y*gridDim.y){
		for(uint_t i1=blockIdx.x * blockDim.x + threadIdx.x; i1<off_N1;
				i1+=blockDim.x*gridDim.x){

			gpuComplex A1=in[POS2D(i1,i0,k_gpu.n[1])];
			gpuComplex A2=in[POS2D(i1+off_n1,i0,k_gpu.n[1])];

			gpuComplex A3=in[POS2D(i1,i0+off_n0,k_gpu.n[1])];
			gpuComplex A4=in[POS2D(i1+off_n1,i0+off_n0,k_gpu.n[1])];

			out[POS2D(i1,i0,k_gpu.N[1])] = complexMul_scalar(A4,
					getPhiHutInv2(i0-off_N0,k_gpu.fac[0],i1-off_N1,k_gpu.fac[1]));
			out[POS2D(i1+off_N1,i0,k_gpu.N[1])] = complexMul_scalar(A3,
					getPhiHutInv2(i0-off_N0,k_gpu.fac[0],i1,k_gpu.fac[1]));

			out[POS2D(i1,i0+off_N0,k_gpu.N[1])] = complexMul_scalar(A2,
					getPhiHutInv2(i0,k_gpu.fac[0],i1-off_N1,k_gpu.fac[1]));
			out[POS2D(i1+off_N1,i0+off_N0,k_gpu.N[1])] = complexMul_scalar(A1,
					getPhiHutInv2(i0,k_gpu.fac[0],i1,k_gpu.fac[1]));
		}
	}
}


//------------------------------------------------------------------------------
//		ROC 3D Kernel
//------------------------------------------------------------------------------

/**
 * Compute ROC 3D on the graphic card. Divide every input plane in 4 submatrices.
 * Every thread loads on element of each submatrix in each plane and write it
 * to out.
 */
__global__ void kernelRollOf_3d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1, uint_t off_n2,
		uint_t off_N0, uint_t off_N1, uint_t off_N2)
{

	for(uint_t i1=blockIdx.y * blockDim.y + threadIdx.y; i1< off_N1;
			i1+=blockDim.y*gridDim.y){
		for(uint_t i2=blockIdx.x * blockDim.x + threadIdx.x; i2<off_N2;
				i2+=blockDim.x*gridDim.x){
			for(uint_t i0=0; i0 < off_N0; i0++){
				// first Front goes behind
				gpuComplex A1=in[POS3D(i2,i1,i0,k_gpu.N[1],k_gpu.N[2])];
				gpuComplex A3=in[POS3D(i2,i1+off_N1,i0,k_gpu.N[1],k_gpu.N[2])];
				gpuComplex A2=in[POS3D(i2+off_N2,i1,i0,k_gpu.N[1],k_gpu.N[2])];
				gpuComplex A4=in[POS3D(i2+off_N2,i1+off_N1,i0,k_gpu.N[1],k_gpu.N[2])];

				out[POS3D(i2+off_n2,i1+off_n1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])] =
						complexMul_scalar(A1,
								getPhiHutInv3(i1-off_N1,k_gpu.fac[1],i2-off_N2,
										k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));
				out[POS3D(i2+off_n2,i1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])] =
						complexMul_scalar(A3,
								getPhiHutInv3(i1,k_gpu.fac[1],i2-off_N2,
								k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));
				out[POS3D(i2,i1+off_n1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])] =
						complexMul_scalar(A2,
								getPhiHutInv3(i1-off_N1,k_gpu.fac[1],i2,
								k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));
				out[POS3D(i2,i1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])] =
						complexMul_scalar(A4,
								getPhiHutInv3(i1,k_gpu.fac[1],i2,
								k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));

				// Front behind comes forward
				A1=in[POS3D(i2,i1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])];//A5
				A3=in[POS3D(i2,i1+off_N1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])];//A7
				A2=in[POS3D(i2+off_N2,i1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])];//A6
				A4=in[POS3D(i2+off_N2,i1+off_N1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])];//A8

				out[POS3D(i2+off_n2,i1+off_n1,i0,k_gpu.n[1],k_gpu.n[2])] =
						complexMul_scalar(A1,
								getPhiHutInv3(i1-off_N1,k_gpu.fac[1],i2-off_N2,
										k_gpu.fac[2],i0,k_gpu.fac[0]));
				out[POS3D(i2+off_n2,i1,i0,k_gpu.n[1],k_gpu.n[2])] =
						complexMul_scalar(A3,
								getPhiHutInv3(i1,k_gpu.fac[1],i2-off_N2,
										k_gpu.fac[2],i0,k_gpu.fac[0]));
				out[POS3D(i2,i1+off_n1,i0,k_gpu.n[1],k_gpu.n[2])] =
						complexMul_scalar(A2,
								getPhiHutInv3(i1-off_N1,k_gpu.fac[1],i2,
										k_gpu.fac[2],i0,k_gpu.fac[0]));
				out[POS3D(i2,i1,i0,k_gpu.n[1],k_gpu.n[2])] = complexMul_scalar(A4,
						getPhiHutInv3(i1,k_gpu.fac[1],i2,
								k_gpu.fac[2],i0,k_gpu.fac[0]));
			}
		}
	}
}

/**
 * Compute Adjoint ROC 3D on the graphic card. Turning back operation to ROC 3D.
 * (see comments above)
 */
__global__ void kernelRollOf_adjoint_3d(const gpuComplex *in, gpuComplex *out,
		uint_t off_n0, uint_t off_n1, uint_t off_n2,
		uint_t off_N0, uint_t off_N1, uint_t off_N2)
{
	for(uint_t i1=blockIdx.y * blockDim.y + threadIdx.y; i1< off_N1;
			i1+=blockDim.y*gridDim.y){
		for(uint_t i2=blockIdx.x * blockDim.x + threadIdx.x; i2<off_N2;
				i2+=blockDim.x*gridDim.x){
			for(uint_t i0=0; i0 < off_N0; i0++){
				// first Front goes behind
				gpuComplex A1=in[POS3D(i2,i1,i0,k_gpu.n[1],k_gpu.n[2])];
				gpuComplex A2=in[POS3D(i2+off_n2,i1,i0,k_gpu.n[1],k_gpu.n[2])];
				gpuComplex A3=in[POS3D(i2,i1+off_n1,i0,k_gpu.n[1],k_gpu.n[2])];
				gpuComplex A4=in[POS3D(i2+off_n2,i1+off_n1,i0,k_gpu.n[1],k_gpu.n[2])];

				out[POS3D(i2,i1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A4,getPhiHutInv3(i1-off_N1,k_gpu.fac[1],
								i2-off_N2,k_gpu.fac[2],i0,k_gpu.fac[0]));
				out[POS3D(i2+off_N2,i1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A3,getPhiHutInv3(i1-off_N1,k_gpu.fac[1],
								i2,k_gpu.fac[2],i0,k_gpu.fac[0]));
				out[POS3D(i2,i1+off_N1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A2,getPhiHutInv3(i1,k_gpu.fac[1],
								i2-off_N2,k_gpu.fac[2],i0,k_gpu.fac[0]));
				out[POS3D(i2+off_N2,i1+off_N1,i0+off_N0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A1,getPhiHutInv3(i1,k_gpu.fac[1],i2,
								k_gpu.fac[2],i0,k_gpu.fac[0]));

				// Front behind comes forward
				A1=in[POS3D(i2,i1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])];//A5
				A2=in[POS3D(i2+off_n2,i1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])];//A6
				A3=in[POS3D(i2,i1+off_n1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])];//A7
				A4=in[POS3D(i2+off_n2,i1+off_n1,i0+off_n0,k_gpu.n[1],k_gpu.n[2])];//A8

				out[POS3D(i2,i1,i0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A4,getPhiHutInv3(i1-off_N1,k_gpu.fac[1],
								i2-off_N2,k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));
				out[POS3D(i2+off_N2,i1,i0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A3,getPhiHutInv3(i1-off_N1,k_gpu.fac[1],
								i2,k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));
				out[POS3D(i2,i1+off_N1,i0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A2,getPhiHutInv3(i1,k_gpu.fac[1],
								i2-off_N2,k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));
				out[POS3D(i2+off_N2,i1+off_N1,i0,k_gpu.N[1],k_gpu.N[2])] =
						complexMul_scalar(A1,getPhiHutInv3(i1,k_gpu.fac[1],i2,
								k_gpu.fac[2],i0-off_N0,k_gpu.fac[0]));
			}
		}
	}
}

//------------------------------------------------------------------------------
// 		CONVOLUTION
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 		CONV: FG_PSI PRECOMPUTATION FUNCTION
//------------------------------------------------------------------------------

/** Precompute factors of \f$\varphi\f$.
 *
 * For a fix node \f$x_j\f$ the evaluation of
 * \f$\varphi(x_j-\frac{l'}n), l'\in I_{n,m}(x_j) )\f$ can be reduced by the
 * splitting \f$ \sqrt{\pi b} \varphi \left( x_j - \frac{l'}{n} \right )=
 * e^{-\frac{(nx_j-l')^2}{b}}= e^{-\frac{(nx_j-u)^2}b}
 * \left ( e^{-\frac{2(nx_j-u)}b} \right )^l
 * e^{-\frac{l^2}b}
 * \f$ where \f$ u= min I_{n,m}(x_j), \quad l=0,...,2m\f$
 *
 * @param fg_psijFac result arrays with factors of \f$\varphi\f$
 * @param c_ju \f$ nx_j-u \f$
 * @param d dimension
 */
__device__ double getPSI_factors(double *fg_psijFac,double c_ju, int d)
{
	fg_psijFac[0]= getPHI(c_ju,k_gpu.sigma[d]);
	fg_psijFac[1]=exp(2.0*c_ju/k_gpu.b[d]);
	fg_psijFac[2]=1.0;

	return fg_psijFac[0];
}

/** Recompute factors of \f$\varphi\f$ iterativ.*/
__device__ inline double updatePSI_factors(double *fg_psijFac, int l)
{
	fg_psijFac[2]  =fg_psijFac[2]*fg_psijFac[1];
	return (fg_psijFac[0]*fg_psijFac[2]*fg_exp_l_gpu[l]);
}


//------------------------------------------------------------------------------
//					CONV: 1D KERNEL
//------------------------------------------------------------------------------

#ifdef COM_FG_PSI
#ifdef TEXTURE
__global__ void kernelConvWithGauss_1d(const dTyp *x_in,gpuComplex *f, uint_t width)
{
	gpuComplex sum=newComplexZero();
	uint_t u0=0;
	double fg_psij_fac[3];
	double psi0;

	for(uint_t fpos = blockDim.x * blockIdx.x + threadIdx.x; fpos < width;
			fpos+=blockDim.x*gridDim.x){
		dTyp pos0=x_in[fpos];
		u0 = cunfft_u(k_gpu.n[0],pos0);
		psi0=getPSI_factors(fg_psij_fac,k_gpu.n[0]*pos0-(dTyp)u0,0);

#pragma unroll 13 // m_min=1
		for(int l0= 0; l0<FILTER_SIZE;++l0){
			sum=complexAdd(sum,
					complexMul_scalar(TEX2COMPLEX((u0+l0)&(k_gpu.n[0]-1)),psi0));
			psi0 = updatePSI_factors(fg_psij_fac,l0+1);
		}
		f[fpos]=sum;
	}
}
#else
__global__ void kernelConvWithGauss_1d(const gpuComplex *g,const dTyp *x_in,gpuComplex *f, uint_t width)
{
	gpuComplex sum=newComplexZero();
	uint_t u0=0;
	double fg_psij_fac[3];
	double psi0;

	for(uint_t fpos = blockDim.x * blockIdx.x + threadIdx.x; fpos < width;
			fpos+=blockDim.x*gridDim.x){
		dTyp pos0=x_in[fpos];
		u0 = cunfft_u(k_gpu.n[0],pos0);
		psi0=getPSI_factors(fg_psij_fac,k_gpu.n[0]*pos0-(dTyp)u0,0);

#pragma unroll 13 // m_min=1
		for(int l0= 0; l0<FILTER_SIZE;++l0){
			sum=complexAdd(sum,
					complexMul_scalar(TEX2COMPLEX2((u0+l0)&(k_gpu.n[0]-1)),psi0));
			psi0 = updatePSI_factors(fg_psij_fac,l0+1);
		}
		f[fpos]=sum;
	}
}
#endif

#else
template<int i> __device__ gpuComplex conv1D(uint_t u,uint_t n,dTyp pos,
		uint_t sigma,dTyp sum_re, dTyp sum_im)
{
	double phi=getPHI(n*pos-((double)u+i),sigma);
		return conv1D<i-1>(u,n,pos,sigma,sum_re +(TEX2COMPLEX((u+i)&n-1).x*phi),
				sum_im +(TEX2COMPLEX((u+i)&n-1).y * phi));
}
template<> __device__ gpuComplex conv1D<-1>(uint_t u,uint_t n,dTyp pos,
		uint_t sigma,dTyp sum_re, dTyp sum_im)
{
	return newComplex(sum_re,sum_im);
}

__global__ void kernelConvWithGauss_1d(const dTyp *x_in,gpuComplex *f,uint_t width)
{
	gpuComplex sum;sum.x=0.0; sum.y=0.0;
	for(uint_t fpos = blockDim.x * blockIdx.x + threadIdx.x; fpos < width;
			fpos+=blockDim.x*gridDim.x){
		dTyp pos0 =x_in[fpos] ;
		uint_t n0=k_gpu.n[0];

#if(UNROLL_INNER)
		f[fpos]=conv1D<FILTER_SIZE-1>(cunfft_u(n0,pos0),n0,pos0,k_gpu.sigma[0],
				sum.x,sum.y);
#else
		uint_t u0=cunfft_u(n0,pos0);
#pragma unroll 13 // m_min=1
		for(int l0= 0; l0<FILTER_SIZE;++l0){
			sum=complexAdd(sum,complexMul_scalar(TEX2COMPLEX((u0+l0)&(n0-1)),
							getPHI(n0*pos0-((double)(u0+l0)),k_gpu.sigma[0])));
		}
		f[fpos]=sum;
#endif
	}
}
#endif //COM_FG_PSI

//------------------------------------------------------------------------------
//					ADJOINT CONV: 1D KERNEL
//------------------------------------------------------------------------------
#ifdef COM_FG_PSI
/**
 * for kepler architectur the const and __restrict keyword may be tagged by the
 * compiler to be loaded through the Read‐Only Data Cache, see the Kepler
 * architecture whitepaper.
 */
__global__ void kernelConvWithGauss_adjoint_1d( gpuComplex *g, dTyp *x_in,
		const gpuComplex* __restrict f, uint_t width)
{
	double psi;
	uint_t u_x=0;
	int l;

	double fg_psij_fac[3];

	for(uint_t fpos = blockDim.x * blockIdx.x + threadIdx.x; fpos < width;
					fpos+=blockDim.x*gridDim.x){
		u_x = cunfft_u(k_gpu.n[0],x_in[fpos]);

		psi=getPSI_factors(fg_psij_fac,k_gpu.n[0]*x_in[fpos]-(dTyp)u_x,0);
#pragma unroll 4 // m_min=1
		for(l= 0; l<FILTER_SIZE;++l){
			uint_t supp_j = (u_x+l)&(k_gpu.n[0]-1);
			ATOMIC_ADD(&(g[supp_j].x),f[fpos].x*psi);
			ATOMIC_ADD(&(g[supp_j].y),f[fpos].y*psi);
			psi = updatePSI_factors(fg_psij_fac,l+1);
		}
		__syncthreads();
	}
}
#else



__global__ void kernelConvWithGauss_adjoint_1d( gpuComplex *f,gpuComplex *g,
		dTyp *x_in,uint_t width)
{
	double psi;
	int l;
	uint_t u_x=0;
	gpuComplex f_in;

	for(uint_t fpos = blockDim.x * blockIdx.x + threadIdx.x; fpos < width;
				fpos+=blockDim.x*gridDim.x){
		dTyp x=x_in[fpos];
		u_x = cunfft_u(k_gpu.n[0],x);
		f_in=f[fpos];
#pragma unroll 13
		for(l=0; l<FILTER_SIZE;++l)
		{
			uint_t supp_j = (u_x+l)&(k_gpu.n[0]-1);
			psi=getPHI(k_gpu.n[0]*x-((double)(u_x+l)),k_gpu.sigma[0]);
			ATOMIC_ADD(&(g[supp_j].x),f_in.x*psi);
			ATOMIC_ADD(&(g[supp_j].y),f_in.y*psi);
		}
	}
}
#endif


//------------------------------------------------------------------------------
//					CONV: 2D KERNEL
//------------------------------------------------------------------------------
#ifdef COM_FG_PSI
#ifdef TEXTURE
__global__ void kernelConvWithGauss_2d(const dTyp *x_in,gpuComplex *f,uint_t width)
{
	gpuComplex subSum;
	gpuComplex sum;
	uint_t i0;

	double fg_psijFac_0[3];
	double fg_psijFac_1[3];
	int l0,l1;
	uint_t u0,u1;
	double psi0,psi1;
	dTyp pos0,pos1;

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
			fpos+=blockDim.x*gridDim.x){

		resetComplex(&sum);

		pos1=x_in[2*fpos+1];//x:1
		pos0=x_in[2*fpos];//y:0

		u0 = cunfft_u(k_gpu.n[0],pos0);
		u1 = cunfft_u(k_gpu.n[1],pos1);

		psi0=getPSI_factors(fg_psijFac_0,k_gpu.n[0]*pos0-(dTyp)u0,0);
		psi1=getPSI_factors(fg_psijFac_1,k_gpu.n[1]*pos1-(dTyp)u1,1);

		for(l0=0; l0<FILTER_SIZE; ++l0){
			i0=((u0+l0)&(k_gpu.n[0]-1));
			resetComplex(&subSum);
			fg_psijFac_1[2]=1.0;
			psi1=fg_psijFac_1[0];

#pragma unroll 13 // m_min=1
			for(l1=0; l1<FILTER_SIZE; ++l1){
				subSum=complexAdd(subSum,complexMul_scalar(TEX2COMPLEX(
						POS2D((u1+l1)&(k_gpu.n[1]-1),i0,k_gpu.n[1])),psi1));
				psi1=updatePSI_factors(fg_psijFac_1,l1+1);
			}

			sum=complexAdd(sum,complexMul_scalar(subSum,psi0));
			psi0=updatePSI_factors(fg_psijFac_0,l0+1);
		}
		f[fpos]=sum;
	}
}
#else
__global__ void kernelConvWithGauss_2d(const gpuComplex *g,const dTyp *x_in,gpuComplex *f,uint_t width)
{
	gpuComplex subSum;
	gpuComplex sum;
	uint_t i0;

	double fg_psijFac_0[3];
	double fg_psijFac_1[3];
	int l0,l1;
	uint_t u0,u1;
	double psi0,psi1;
	dTyp pos0,pos1;

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
			fpos+=blockDim.x*gridDim.x){

		resetComplex(&sum);

		pos1=x_in[2*fpos+1];//x:1
		pos0=x_in[2*fpos];//y:0

		u0 = cunfft_u(k_gpu.n[0],pos0);
		u1 = cunfft_u(k_gpu.n[1],pos1);

		psi0=getPSI_factors(fg_psijFac_0,k_gpu.n[0]*pos0-(dTyp)u0,0);
		psi1=getPSI_factors(fg_psijFac_1,k_gpu.n[1]*pos1-(dTyp)u1,1);

		for(l0=0; l0<FILTER_SIZE; ++l0){
			i0=((u0+l0)&(k_gpu.n[0]-1));
			resetComplex(&subSum);
			fg_psijFac_1[2]=1.0;
			psi1=fg_psijFac_1[0];

#pragma unroll 13 // m_min=1
			for(l1=0; l1<FILTER_SIZE; ++l1){
				subSum=complexAdd(subSum,complexMul_scalar(TEX2COMPLEX2(
						POS2D((u1+l1)&(k_gpu.n[1]-1),i0,k_gpu.n[1])),psi1));
				psi1=updatePSI_factors(fg_psijFac_1,l1+1);
			}

			sum=complexAdd(sum,complexMul_scalar(subSum,psi0));
			psi0=updatePSI_factors(fg_psijFac_0,l0+1);
		}
		f[fpos]=sum;
	}
}
#endif //TEXTURE
#else


template<int i> __device__ __forceinline__ gpuComplex conv2D_l1(uint_t u,
		uint_t i0,uint_t n,dTyp pos,double sigma,dTyp sum_re, dTyp sum_im)
{
	double phi=getPHI(n*pos-((double)u+i),sigma);
	gpuComplex g=TEX2COMPLEX(POS2D((u+i)&(n-1),i0,n));
	return conv2D_l1<i-1>(u,i0,n,pos,sigma,sum_re +g.x *phi,sum_im + g.y *phi );

}
template<> __device__ __forceinline__ gpuComplex
	conv2D_l1<-1>(uint_t u,uint_t pos_g,uint_t n,dTyp pos,double sigma,
			dTyp sum_re, dTyp sum_im)
{
	return newComplex(sum_re,sum_im);
}

template<int i> __device__ __forceinline__
	gpuComplex conv2D_l0(uint_t *u,dTyp *pos,double *sigma,dTyp sum_re, dTyp sum_im)
{
	gpuComplex g=newComplexZero();
	g=conv2D_l1<FILTER_SIZE-1>(u[1],(u[0]+i)&(k_gpu.n[0]-1),k_gpu.n[1],pos[1],
			sigma[1],g.x,g.y);
	double phi=getPHI(k_gpu.n[0]*pos[0]-((double)u[0]+i),sigma[0]);
	return conv2D_l0<i-1>(u,pos,sigma,sum_re +g.x *phi,sum_im + g.y *phi );

}
template<> __device__ __forceinline__ gpuComplex
	conv2D_l0<-1>(uint_t *u,dTyp *pos, double *sigma,dTyp sum_re, dTyp sum_im)
{
	return newComplex(sum_re,sum_im);
}


__global__ void kernelConvWithGauss_2d(const dTyp *x_in,gpuComplex *f,uint_t width)
{
	gpuComplex sum;
	dTyp pos[2];
	uint_t u[2];
	double sigma[2]={k_gpu.n[0]/k_gpu.N[0],k_gpu.n[1]/k_gpu.N[1]};

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
				fpos+=blockDim.x*gridDim.x){
		resetComplex(&sum);

		pos[0] = x_in[2*fpos];
		u[0] = cunfft_u(k_gpu.n[0],pos[0]);

		pos[1] = x_in[2*fpos+1];
		u[1] = cunfft_u(k_gpu.n[1],pos[1]);

#if UNROLL_INNER
		f[fpos]=conv2D_l0<FILTER_SIZE-1>(u,pos,sigma,sum.x,sum.y);

#else
		uint_t i[2];
		gpuComplex subSum;
		for(int l0=0; l0<FILTER_SIZE; ++l0){
			i[0] = (u[0]+l0)&(k_gpu.n[0]-1);
			resetComplex(&subSum);
#pragma unroll 6
			for(int l1=0; l1<FILTER_SIZE; ++l1){
				subSum=complexAdd(subSum,complexMul_scalar(TEX2COMPLEX(
						POS2D(((u[1]+l1)&(k_gpu.n[0]-1)),i[0],k_gpu.n[1])),
							getPHI(k_gpu.n[1]*pos[1]-((dTyp)(u[1]+l1)),sigma[1])));
			}
			sum=complexAdd(sum,complexMul_scalar(subSum,
					getPHI(k_gpu.n[0]*pos[0]-((dTyp)(u[0]+l0)),sigma[0])));
		}
		f[fpos]=sum;

#endif
	}
}
#endif

//------------------------------------------------------------------------------
//					ADJOINT CONV: 2D KERNEL
//------------------------------------------------------------------------------
#ifdef COM_FG_PSI
__global__ void kernelConvWithGauss_adjoint_2d( gpuComplex *g, dTyp *x_in,
		uint_t width)
{
	double psi1,psi0;
	int l1,l0;
	uint_t u0=0,u1=0;
	dTyp pos0,pos1;
	uint_t supp1,supp0;
	double fg_psijFac_0[3];
	double fg_psijFac_1[3];

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
			fpos+=blockDim.x*gridDim.x){
		gpuComplex f=TEX2COMPLEX(fpos);

		pos0=x_in[2*fpos];
		u0 = cunfft_u(k_gpu.n[0],pos0);
		pos1=x_in[2*fpos+1];
		u1 = cunfft_u(k_gpu.n[1],pos1);
		psi0=getPSI_factors(fg_psijFac_0,k_gpu.n[0]*pos0-(dTyp)u0,0);
		psi1=getPSI_factors(fg_psijFac_1,k_gpu.n[1]*pos1-(dTyp)u1,1);
		psi0=fg_psijFac_0[0];
		for(l0=0; l0<FILTER_SIZE; ++l0){
			supp0=((u0+l0)&k_gpu.n[0]-1);
			fg_psijFac_1[2]=1.0;
			psi1=fg_psijFac_1[0];
#pragma unroll 4
			for(l1=0; l1<FILTER_SIZE;++l1){
				supp1 = (u1+l1)&(k_gpu.n[1]-1);
				ATOMIC_ADD(&(g[POS2D(supp1,supp0,k_gpu.n[1])].x),f.x*psi1*psi0);
				ATOMIC_ADD(&(g[POS2D(supp1,supp0,k_gpu.n[1])].y),f.y*psi1*psi0);
				psi1=updatePSI_factors(fg_psijFac_1,l1+1);
			}
			psi0=updatePSI_factors(fg_psijFac_0,l0+1);
		}
	}
}
#else

__global__ void kernelConvWithGauss_adjoint_2d( gpuComplex *f,gpuComplex *g,
		dTyp *x_in,uint_t width)
{
	double psi1,psi0;
	int l1,l0;
	uint_t u0=0,u1=0;
	dTyp pos0,pos1;
	uint_t supp1,supp0;

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
				fpos+=blockDim.x*gridDim.x){
		gpuComplex f_in=f[fpos];
		pos0=x_in[2*fpos];
		u0 = cunfft_u(k_gpu.n[0],pos0);
		pos1=x_in[2*fpos+1];
		u1 = cunfft_u(k_gpu.n[1],pos1);

		for(l0=0; l0<FILTER_SIZE; ++l0){
			supp0=((u0+l0)&k_gpu.n[0]-1);
			psi0=getPHI(k_gpu.n[0]*pos0-((dTyp)(u0+l0)),k_gpu.sigma[1]);
#pragma unroll 13
			for(l1=0; l1<FILTER_SIZE;++l1){
				supp1 = (u1+l1)&(k_gpu.n[1]-1);
				psi1=getPHI(k_gpu.n[1]*pos1-((double)(u1+l1)),k_gpu.sigma[1]);
				ATOMIC_ADD(&(g[POS2D(supp1,supp0,k_gpu.n[1])].x),f_in.x*psi1*psi0);
				ATOMIC_ADD(&(g[POS2D(supp1,supp0,k_gpu.n[1])].y),f_in.y*psi1*psi0);
			}
		}
	}
}
#endif

//------------------------------------------------------------------------------
//					CONV: 3D KERNEL
//------------------------------------------------------------------------------
#ifdef COM_FG_PSI
#ifdef TEXTURE
__global__ void kernelConvWithGauss_3d(const dTyp *x_in,
		gpuComplex *f, uint_t width, uint_t planeSize)
{
	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
				fpos+=blockDim.x*gridDim.x){
			gpuComplex subSum2;
			gpuComplex subSum1;
			gpuComplex sum;
			resetComplex(&sum);

			dTyp pos2=x_in[3*fpos+2];
			dTyp pos1=x_in[3*fpos+1];
			dTyp pos0=x_in[3*fpos];

			uint_t u0 = cunfft_u(k_gpu.n[0],pos0);
			uint_t u1 = cunfft_u(k_gpu.n[1],pos1);
			uint_t u2 = cunfft_u(k_gpu.n[2],pos2);
			uint_t i2,i1,i0;

			double fg_psijFac_0[3];
			double fg_psijFac_1[3];
			double fg_psijFac_2[3];
			double psi0=getPSI_factors(fg_psijFac_0,k_gpu.n[0]*pos0-(dTyp)u0,0);
			double psi1=getPSI_factors(fg_psijFac_1,k_gpu.n[1]*pos1-(dTyp)u1,1);
			double psi2=getPSI_factors(fg_psijFac_2,k_gpu.n[2]*pos2-(dTyp)u2,2);

			int l2,l1,l0;

			for(l0=0; l0<FILTER_SIZE; ++l0){
				i0=((u0+l0)&(k_gpu.n[0]-1));
				resetComplex(&subSum1);
				fg_psijFac_1[2]=1.0;
				psi1=fg_psijFac_1[0];
#pragma unroll
				for(l1=0; l1<FILTER_SIZE; ++l1){
					i1=((u1+l1)&(k_gpu.n[1]-1));
					resetComplex(&subSum2);
					fg_psijFac_2[2]=1.0;
					psi2=fg_psijFac_2[0];
#pragma unroll
					for(l2=0; l2<FILTER_SIZE; ++l2){
						i2 = (u2+l2)&(k_gpu.n[2]-1);
						subSum2=complexAdd(subSum2,complexMul_scalar(
								TEX2COMPLEX(POS3D_C(i2,i1*k_gpu.n[2],i0*planeSize)),psi2));
						psi2=updatePSI_factors(fg_psijFac_2,l2+1);
					}
					subSum1=complexAdd(subSum1,complexMul_scalar(subSum2,psi1));
					psi1=updatePSI_factors(fg_psijFac_1,l1+1);
				}
				sum=complexAdd(sum,complexMul_scalar(subSum1,psi0));
				psi0=updatePSI_factors(fg_psijFac_0,l0+1);
			}
			f[fpos]=sum;
		}
}
#else
__global__ void kernelConvWithGauss_3d(const gpuComplex *g,const dTyp *x_in,
		gpuComplex *f, uint_t width, uint_t planeSize)
{
	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
				fpos+=blockDim.x*gridDim.x){
			gpuComplex subSum2;
			gpuComplex subSum1;
			gpuComplex sum;
			resetComplex(&sum);

			dTyp pos2=x_in[3*fpos+2];
			dTyp pos1=x_in[3*fpos+1];
			dTyp pos0=x_in[3*fpos];

			uint_t u0 = cunfft_u(k_gpu.n[0],pos0);
			uint_t u1 = cunfft_u(k_gpu.n[1],pos1);
			uint_t u2 = cunfft_u(k_gpu.n[2],pos2);
			uint_t i2,i1,i0;

			double fg_psijFac_0[3];
			double fg_psijFac_1[3];
			double fg_psijFac_2[3];
			double psi0=getPSI_factors(fg_psijFac_0,k_gpu.n[0]*pos0-(dTyp)u0,0);
			double psi1=getPSI_factors(fg_psijFac_1,k_gpu.n[1]*pos1-(dTyp)u1,1);
			double psi2=getPSI_factors(fg_psijFac_2,k_gpu.n[2]*pos2-(dTyp)u2,2);

			int l2,l1,l0;

			for(l0=0; l0<FILTER_SIZE; ++l0){
				i0=((u0+l0)&(k_gpu.n[0]-1));
				resetComplex(&subSum1);
				fg_psijFac_1[2]=1.0;
				psi1=fg_psijFac_1[0];
#pragma unroll
				for(l1=0; l1<FILTER_SIZE; ++l1){
					i1=((u1+l1)&(k_gpu.n[1]-1));
					resetComplex(&subSum2);
					fg_psijFac_2[2]=1.0;
					psi2=fg_psijFac_2[0];
#pragma unroll
					for(l2=0; l2<FILTER_SIZE; ++l2){
						i2 = (u2+l2)&(k_gpu.n[2]-1);
						subSum2=complexAdd(subSum2,complexMul_scalar(
								TEX2COMPLEX2(POS3D_C(i2,i1*k_gpu.n[2],i0*planeSize)),psi2));
						psi2=updatePSI_factors(fg_psijFac_2,l2+1);
					}
					subSum1=complexAdd(subSum1,complexMul_scalar(subSum2,psi1));
					psi1=updatePSI_factors(fg_psijFac_1,l1+1);
				}
				sum=complexAdd(sum,complexMul_scalar(subSum1,psi0));
				psi0=updatePSI_factors(fg_psijFac_0,l0+1);
			}
			f[fpos]=sum;
		}
}
#endif
#else // no fg_psi precomputation

template<int i> __device__ __forceinline__ gpuComplex conv3D_l2(uint_t u,
		uint_t i1,uint_t i0,
		uint_t n,uint_t n1,dTyp pos,double sigma,dTyp sum_re, dTyp sum_im)
{
	gpuComplex g=TEX2COMPLEX(POS3D((u+i)&(n-1),i1,i0,n1,n));
	double phi=getPHI(n*pos-((double)u+i),sigma);
	return conv3D_l2<i-1>(u,i1,i0,n,n1,pos,sigma,sum_re+g.x*phi,sum_im+g.y*phi);
}
template<> __device__ __forceinline__ gpuComplex conv3D_l2<-1>(uint_t u,
		uint_t i1,uint_t i0,
		uint_t n,uint_t n1,dTyp pos, double sigma,dTyp sum_re, dTyp sum_im)
{
	return newComplex(sum_re,sum_im);
}

template<int i> __device__ __forceinline__
	gpuComplex conv3D_l1(uint_t *u,dTyp *pos,uint_t i0,dTyp sum_re, dTyp sum_im)
{
	gpuComplex g=newComplexZero();
	g=conv3D_l2<FILTER_SIZE-1>(u[2],(u[1]+i)&(k_gpu.n[1]-1),i0,k_gpu.n[2],
			k_gpu.n[1],pos[2],k_gpu.sigma[2],g.x,g.y);
	double phi=getPHI(k_gpu.n[1]*pos[1]-((double)u[1]+i),k_gpu.sigma[1]);
	return conv3D_l1<i-1>(u,pos,i0,sum_re +g.x *phi,sum_im + g.y *phi );

}
template<> __device__ __forceinline__ gpuComplex
	conv3D_l1<-1>(uint_t *u,dTyp *pos, uint_t i0,dTyp sum_re, dTyp sum_im)
{
	return newComplex(sum_re,sum_im);
}

__global__ void kernelConvWithGauss_3d(const dTyp *x_in,gpuComplex *f,
		uint_t width,uint_t planeSize)
{
	gpuComplex sum0,sum1;
	dTyp pos[3];
	int l0;
	uint_t u[3];

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
					fpos+=blockDim.x*gridDim.x){
		resetComplex(&sum0);

		pos[0] = x_in[3*fpos];
		pos[1] = x_in[3*fpos+1];
		pos[2] = x_in[3*fpos+2];

		u[0] = cunfft_u(k_gpu.n[0],pos[0]);
		u[1] = cunfft_u(k_gpu.n[1],pos[1]);
		u[2] = cunfft_u(k_gpu.n[2],pos[2]);

#pragma unroll 6
		for(l0=0; l0<FILTER_SIZE; ++l0){
			resetComplex(&sum1);
			sum1=conv3D_l1<FILTER_SIZE-1>(u,pos,(u[0]+l0)&(k_gpu.n[0]-1),
					sum1.x,sum1.y);
			sum0=complexAdd(sum0,complexMul_scalar(sum1,
					getPHI(k_gpu.n[0]*pos[0]-((dTyp)(u[0]+l0)),k_gpu.sigma[0])));
		}
		f[fpos]=sum0;
	}
}

#endif

#ifdef COM_FG_PSI
__global__ void kernelConvWithGauss_adjoint_3d( gpuComplex *g, dTyp *x_in,
		uint_t width)
{
	int l2,l1,l0;
	dTyp pos2,pos1,pos0;
	uint_t supp2,supp1,supp0;
	uint_t u2=0,u1=0, u0=0;
	double fg_psijFac_0[3];
	double fg_psijFac_1[3];
	double fg_psijFac_2[3];

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
			fpos+=blockDim.x*gridDim.x){
		pos0=x_in[3*fpos];
		u0 = cunfft_u(k_gpu.n[0],pos0);

		pos1=x_in[3*fpos+1];
		u1 = cunfft_u(k_gpu.n[1],pos1);

		pos2=x_in[3*fpos+2];
		u2 = cunfft_u(k_gpu.n[2],pos2);

		double psi0=getPSI_factors(fg_psijFac_0,k_gpu.n[0]*pos0-(dTyp)u0,0);
		double psi1=getPSI_factors(fg_psijFac_1,k_gpu.n[1]*pos1-(dTyp)u1,1);
		double psi2=getPSI_factors(fg_psijFac_2,k_gpu.n[2]*pos2-(dTyp)u2,2);

		gpuComplex f=TEX2COMPLEX(fpos);

		for(l0=0; l0<FILTER_SIZE; ++l0){
			supp0=((u0+l0)&k_gpu.n[0]-1);

			fg_psijFac_1[2]=1.0;
			psi1=fg_psijFac_1[0];
#pragma unroll 4
			for(l1=0; l1<FILTER_SIZE;++l1){
				supp1 = (u1+l1)&(k_gpu.n[1]-1);
				fg_psijFac_2[2]=1.0;
				psi2=fg_psijFac_2[0];
				for(l2=0; l2<FILTER_SIZE;++l2){
					supp2 = (u2+l2)&(k_gpu.n[2]-1);
					ATOMIC_ADD(&(g[POS3D(supp2,supp1,supp0,k_gpu.n[1],k_gpu.n[2])].x),
							f.x*psi0*psi1*psi2);
					ATOMIC_ADD(&(g[POS3D(supp2,supp1,supp0,k_gpu.n[1],k_gpu.n[2])].y),
							f.y*psi0*psi1*psi2);
					psi2=updatePSI_factors(fg_psijFac_2,l2+1);
				}
				psi1=updatePSI_factors(fg_psijFac_1,l1+1);
			}
			psi0=updatePSI_factors(fg_psijFac_0,l0+1);
		}
	}
}
#else

__global__ void kernelConvWithGauss_adjoint_3d(gpuComplex *f, gpuComplex *g,
		dTyp *x_in,uint_t width)
{

	double psi2,psi1,psi0;
	int l2,l1,l0;
	dTyp pos2,pos1,pos0;
	uint_t supp2,supp1,supp0;
	uint_t u2=0,u1=0, u0=0;

	for(uint_t fpos=blockIdx.x*blockDim.x+threadIdx.x; fpos< width;
						fpos+=blockDim.x*gridDim.x){
		pos0=x_in[3*fpos];
		u0 = cunfft_u(k_gpu.n[0],pos0);

		pos1=x_in[3*fpos+1];
		u1 = cunfft_u(k_gpu.n[1],pos1);

		pos2=x_in[3*fpos+2];
		u2 = cunfft_u(k_gpu.n[2],pos2);

//		gpuComplex f=tex2Complex(fpos);
		gpuComplex f_in=f[fpos];

		for(l0=0; l0<FILTER_SIZE; ++l0){
			supp0=((u0+l0)&k_gpu.n[0]-1);
			psi0=getPHI(k_gpu.n[0]*pos0-((dTyp)(u0+l0)),k_gpu.sigma[0]);
#pragma unroll 4
			for(l1=0; l1<FILTER_SIZE;++l1)
			{
				supp1 = (u1+l1)&(k_gpu.n[1]-1);
				psi1=getPHI(k_gpu.n[1]*pos1-((dTyp)(u1+l1)),k_gpu.sigma[1]);

				for(l2=0; l2<FILTER_SIZE;++l2){
					supp2 = (u2+l2)&(k_gpu.n[2]-1);
					psi2=getPHI(k_gpu.n[2]*pos2-((dTyp)(u2+l2)),k_gpu.sigma[2]);

					ATOMIC_ADD(&(g[POS3D(supp2,supp1,supp0,k_gpu.n[1],k_gpu.n[2])].x),
							f_in.x*psi0*psi1*psi2);
					ATOMIC_ADD(&(g[POS3D(supp2,supp1,supp0,k_gpu.n[1],k_gpu.n[2])].y),
							f_in.y*psi0*psi1*psi2);
				}
			}
		}
	}
}

#endif //FG_PSI



//------------------------------------------------------------------------------
// 			NDFT
//------------------------------------------------------------------------------

__global__ void kernelNDFT_1d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total)
{
	gpuComplex sum;
	uint_t k_L;
	double omega;

	for(uint_t j=blockIdx.x*blockDim.x+threadIdx.x;j<M_total;
			j+=blockDim.x*gridDim.x){
		sum=newComplexZero();
		dTyp x=x_in[j];
		for(k_L = 0; k_L < N_total; k_L++){
			omega = (k_L - N_total/2) * K2PI*x;
			sum=complexAdd(sum,complexMul(f_hat[k_L],
					newComplex(cos(-omega),sin(-omega))));
		}
		f[j]=sum;
	}


}

__device__ gpuComplex NDFT_forElem(uint_t j,dTyp *x_in,gpuComplex *f_hat,
		uint_t N_total,	sint_t *kd,double *x,double *Omega, int d)
{
	uint_t t, t2, k_L;
	gpuComplex sum=newComplexZero();
	double omega;
	Omega[0] = 0.0;

	for(t=0; t<d;t++){
		kd[t]=-k_gpu.N[t]/2;
		x[t]=K2PI*x_in[j*d+t];
		Omega[t+1]=kd[t]*x[t]+Omega[t];
	}

	omega= Omega[d];

	for(k_L = 0; k_L < N_total; k_L++){
		//exp(0)=1
		sum=complexAdd(sum,complexMul(f_hat[k_L],newComplex(cos(-omega),
				sin(-omega))));
		for(t = d-1; (t >= 1) && (kd[t] == k_gpu.N[t]/2-1); t--)
			kd[t]-= k_gpu.N[t]-1;

		kd[t]++;

		for(t2 = t; t2 < d; t2++)
			Omega[t2+1] = kd[t2]*x[t2]+Omega[t2];

		omega = Omega[d];
	} /* for(k_L) */
	return sum;
}

__global__ void kernelNDFT_2d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total)
{
	double x[2];
	uint_t kd[2];
	double Omega[3];
	for(uint_t j=blockIdx.x*blockDim.x+threadIdx.x;j<M_total;
				j+=blockDim.x*gridDim.x){
		f[j]=NDFT_forElem(j,x_in,f_hat,N_total,kd,x,Omega,2);
	}
}

__global__ void kernelNDFT_3d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total)
{
	double x[3];
	uint_t kd[3];
	double Omega[4];
	for(uint_t j=blockIdx.x*blockDim.x+threadIdx.x;j<M_total;
					j+=blockDim.x*gridDim.x){
		f[j]=NDFT_forElem(j,x_in,f_hat,N_total,kd,x,Omega,3);
	}
}

//------------------------------------------------------------------------------
// 			NDFT ADJOINT: 1D KERNEL
//------------------------------------------------------------------------------
__global__ void kernelNDFT_adjoint_1d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total)
{
	for(uint_t k_L=blockIdx.x*blockDim.x+threadIdx.x;k_L<N_total;
				k_L+=blockDim.x*gridDim.x){
		gpuComplex sum=newComplexZero();
		uint_t j;
		double omega;
		for(j = 0; j < M_total; j++){
			omega = (k_L - N_total/2) * K2PI*x_in[j];
			sum=complexAdd(sum,complexMul(f[j],newComplex(cos(omega),sin(omega))));
		}
		f_hat[k_L]=sum;
	}
}

__device__ gpuComplex NDFT_adjoint_forElem(uint_t k_L,dTyp *x_in,gpuComplex *f,
		uint_t M_total,sint_t *k, int d)
{
	gpuComplex sum=newComplexZero();
	uint_t k_temp=k_L;
	uint_t t, j;
	double omega;
	for(t=d-1; t>=0;t--){
		k[t]=k_temp % k_gpu.N[t] - k_gpu.N[t]/2;
		k_temp /=k_gpu.N[t];
	}

	for(j=0; j< M_total; j++){
		omega=0.0;
		for(t=0; t< d; t++){
			omega += k[t]*K2PI * x_in[j*d+t];
		}
		sum=complexAdd(sum,complexMul(f[j],newComplex(cos(omega),sin(omega))));
	}
	return sum;
}

__global__ void kernelNDFT_adjoint_2d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total)
{
	sint_t k[2];
	for(uint_t k_L=blockIdx.x*blockDim.x+threadIdx.x;k_L<N_total;
			k_L+=blockDim.x*gridDim.x){
		f_hat[k_L]=NDFT_adjoint_forElem(k_L,x_in,f,M_total,k,2);
	}
}

__global__ void kernelNDFT_adjoint_3d(dTyp *x_in,gpuComplex *f_hat,gpuComplex *f,
		uint_t M_total, uint_t N_total)
{
	sint_t k[3];
	for(uint_t k_L=blockIdx.x*blockDim.x+threadIdx.x;k_L<N_total;
				k_L+=blockDim.x*gridDim.x){
		f_hat[k_L]=NDFT_adjoint_forElem(k_L,x_in,f,M_total,k,3);
	}
}

//------------------------------------------------------------------------------
// 		LAUNCH CONFIGURATION FUNCTION
//------------------------------------------------------------------------------
int getComputationGridParams_Matrix(dim3 *blockDim, dim3 *gridDim, uint_t sizey,
		uint_t sizex)
{
//	printf("\tCall getComputationGridParams2D_ROC with y=" PRINT_FORMAT ",x=" PRINT_FORMAT "\n",sizey,sizex);

	int res=1;
	uint_t threadsize1 = sizex < THREAD_DIM_X ? sizex : THREAD_DIM_X;
	uint_t threadsize2 = sizey < THREAD_DIM_Y ? sizey : THREAD_DIM_Y;

	if(threadsize1*threadsize2 >= MAX_NUM_THREADS){
//		printf("ATTENTION: blockDim(" PRINT_FORMAT "," PRINT_FORMAT ") exceed Max_ThreadDim\n",threadsize1,threadsize2);
		while(threadsize1*threadsize2 >=MAX_NUM_THREADS){
			threadsize1>>=1;
			threadsize2>>=1;
		}
//		printf("\t change to " PRINT_FORMAT"," PRINT_FORMAT "\n",threadsize1,threadsize2);
	}
	dim3 bDim(threadsize1,threadsize2,1);
	dim3 gDim((sizex + bDim.x - 1)/ bDim.x, (sizey + bDim.y - 1) / bDim.y, 1);

	(*blockDim)=bDim;
	if(gDim.x > MAX_GRID_DIM_X){
		(*gridDim).x=MAX_GRID_DIM_X;
		res=0;
	}else{
		(*gridDim).x=gDim.x; // TODO: >MAX_GRID_DIM??? (graphic-Karten abhaengig)
	}
	if(gDim.y > MAX_GRID_DIM_Y){
		(*gridDim).y=MAX_GRID_DIM_Y;
		res=0;
	}else{
		(*gridDim).y=gDim.y; // TODO: >MAX_GRID_DIM??? (graphic-Karten abhaengig)
	}

	return res;
}

int getComputationGridParams_Array(dim3 *blockDim, dim3 *gridDim, uint_t sizex)
{
	uint_t threadSize =  sizex < THREAD_DIM_X ? sizex : THREAD_DIM_X;

	dim3 bDim(threadSize,1,1);
	dim3 gDim((sizex + bDim.x - 1)/ bDim.x, 1, 1);

	(*blockDim)=bDim;
	if(gDim.x > MAX_GRID_DIM_X){
		(*gridDim)=MAX_GRID_DIM_X;
		return 0;
	}else{
		(*gridDim)=gDim;
		return 1;
	}
}


//------------------------------------------------------------------------------
// 		TEXTURE FUNCTIONS
//------------------------------------------------------------------------------
void allocateTexture1DFetch(gpuComplex *h_in,int width)
{
	int size=width*sizeof(gpuComplex);
	cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc<TEXTURE_TYPE>();
	//cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
	cudaVerify(cudaMalloc((void **)&d_dCData, size));
	cudaVerify(cudaMemcpy(d_dCData, h_in, size, cudaMemcpyHostToDevice));
//	texData.addressMode[0] = cudaAddressModeWrap;

	cudaVerify(cudaBindTexture(0, texData, d_dCData, channelDesc));
}

void freeTextureFetch()
{
	cudaVerify(cudaFree(d_dCData));
	cudaVerify(cudaUnbindTexture(texData));
}









