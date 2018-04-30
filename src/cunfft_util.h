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
 * $Id: cunfft1util.h 2012-05-31 11:36:00Z sukunis $
 *
 */

/**
 * @file cunfft1util.h
 * @brief Header file for utility functions used by the cunfft1 library.
 */

#ifndef CUNFFTUTIL_H_
#define CUNFFTUTIL_H_

#include <stdio.h>
//#define _USE_MATH_DEFINES
//#include <math.h>
#include <stdlib.h>
#include <malloc.h>


#include "cunfft_typedefs.h"

////////////////////////////////////////////////////////////////////////////////
/**Set to \f$=2\cdot\pi\f$ */
#define K2PI 6.2831853071795864769252867665590057683943387987502

#ifdef MILLI_SEC
/** Unit for time measurement is millisecond. See also MILLI_SEC. */
#define UNIT "ms"
/** Factor from second to millisecond is 1000.See also MILLI_SEC.*/
#define UNIT_FAC 1000.0
#else
/** Unit for time measurement is second. See also MILLI_SEC. */
#define UNIT "s"
/** Factor from second to second is 1.See also MILLI_SEC.*/
#define UNIT_FAC 1.0
#endif

/** binare logarithm*/
#define LOG2(x) log(x)/log(2)



//------------------------------------------------------------------------------
// 		TIMER OUTPUT FUNCTION
//------------------------------------------------------------------------------
/**
 * Print to stdout the given runtime in given unit (see UNIT and UNIT_FAC)
 *
 * @param name of runtime
 * @param runtime_in_sec in seconds
 */
static inline void showTime_t(const char* name,double runtime)
{
	printf("%s:\t %.2lE %s\n",
					name,runtime*UNIT_FAC,UNIT);
}

/**
 * Print to stdout the bandwith
 *
 * @param runtime_in_sec
 * @param mem_in_bytes input size of kernel function
 */
static inline void showTime_bandw(double runtime_in_sec,double mem_in_bytes)
{
	printf("\tMemory bandwith (Kernel)= %e GB/s\n",(mem_in_bytes/pow(10,9))/
			runtime_in_sec);
}

/**
 * Print to stdout the members of times. If MEASURE_TIMES defined print for all
 * steps, otherwise only the whole runtime.
 *
 * @param times struct of timers
 * @param tfac : number of runs. Print out given times/tfac.
 **/
void showTimes(NFFTTimeSpec *times,int tfac);

//------------------------------------------------------------------------------
// 		HELPER FUNCTION FOR MEMORY INFORMATIONS
//------------------------------------------------------------------------------
/**
 * Return given bytes in KB
 *
 * @param bytes in bytes
 * @return bytes in KB
 */
static inline unsigned long inKB(unsigned long bytes)
{
	return bytes/1024;
}

/**
 * Return given bytes in MB
 *
 * @param bytes
 * @return bytes in MB
 */
static inline unsigned long inMB(unsigned long bytes)
{
	return bytes/(1024*1024);
}



/**
 * Print memory overview to standard output
 *
 * @param free bytes
 * @param total bytes
 */
void printStats(unsigned long free, unsigned long total);

/**
 * Print memory overview to given file
 *
 * @param free bytes
 * @param total bytes
 * @param file output file
 * @param device number
 */
void printStatsToFile(unsigned long free, unsigned long total,FILE *file,int device);

/**
 * Return restriction for input size N as log_2(N).
 *
 * N=2^x, g: n_total= 2^d*N^d, f:M_total=N^d, f_hat:N_total=N^d,
 * x: d*M_total= d*N^d, size cufft: cN^d
 * use memory : \f$ 2^dN^d+N^d+N^d+dN^d+cN^d\f$
 * \f$ \Rightarrow log_2(freeMem/(2^d+d+2+c)) >= xd \f$
 * @param freeMem in bytes
 * @param dim dimension of problem
 * @return exponent for power of 2 (return x of \f$2^x\f$)
 */
int getMemRestrictions(unsigned long freeMem, int d);

/**
 * Print to stdout restriction to input size N as power of 2 for 1D,2D,3D.
 *
 * @param freeMem in bytes
 */
void showMemRestrictions(unsigned long freeMem);

/**
 * Computes the free memory and total memory for existing graphic cards on your
 * system and print to stdout their memory properties.
 *
 * @return free memory in bytes of last device
 */
unsigned long getGPUMemProps_ToStdout();

/**
* Compute used minimal memory space for input size
*
* @param p cunfft structur
*/
void showCPUMemUse(cunfft_plan *p);

//------------------------------------------------------------------------------
//			 OUTPUT FUNCTIONS
//------------------------------------------------------------------------------
/**
 * Print to stdout text and n elements of given dTyp array, four elements in one
 * line.
 *
 * @param x input array
 * @param n number of elements to print
 * @param text for output
 */
void showCoeff_double(const dTyp *x, int n, const char* text);

/**
 * Print to stdout text and n elements of given gpuComplex array, four elements
 * in one line.
 *
 * @param x input array
 * @param n number of elements to print
 * @param text for output
 */
void showCoeff_cuComplex(const gpuComplex *x, int n, const char* text);


/**
 * Print kernel configuration
 *
 * @param text additional text to print out
 * @param gridDim grid dimension for kernel call
 * @param blockDim block dimension for kernel call
 * @param k additional kernel identifier
 */
void config_out(const char* text, dim3 gridDim, dim3 blockDim, int k);

//------------------------------------------------------------------------------
//			 MATH HELPER FUNCTION
//------------------------------------------------------------------------------

/**
 *  \brief Computes the product of d elements of given vector.
 *
 *  @param vec array
 *  @param d number of elements
 *  @return product of d elements of vec
 **/
uint_t prod_int(uint_t *vec, int d);

/**
 * \brief Computes the next power for given N.
 *
 * @param N input value for witch next power of two should be compute.
 * @return power of two
 */
uint_t next_power_of_2(uint_t N);


//------------------------------------------------------------------------------
//				 EXAMPLE DATA FUNCTION
//------------------------------------------------------------------------------
/** Create clustered data and write it to the input of length N.
 *
 * @param in return vector for clustered Fourier Coeff
 * @param N size of in
 * */
void createFourierCoeff_clustered(gpuComplex *in, uint_t N);

/** Create uni distributed random data and write it to the input of length N.
 *
 *  @param in return vector for uniformly distributed Fourier Coeff
 * @param N size of in
 * */
void createFourierCoeff_uniDistr(gpuComplex *in, uint_t N);

/** Set given data to null
 *
 * @param in return vector for Fourier Coeff=0
 * @param N size of in
 * */
void createFourierCoeff_null(gpuComplex *in, uint_t N);

/** Create clustered data for \f$\hat{f}\f$ and x.
 *
 * @param cunfft_plan object
 */
void getExampleData_clustered(cunfft_plan *plan);

/** Create uni distributed random data for \f$\hat{f}\f$ and x.
 *
 * @param plan cunfft_plan object
 * */
void getExampleData_uniDistr(cunfft_plan *plan);

/** Create uni distributed random data for f and x.
 *
 *  @param plan cunfft_plan object
 * */
void getExampleDataAd_uniDistr(cunfft_plan *plan);

/** Create null data array for f and random data for x.
 *
 *  @param plan cunfft_plan object
 *  */
void getExampleData_null(cunfft_plan *plan);

/** Set f and x in plan with given data
 *
 *  @param plan cunfft_plan object
 *  @param f vector of length M_total as input for vector of samples f
 *  @param x vector of length M_total as input for nodes in time/spatial domain x
 * */
void setExampleData(cunfft_plan *plan, gpuComplex *f, dTyp *x);

//------------------------------------------------------------------------------
//			 ERROR NORM FUNCTION
//------------------------------------------------------------------------------
/**
 * computes \f$ L_{\infty} Norm  \f$
 *
 * @param x input vector 1
 * @param y input vector 2
 * @param n size of x and y
 */
double l_infty(gpuComplex *x, gpuComplex *y, int n);

/**
 * \brief Computes \f$ L_{\infty} \f$ error norm.
 *
 *@param x input vector 1
 * @param y input vector 2
 * @param n size of x and y
 * */

double compute_error_l_infty(gpuComplex *x, gpuComplex *y, int n);

//------------------------------------------------------------------------------
//				 CUDA ERROR FUNCTIONs
//------------------------------------------------------------------------------
#ifdef DEBUG

/** \def CHECK_LAUNCH_ERROR
 * Catch CUDA errors in kernel launches
 */
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
/** \def cudaVerify
 * Catch CUDA errors in kernel launches
 */
#define cufftVerify(err)  __cufftVerify(err, __FILE__, __LINE__)

inline void __cufftVerify( cufftResult err, const char *file, const int line )
{
    if( CUFFT_SUCCESS != err) {
    	fprintf(stderr, "%s(%i) : cufftSafeCall() CUFFT error %d: ",file, line,
    			(int)err);
        switch (err) {
            case CUFFT_INVALID_PLAN:   fprintf(stderr,"CUFFT_INVALID_PLAN\n");
            							break;
            case CUFFT_ALLOC_FAILED:   fprintf(stderr,"CUFFT_ALLOC_FAILED\n");
            							break;
            case CUFFT_INVALID_TYPE:   fprintf(stderr,"CUFFT_INVALID_TYPE\n");
            							break;
            case CUFFT_INVALID_VALUE:  fprintf(stderr,"CUFFT_INVALID_VALUE\n");
            							break;
            case CUFFT_INTERNAL_ERROR: fprintf(stderr,"CUFFT_INTERNAL_ERROR\n");
            							break;
            case CUFFT_EXEC_FAILED:    fprintf(stderr,"CUFFT_EXEC_FAILED\n");
            							break;
            case CUFFT_SETUP_FAILED:   fprintf(stderr,"CUFFT_SETUP_FAILED\n");
            							break;
            case CUFFT_INVALID_SIZE:   fprintf(stderr,"CUFFT_INVALID_SIZE\n");
            							break;
            case CUFFT_UNALIGNED_DATA: fprintf(stderr,"CUFFT_UNALIGNED_DATA\n");
            							break;
            default: fprintf(stderr, "CUFFT Unknown error code\n");break;
        }
        exit(EXIT_FAILURE);
    }
}

/** \def cudaVerify
 * Catch CUDA errors in kernel launches
 */
#define cudaVerify(x) do{												\
	cudaError_t __cu_result = (x);										\
	if(__cu_result != cudaSuccess){										\
		fprintf(stderr,"\nCUNFFT ERROR %s(%i):cuda function call failed:\n"	\
				"%s;\n => message: %s\n",									\
				__FILE__,__LINE__,#x,cudaGetErrorString(__cu_result));	\
				if(__cu_result == cudaErrorMemoryAllocation)	{				\
					fprintf(stderr,"\n ### PLEASE CHECK YOUR INPUT SIZE REGARDING TO DEVICE MEMORY CAPACITY! ###\n\n");\
					getGPUMemProps_ToStdout();}\
		exit(EXIT_FAILURE);													\
	}																	\
}while(0)

/** \def cudaVerifyKernel
 * Catch CUDA errors in kernel launches
 */
#define cudaVerifyKernel(x) do{											\
	(x);																\
	cudaError_t __cu_result = cudaGetLastError();						\
	if(__cu_result != cudaSuccess){										\
		fprintf(stderr,"\nCUFFT KERNEL ERROR %s(%i):cuda function call failed:\n"\
				"%s;\n => message: %s\n",									\
				__FILE__,__LINE__,#x,cudaGetErrorString(__cu_result));	\
		exit(-1);														\
	}																	\
}while(0)

/** \def cudaFFTVerify
 * Catch CUDA errors in kernel launches
 */
#define cudaFFTVerify(x) do{												\
	cufftResulz __cu_result = (x);										\
	if(__cu_result != CUFFT_SUCCESS){										\
		fprintf(stderr,"\nCUFFT ERROR %s(%i):cuda function call failed:\n"	\
				"%s;\n => message: %s\n",									\
				__FILE__,__LINE__,#x,cudaGetErrorString(__cu_result));	\
		exit(1);														\
	}																	\
}while(0)

#else
/** \def CHECK_LAUNCH_ERROR
 * Catch CUDA errors in kernel launches
 */
#define CHECK_LAUNCH_ERROR()

/** \def cudaVerify
 * Catch CUDA errors in kernel launches
 */
#define cudaVerify(x) do{												\
	(x);																\
}while(0)

/** \def cudaVerifyKernel
 * Catch CUDA errors in kernel launches
 */
#define cudaVerifyKernel(x) do{											\
	(x);																\
}while(0)

/** \def cudaFFTVerify
 * Catch CUDA errors in kernel launches
 */
#define cudaFFTVerify(x) do{\
	(x);\
	}while(0)

/** \def cudaVerify
 * Catch CUDA errors in kernel launches
 */
#define cufftVerify(err) do{(err);}while(0)
#endif /* DEBUG */

#endif /* CUNFFTUTIL_H_ */
