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
 * $Id: c_util.c 2012-05-31 11:36:00Z kunis $
 */

/**
 * @file cuda_util.cpp
 * @brief Utilities for CUDA
 */

#include "cunfft_util.h"
//------------------------------------------------------------------------------
// 		HELPER FUNCTION FOR MEMORY INFORMATIONS
//------------------------------------------------------------------------------

int getMemRestrictions(unsigned long freeMem, int d)
{
	int res=1;
	int size_g = pow(2,d); //2(N^d)==n
	int size_fhat = 1; //N
	int size_f = 1;//M==N
	int size_x = d; //M==N

	cufftType type;

#ifdef CUNFFT_DOUBLE_PRECISION
	type=CUFFT_Z2Z;
#else
	type=CUFFT_C2C;
#endif
	int neededMem = sizeof(gpuComplex)*(size_g*2 + size_f+size_fhat)+
			sizeof(dTyp)*size_x +sizeof(type)*size_g*2;
	res=LOG2((freeMem)/neededMem);
	return floor(res/d)-1;
}


void showMemRestrictions(unsigned long freeMem)
{
	printf("\n\tRestrictions cunfft for problemsize if M=N_total and one CUFFT Plan \n");

	int maxN_pot=getMemRestrictions(freeMem,1);
	printf("\t1D: max N in each direction= 2^%d\n",maxN_pot);

	maxN_pot=getMemRestrictions(freeMem,2);
	printf("\t2D: max N in each direction= 2^%d\n",maxN_pot);

	maxN_pot=getMemRestrictions(freeMem,3);
	printf("\t3D: max N in each direction= 2^%d\n\n",maxN_pot);
}


unsigned long getGPUMemProps_ToStdout()
{
	unsigned long free=0, total=0;
	int gpuCount, i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;

	cuInit(0);

	cuDeviceGetCount(&gpuCount);
	printf("===================================================\n");
	printf("Detected %d GPU\n",gpuCount);

	for (i=0; i<gpuCount; i++){
		cuDeviceGet(&dev,i);
		cuCtxCreate(&ctx, 0, dev);
		res = cuMemGetInfo(&free, &total);
		if(res != CUDA_SUCCESS){
			printf("Device: %d -- cuMemGetInfo failed! (status = %x)\n",i, res);
		}else{
			printf("....Device: %d\n",i);

			printStats(free, total);
			showMemRestrictions(free);
			cuCtxDetach(ctx);
		}
	}

	printf("===================================================\n");
   return free;
}

//------------------------------------------------------------------------------
//			 OUTPUT FUNCTIONS
//------------------------------------------------------------------------------

void showCoeff_cuComplex(const gpuComplex x[], int n, const char* text)
{
	if (text != NULL){
		printf ("\n %s\n", text);
	}

	int k;
	for (k = 0; k < n; k++)
	{
		if (k%4 == 0){
			printf("%6d.\t", k);
		}
		printf("%+.2lE %+.2lEi, ",x[k].x, x[k].y);

		if (k%4==3)
			printf("\n");
	}
	if (n%4!=0)
		printf("\n");

	printf("\n");
}

void config_out(const char* text, dim3 gridDim, dim3 blockDim, int k)
{
	printf("\tSTART %s[%d x %d,%d x %d]:%d ...\n",text,
			blockDim.x,blockDim.y,gridDim.x, gridDim.y,k);
}
//------------------------------------------------------------------------------
//				 EXAMPLE DATA FUNCTION
//------------------------------------------------------------------------------

void getExampleData_clustered(cunfft_plan *plan)
{
	createFourierCoeff_clustered(plan->f_hat,plan->N_total);

	uint_t i;
	uint_t M = plan->M_total*plan->d;

	double j;
	for(i=0,j= 0.0; i < M; ++i, j += 1./M){
		plan->x[i] =(dTyp)pow(j,1.0/100.0);
	}
}

void getExampleData_uniDistr(cunfft_plan *plan)
{
	createFourierCoeff_uniDistr(plan->f_hat,plan->N_total);

	uint_t i;
	uint_t M = plan->M_total*plan->d;

	for (i = 0; i < M; i++)
		plan->x[i] = (double)drand48() - (double)0.5;
}

void getExampleDataAd_uniDistr(cunfft_plan *plan)
{
	createFourierCoeff_uniDistr(plan->f,plan->M_total);

	uint_t i;
	uint_t M = plan->M_total*plan->d;

	for (i = 0; i < M; i++)
		plan->x[i] = (double)drand48() - (double)0.5;
}

void getExampleData_null(cunfft_plan *plan)
{
	createFourierCoeff_null(plan->f_hat,plan->N_total);

	uint_t i;
	uint_t M = plan->M_total*plan->d;

	for (i = 0; i < M; i++)
		plan->x[i] = (double)drand48() - (double)0.5;
}

void setExampleData(cunfft_plan *plan, gpuComplex *f, dTyp *x)
{
	int i;
	for(i=0; i<plan->M_total*plan->d;++i)
		plan->x[i]=x[i];

	for(i=0; i<plan->M_total; ++i)
		plan->f[i]=f[i];
}

void createFourierCoeff_clustered(gpuComplex *in, uint_t N)
{
	uint_t k;
	double kk;
	for (k = 0, kk=-N/2; k < N; k++,kk++) {
		in[k].x= (dTyp)(sin(10*kk/N) * exp(-pow(kk/pow(N,0.8),2)));
		in[k].y=(dTyp)0.0;
	}
}

void createFourierCoeff_uniDistr(gpuComplex *in, uint_t N)
{
	uint_t k;

	for (k = 0; k < N; k++){
	    in[k].x = (dTyp)drand48();
	    in[k].y= (dTyp)drand48();
	}
}

void createFourierCoeff_null(gpuComplex *in, uint_t N)
{
	uint_t k;

	for (k = 0; k < N; k++){
	    in[k].x = (dTyp)0.0;
	    in[k].y= (dTyp)0.0;
	}
}


//------------------------------------------------------------------------------
//			 ERROR NORM FUNCTION
//------------------------------------------------------------------------------
#ifdef CUNFFT_DOUBLE_PRECISION
#define ABS(x) cuCabs(x)
#define SUB(x,y) cuCsub(x,y)
#else
#define ABS(x) cuCabsf(x)
#define SUB(x,y) cuCsubf(x,y)
#endif


double l_infty(gpuComplex *x, gpuComplex *y, int n)
{
	int k;
	double linfty;

	if(y==NULL)
		for(k=0,linfty=0; k<n; k++)
			linfty=((linfty<ABS(x[k]))?ABS(x[k]):linfty);
	else
		for(k=0,linfty=0; k<n; k++){
			double absSub= ABS(SUB(x[k],y[k]));
			linfty=((linfty < absSub) ? absSub :linfty);
		}

	return linfty;
}

/** Computes error norm \f$\frac{\|x-y\|_{\infty}}{\|x\|_{\infty}} \f$*/
double compute_error_l_infty(gpuComplex *x, gpuComplex *y, int n)
{
  return (l_infty(x, y, n)/l_infty(x, NULL, n));
}



