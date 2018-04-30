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
 * $Id: cunfft.cu 2012-06-13 11:36:00Z sukunis $
 */

/**
 * @file cunfft.cu
 * @brief CUDA calling functions source file
 */

#include "cunfft_kernel.cuh"
//------------------------------------------------------------------------------
// 		FG_PSI COMPUTATION FUNCTION
//------------------------------------------------------------------------------
#ifdef COM_FG_PSI
/** If COM_FG_PSI defined: precomputation for gaussian window*/
#define FG_PSI_PRECOMPUTATION(a,dim){					\
	(a)=(double*)calloc(FILTER_SIZE*MAX_DIM,sizeof(double));\
	nfft_init_fg_exp_l((a),plan->b[0],0);			\
	if((dim)==2){										\
		nfft_init_fg_exp_l((a), plan->b[1],FILTER_SIZE);	\
	}													\
	if((dim)==3){										\
		nfft_init_fg_exp_l((a), plan->b[1],FILTER_SIZE);	\
		nfft_init_fg_exp_l((a),plan->b[2],2*FILTER_SIZE);\
	}												\
}
/** If COM_FG_PSI defined: copy precomputed value to device*/
#define FG_PSI_COPY(a){								\
	setPREexp((a), sizeof(double)*FILTER_SIZE*(MAX_DIM));\
}
/** If COM_FG_PSI defined: free precomputed data*/
#define FG_PSI_FREE(a)	{free((a));}

#else
/** If COM_FG_PSI undefined: do nothing*/
#define FG_PSI_PRECOMPUTATION(a,dim)
/** If COM_FG_PSI undefined: do nothing*/
#define FG_PSI_COPY(a)
/** If COM_FG_PSI undefined: do nothing*/
#define FG_PSI_FREE(a)
#endif

void nfft_init_fg_exp_l(double *fg_exp_l,const double b,
		const int offset)
{
  int l;
  double fg_exp_b0, fg_exp_b1, fg_exp_b2, fg_exp_b0_sq;

  fg_exp_b0 = exp(-1.0/b);
  fg_exp_b0_sq = fg_exp_b0*fg_exp_b0;
  fg_exp_b1 = 1.0;
  fg_exp_b2 = 1.0;
  fg_exp_l[0+offset] = 1.0;
  for(l=1; l < FILTER_SIZE; l++){
	  fg_exp_b2 = fg_exp_b1*fg_exp_b0;
	  fg_exp_b1 *= fg_exp_b0_sq;
	  fg_exp_l[l+offset] = fg_exp_l[l-1+offset]*fg_exp_b2;
  }
}


void setProperties(uint_t *N,uint_t *n,double *b,double *sigma,int dim,kdata *k_c)
{
	int i;
	for(i=0; i<dim;i++){
		k_c->N[i]=N[i];
		k_c->n[i]=n[i];
		k_c->b[i]=b[i];
		k_c->bsqrt[i]=sqrt(b[i]);
		k_c->fac[i]=(M_PI*M_PI*b[i])/(n[i]*n[i]);
		k_c->sigma[i]=sigma[i];
	}
}


void warmUp_createPlan(uint_t *n)
{
	cufftHandle fft_backward;
	int BATCH=1;
	uint_t NX=n[0];
	cufftVerify(cufftPlan1d(&fft_backward,NX,CUFFT_Z2Z,BATCH));
	cufftVerify(cufftDestroy(fft_backward));
	cudaVerify(cudaThreadSynchronize());
}


//------------------------------------------------------------------------------
// 		MEMORY TRANSFER FUNCTION
//------------------------------------------------------------------------------
void copyDataToDevice(cunfft_plan *plan)
{
	kdata k_cpu;
	setProperties(plan->N, plan->n, plan->b, plan->sigma,plan->d,&k_cpu);

	GPUTimer t0;
	T_GPU(t0);

	copyProperties(&k_cpu);
	FG_PSI_COPY(plan->fg_exp_l_cpu)
	cudaVerify(cudaMemcpy(plan->fhat_gpu,plan->f_hat,
			sizeof(gpuComplex)*plan->N_total,cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(plan->x_gpu,plan->x,
		plan->M_total*plan->d*sizeof(dTyp),cudaMemcpyHostToDevice));
	cudaVerify(cudaThreadSynchronize());
	plan->CUNFFTTimes.time_COPY_IN=T_GPU_DIFF(t0);
}

void copyDataToDeviceAd(cunfft_plan *plan)
{
	kdata k_cpu;
	setProperties(plan->N, plan->n, plan->b, plan->sigma,plan->d,&k_cpu);

	GPUTimer t0;
	T_GPU(t0);

	copyProperties(&k_cpu);
	FG_PSI_COPY(plan->fg_exp_l_cpu);
	cudaVerify(cudaMemcpy(plan->f_gpu,plan->f,
			sizeof(gpuComplex)*plan->M_total,cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(plan->x_gpu,plan->x,
			plan->M_total*plan->d*sizeof(dTyp),cudaMemcpyHostToDevice));
	cudaVerify(cudaThreadSynchronize());
	plan->CUNFFTTimes.time_COPY_IN=T_GPU_DIFF(t0);
}

void copyDataToHost(cunfft_plan *plan)
{
	GPUTimer t0;
	T_GPU(t0);

	cudaVerify(cudaMemcpy(plan->f,plan->f_gpu,plan->M_total*sizeof(gpuComplex),
			cudaMemcpyDeviceToHost));
	cudaVerify(cudaThreadSynchronize());
	plan->CUNFFTTimes.time_COPY_OUT=T_GPU_DIFF(t0);
}

void copyDataToHostAd(cunfft_plan *plan)
{
	GPUTimer t0;
	T_GPU(t0);

	cudaVerify(cudaMemcpy(plan->f_hat,plan->fhat_gpu,plan->N_total*sizeof(gpuComplex),
			cudaMemcpyDeviceToHost));
	cudaVerify(cudaThreadSynchronize());
	plan->CUNFFTTimes.time_COPY_OUT=T_GPU_DIFF(t0);
}

void copy_f_ToHost(cunfft_plan *plan, gpuComplex *arr)
{
	cudaVerify(cudaMemcpy(arr,plan->f_gpu,plan->M_total*sizeof(gpuComplex),
			cudaMemcpyDeviceToHost));
	cudaVerify(cudaThreadSynchronize());
}

void copy_g_ToHost(cunfft_plan *plan)
{
	cudaVerify(cudaMemcpy(plan->g,plan->g_gpu,plan->n_total*sizeof(gpuComplex),
			cudaMemcpyDeviceToHost));
	cudaVerify(cudaThreadSynchronize());
}
void copy_f_hat_ToHost(cunfft_plan *plan, gpuComplex *arr)
{
	cudaVerify(cudaMemcpy(arr,plan->fhat_gpu,plan->N_total*sizeof(gpuComplex),
			cudaMemcpyDeviceToHost));
	cudaVerify(cudaThreadSynchronize());
}

void copy_x_toHost(cunfft_plan *plan, dTyp *arr)
{
	cudaVerify(cudaMemcpy(arr,plan->x_gpu,plan->M_total*plan->d*sizeof(dTyp),
			cudaMemcpyDeviceToHost));
	cudaVerify(cudaThreadSynchronize());
}

//------------------------------------------------------------------------------
// 		CUNFFT
//------------------------------------------------------------------------------

void cunfft_transform(cunfft_plan *plan)
{
	double t=0.0;
	// configuration device
	cudaVerify(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	t=cuKernelRolloffCorrection(plan);
	plan->CUNFFTTimes.time_ROC=t;
	t=cuFFT(plan->g_gpu,plan->fft_forward);
	plan->CUNFFTTimes.time_FFT=t;
	t=cuConvolution(plan);
	plan->CUNFFTTimes.time_CONV=t;
}

void cunfft_adjoint(cunfft_plan *plan)
{
	// configuration device
	cudaVerify(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	double t=0.0;

	t=cuConvolution_adjoint(plan);
	plan->CUNFFTTimes.time_CONV=t;

	t=cuFFT_adjoint(plan->g_gpu,plan->fft_backward);
	plan->CUNFFTTimes.time_FFT=t;

	t=cuKernelRollOffCorrection_adjoint(plan);
	plan->CUNFFTTimes.time_ROC=t;

}

void cunfft_initPlan(int dim,uint_t *n,cufftHandle *h)
{
	cuFFT_createPlan(dim,n,h);
}

void cunfft_freePlan(cunfft_plan *p)
{
	cuFFT_DestroyPlan(p);
}

void cunfft_reinit(cunfft_plan *plan)
{
resetCUNFFTTimes(&plan->CUNFFTTimes);
	cunfft_finalizeDevice(plan);

	memset(plan->f,0,plan->M_total*sizeof(gpuComplex));
	memset(plan->g,0,plan->n_total*sizeof(gpuComplex));

	cunfft_initDevicePara(plan);

	cudaVerify(cudaThreadSynchronize());
}

void cunfft_reinitAd(cunfft_plan *plan)
{
resetCUNFFTTimes(&plan->CUNFFTTimes);
	cunfft_finalizeDevice(plan);

	cudaVerify(cudaMallocHost((void**)&plan->f_hat,plan->N_total*sizeof(gpuComplex)));
	cudaVerify(cudaMallocHost((void**)&plan->g,plan->n_total*sizeof(gpuComplex)));

	cunfft_initDevicePara(plan);

	cudaVerify(cudaThreadSynchronize());
}

void cunfft_initDevicePara(cunfft_plan *plan)
{
	// GPU Parameter ---------------------------------
	cudaVerify(cudaMalloc((void**)&plan->g_gpu,	sizeof(gpuComplex)*plan->n_total));
	cudaVerify(cudaMemset(plan->g_gpu,0,sizeof(gpuComplex)*plan->n_total));

	cudaVerify(cudaMalloc((void**)&plan->fhat_gpu,sizeof(gpuComplex)*plan->N_total));
	cudaVerify(cudaMemset(plan->fhat_gpu,0,sizeof(gpuComplex)*plan->N_total));

	cudaVerify(cudaMalloc((void**)&plan->x_gpu,	plan->M_total*plan->d*sizeof(dTyp)));
	cudaVerify(cudaMemset(plan->x_gpu,0,sizeof(dTyp)*plan->M_total*plan->d));

	cudaVerify(cudaMalloc((void**)&plan->f_gpu,	plan->M_total*sizeof(gpuComplex)));
	cudaVerify(cudaMemset(plan->f_gpu,0,sizeof(gpuComplex)*plan->M_total));

	kdata k_cpu;
	setProperties(plan->N, plan->n, plan->b, plan->sigma,plan->d,&k_cpu);

}

void cunfft_init(cunfft_plan *plan,int dim, uint_t *N, uint_t M_total)
{
	int t;

	// CPU Parameter ---------------------------
	plan->d=dim;
	plan->N=(uint_t*)malloc(plan->d*sizeof(uint_t));
	for(t=0;t<dim;t++){
		plan->N[t] = N[t];
	}

	plan->M_total = M_total;
	plan->m=CUT_OFF; // N>2*m!!!

	plan->n= (uint_t*) malloc(dim*sizeof(uint_t));
	for(t=0; t<dim; t++){
		plan->n[t]=2*next_power_of_2(plan->N[t]);
	}

	//Set  flags
	plan->flags=CUFFT_INIT | CUFFT_BOTH;

	cunfft_initHelp(plan);


}

void cunfft_initGuru(cunfft_plan *plan,int dim, uint_t *N, uint_t M_total,
		uint_t *n, unsigned flags)
{
	int t;

	// CPU Parameter ---------------------------
	plan->d=dim;
	plan->N=(uint_t*)malloc(plan->d*sizeof(uint_t));
	for(t=0;t<dim;t++){
		plan->N[t] = N[t];
	}

	plan->M_total = M_total;
	plan->m=CUT_OFF; // N>2*m!!!

	plan->n= (uint_t*) malloc(dim*sizeof(uint_t));
	for(t=0; t<dim; t++){
		plan->n[t]=n[t];
	}

	plan->flags=flags;

	cunfft_initHelp(plan);
}

void cunfft_initHelp(cunfft_plan *plan)
{
	int t;
	plan->N_total = prod_int(plan->N,plan->d);
	plan->n_total = prod_int(plan->n,plan->d);

	plan->sigma = (double*) malloc(plan->d*sizeof(double));
	for(t = 0;t < plan->d; t++){
		plan->sigma[t] = ((double)plan->n[t])/plan->N[t];
	}

	// parameter to define the width of the gauss
	plan->b = (double*) malloc(plan->d*sizeof(double));
	for(t=0; t<plan->d; t++){
		plan->b[t]=((double)2*plan->sigma[t])/
				(2*plan->sigma[t]-1)*(((double)plan->m) / M_PI);
	}

	FG_PSI_PRECOMPUTATION(plan->fg_exp_l_cpu,plan->d)

	resetCUNFFTTimes(&plan->CUNFFTTimes);

	cudaVerify(cudaMallocHost((void**)&plan->x,plan->M_total*plan->d*sizeof(dTyp)));
	cudaVerify(cudaMallocHost((void**)&plan->f_hat,plan->N_total*sizeof(gpuComplex)));

	cudaVerify(cudaMallocHost((void**)&plan->f,	plan->M_total*sizeof(gpuComplex)));
	cudaVerify(cudaMallocHost((void**)&plan->g,	plan->n_total*sizeof(gpuComplex)));

	// GPU Parameter ---------------------------------
	plan->fft_backward=0;
	plan->fft_forward=0;
	cunfft_initDevicePara(plan);
	cudaVerify(cudaThreadSynchronize());

	if((plan->flags & CUFFT_INIT)){
		if(plan->flags & CUFFT_BOTH){
			cunfft_initPlan(plan->d,plan->n,&plan->fft_forward);
			cunfft_initPlan(plan->d,plan->n,&plan->fft_backward);
		}
		if(plan->flags & CUFFT_FORW)
			cunfft_initPlan(plan->d,plan->n,&plan->fft_forward);
		if(plan->flags & CUFFT_BACKW)
			cunfft_initPlan(plan->d,plan->n,&plan->fft_backward);
	}
}


void cunfft_finalizeDevice(cunfft_plan *plan)
{
	// free gpu memory //
	cudaVerify(cudaFree(plan->x_gpu));
	cudaVerify(cudaFree(plan->f_gpu));
	cudaVerify(cudaFree(plan->fhat_gpu));
	cudaVerify(cudaFree(plan->g_gpu));
	cudaVerify(cudaThreadSynchronize());

}

void cunfft_finalize(cunfft_plan *plan)
{
	if(plan->flags & CUFFT_INIT)
			cunfft_freePlan(plan);
	cudaVerify(cudaFreeHost(plan->g));
	cudaVerify(cudaFreeHost(plan->f));
	FG_PSI_FREE(plan->fg_exp_l_cpu)

	// free cpu mem
	cudaVerify(cudaFreeHost(plan->x));
	cudaVerify(cudaFreeHost(plan->f_hat));
	free(plan->sigma);
	free(plan->n);
	free(plan->N);
	free(plan->b);
	cunfft_finalizeDevice(plan);
	cudaVerify(cudaThreadSynchronize());
	cudaVerify(cudaDeviceReset());
}


void resetCUNFFTTimes(NFFTTimeSpec *CUNFFTTimes)
{
	CUNFFTTimes->runTime=0.0;
	CUNFFTTimes->time_CONV=0.0;
	CUNFFTTimes->time_COPY_IN=0.0;
	CUNFFTTimes->time_COPY_OUT=0.0;
	CUNFFTTimes->time_FFT=0.0;
	CUNFFTTimes->time_ROC=0.0;
}

//------------------------------------------------------------------------------
// 		CUNDFT
//------------------------------------------------------------------------------
void cundft_transform(cunfft_plan *plan)
{
	// configuration device
	cudaVerify(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	cuNDFT(plan);
}

void cundft_adjoint(cunfft_plan *plan)
{
	// configuration device
	cudaVerify(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	cuNDFT_adjoint(plan);
}


void resetDevice()
{
	cudaVerify(cudaDeviceReset());
}







