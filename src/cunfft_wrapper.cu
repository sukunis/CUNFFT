/*
 * cunfft_wrapper.cu
 *
 *  Created on: 23.06.2014
 *      Author: sukunis
 */



#include "cunfft_kernel.cuh"

#ifdef COM_FG_PSI
#define CONF_STRING " FG_PSI "
#else
#define CONF_STRING ""
#endif

#ifdef PRINT_CONFIG
/** If PRINT_CONFIG is defined: Print kernel configuration on stdout*/
#define CONFIG_OUT(text,gDim,bDim,d) config_out(text,gDim,bDim,d)
#else
/** If PRINT_CONFIG is undefined: do nothing*/
#define CONFIG_OUT(text,gDim,bDim,d) printf("...")
#endif 




//------------------------------------------------------------------------------
// ROC
//------------------------------------------------------------------------------
double cuKernelRolloffCorrection(cunfft_plan *plan)
{
	cudaVerify(cudaMemset(plan->g_gpu,0,sizeof(gpuComplex)*plan->n_total));

	dim3 blockDim,gridDim;
	double t=0.0;GPUTimer t0;
	uint_t off_N[3], off_n[3];
	for(int i=0; i<plan->d; i++){
		off_N[i]=plan->N[i]/2;
		off_n[i]=plan->n[i]-off_N[i];
	}

	switch (plan->d) {
	case 1:
		getComputationGridParams_Array(&blockDim, &gridDim, off_N[0]);
		CONFIG_OUT("kernelRollOf", gridDim, blockDim,plan->d);
		T_GPU(t0);
		cudaVerifyKernel((kernelRollOf_1d<<<gridDim, blockDim>>>
				(plan->fhat_gpu,plan->g_gpu,off_n[0],off_N[0])));
		t=T_GPU_DIFF(t0);
		cudaVerify(cudaThreadSynchronize());
		break;
	case 2:
		getComputationGridParams_Matrix(&blockDim, &gridDim,off_N[0], off_N[1]);
		CONFIG_OUT("kernelRollOf", gridDim, blockDim,plan->d);
		T_GPU(t0);
		cudaVerifyKernel((kernelRollOf_2d<<<gridDim, blockDim>>>
				(plan->fhat_gpu,plan->g_gpu,off_n[0],off_n[1],off_N[0],off_N[1])));
		t=T_GPU_DIFF(t0);
				cudaVerify(cudaThreadSynchronize());
		break;
	default:
		getComputationGridParams_Matrix(&blockDim, &gridDim,off_N[1], off_N[2]);
		CONFIG_OUT("kernelRollOf", gridDim, blockDim,plan->d);
		T_GPU(t0);
		cudaVerifyKernel((kernelRollOf_3d<<<gridDim, blockDim>>>
				(plan->fhat_gpu,plan->g_gpu,off_n[0],off_n[1],off_n[2],off_N[0],
						off_N[1],off_N[2])));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	}
	return t;
}

double cuKernelRollOffCorrection_adjoint(cunfft_plan *plan)
{
	cudaVerify(cudaMemset(plan->fhat_gpu,0,sizeof(gpuComplex)*plan->N_total));
	dim3 blockDim,gridDim;
	double t=0.0;
	uint_t off_N[3], off_n[3];
	GPUTimer t0;
	for(int i=0; i<plan->d; i++){
		off_N[i]=plan->N[i]/2;
		off_n[i]=plan->n[i]-off_N[i];
	}

	switch (plan->d) {
	case 1:
		getComputationGridParams_Array(&blockDim, &gridDim,  off_N[0]);
		CONFIG_OUT("kernelRollOf__adjoint", gridDim, blockDim,plan->d);
		T_GPU(t0);
		cudaVerifyKernel((kernelRollOf_adjoint_1d<<<gridDim, blockDim>>>
				(plan->g_gpu,plan->fhat_gpu,off_n[0],off_N[0])));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	case 2:
		getComputationGridParams_Matrix(&blockDim, &gridDim,  off_N[0], off_N[1]);

		CONFIG_OUT("kernelRollOf__adjoint", gridDim, blockDim,plan->d);
		T_GPU(t0);
		cudaVerifyKernel((kernelRollOf_adjoint_2d<<<gridDim, blockDim>>>
				(plan->g_gpu,plan->fhat_gpu,off_n[0],off_n[1],off_N[0],off_N[1])));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	default:
		getComputationGridParams_Matrix(&blockDim, &gridDim,  off_N[1], off_N[2]);
		CONFIG_OUT("kernelRollOf__adjoint", gridDim, blockDim,plan->d);
		T_GPU(t0);
		cudaVerifyKernel((kernelRollOf_adjoint_3d<<<gridDim, blockDim>>>
				(plan->g_gpu,plan->fhat_gpu,off_n[0],off_n[1],off_n[2],
						off_N[0],off_N[1],off_N[2])));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	}
	return t;
}
//------------------------------------------------------------------------------
// CONVOLUTION
//------------------------------------------------------------------------------
#ifdef TEXTURE
#define ALLOC_TEXTURE allocateTexture1DFetch(plan->g_gpu,plan->n_total)
#define FREE_TEXTURE freeTextureFetch()
#define CONV_1D_PARA (plan->x_gpu,plan->f_gpu,plan->M_total)
#define CONV_2D_PARA (plan->x_gpu,plan->f_gpu,plan->M_total)
#define CONV_3D_PARA (plan->x_gpu,plan->f_gpu,plan->M_total,plan->n[2]*plan->n[1])
#else
#define ALLOC_TEXTURE
#define FREE_TEXTURE
#define CONV_1D_PARA (plan->g_gpu,plan->x_gpu,plan->f_gpu,plan->M_total)
#define CONV_2D_PARA (plan->g_gpu,plan->x_gpu,plan->f_gpu,plan->M_total)
#define CONV_3D_PARA (plan->g_gpu,plan->x_gpu,plan->f_gpu,plan->M_total,plan->n[2]*plan->n[1])
#endif

double cuConvolution(cunfft_plan *plan)
{
	double t=0.0;//return value

	dim3 blockDim,gridDim;
	getComputationGridParams_Array(&blockDim, &gridDim,  plan->M_total);

	GPUTimer t0;

	char str[25]="convolution";
	strcat(str,CONF_STRING);
	CONFIG_OUT(str, gridDim, blockDim,plan->d);

	ALLOC_TEXTURE;
	switch (plan->d) {
	case 1:
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_1d<<<gridDim,blockDim>>>CONV_1D_PARA));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	case 2:
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_2d<<<gridDim,blockDim>>>CONV_2D_PARA));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	default:
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_3d<<<gridDim,blockDim>>>CONV_3D_PARA));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	}
	FREE_TEXTURE;

	return t;
}

double cuConvolution_adjoint(cunfft_plan *plan)
{
	double t=0.0;
	cudaVerify(cudaMemset(plan->g_gpu,0,sizeof(gpuComplex)*plan->n_total));
	dim3 blockDim,gridDim;
	getComputationGridParams_Array(&blockDim, &gridDim,  plan->M_total);
	char str[35]="convolution adjoint";
	strcat(str,CONF_STRING);
	CONFIG_OUT(str, gridDim, blockDim,plan->d);

	GPUTimer t0;

#ifdef COM_FG_PSI
	switch (plan->d) {
	case 1:
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_adjoint_1d<<<gridDim,blockDim>>>
				(plan->g_gpu,plan->x_gpu,plan->f_gpu,plan->M_total)));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	case 2:
		allocateTexture1DFetch(plan->f,plan->M_total);
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_adjoint_2d<<<gridDim,blockDim>>>
				(plan->g_gpu,plan->x_gpu,plan->M_total)));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		freeTextureFetch();
		break;
	default:
		allocateTexture1DFetch(plan->f,plan->M_total);
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_adjoint_3d<<<gridDim,blockDim>>>
				(plan->g_gpu,plan->x_gpu,plan->M_total)));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		freeTextureFetch();
		break;
	}
#else

	//allocateTexture1DFetch(plan->f,plan->M_total);
	switch (plan->d) {
	case 1:
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_adjoint_1d<<<gridDim,blockDim>>>
				(plan->f_gpu,plan->g_gpu,plan->x_gpu,plan->M_total)));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	case 2:
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_adjoint_2d<<<gridDim,blockDim>>>
				(plan->f_gpu,plan->g_gpu,plan->x_gpu,plan->M_total)));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	default:
		T_GPU(t0);
		cudaVerifyKernel((kernelConvWithGauss_adjoint_3d<<<gridDim,blockDim>>>
				(plan->f_gpu,plan->g_gpu,plan->x_gpu,plan->M_total)));
		t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
		break;
	}
	//freeTextureFetch();
#endif
	return t;
}
//------------------------------------------------------------------------------
// FFT
//------------------------------------------------------------------------------

void cuFFT_DestroyPlan(cunfft_plan *p)
{
	if(p->fft_backward){
		cufftVerify(cufftDestroy(p->fft_backward));
	}
	if(p->fft_forward){
		cufftVerify(cufftDestroy(p->fft_forward));
	}
	cudaVerify(cudaThreadSynchronize());
	p->fft_backward=0;
	p->fft_forward=0;
}

void cuFFT_createPlan(int dim,uint_t *n,cufftHandle *h)
{
	int BATCH=1;
	cufftType type;

#ifdef CUNFFT_DOUBLE_PRECISION
	type=CUFFT_Z2Z;
#else
	type=CUFFT_C2C;
#endif

	cufftCreate(h);
	switch (dim) {
		case 2: cufftVerify(cufftPlan2d(h,n[0],n[1],type));
			break;
		case 3:cufftVerify(cufftPlan3d(h,n[0],n[1],n[2],type));
			break;
		default:cufftVerify(cufftPlan1d(h,n[0],type,BATCH));
			break;
	}
	cudaVerify(cudaThreadSynchronize());
}

double cuFFT(gpuComplex *d_g,cufftHandle h)
{
	double t=0.0;
	GPUTimer t0;
	// create  FFT plan
#ifdef CUNFFT_DOUBLE_PRECISION
	// execute FFT
	T_GPU(t0);
	cufftVerify(cufftExecZ2Z(h,d_g,d_g,CUFFT_FORWARD));
	t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
#else
	// execute FFT
	T_GPU(t0);
	cufftVerify(cufftExecC2C(h,d_g,d_g,CUFFT_FORWARD));
	t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());

#endif
	return t;
}

double cuFFT_adjoint(gpuComplex *d_g,cufftHandle h)
{
	double t=0.0;
	GPUTimer t0;
	// create FFT plan
#ifdef CUNFFT_DOUBLE_PRECISION
	// execute FFT
	T_GPU(t0);
	cufftVerify(cufftExecZ2Z(h,d_g,d_g,CUFFT_INVERSE));
	t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
#else
	// execute FFT
	T_GPU(t0);
	cufftVerify(cufftExecC2C(h,d_g,d_g,CUFFT_INVERSE));
	t=T_GPU_DIFF(t0);
						cudaVerify(cudaThreadSynchronize());
#endif
	return t;
}
//------------------------------------------------------------------------------
// 		NDFT
//------------------------------------------------------------------------------
void cuNDFT(cunfft_plan *plan)
{
	dim3 blockDim,gridDim;
	getComputationGridParams_Array(&blockDim, &gridDim,	plan->M_total);
	CONFIG_OUT("kernelndft", gridDim, blockDim,plan->d);
	switch (plan->d) {
	case 1:
		cudaVerifyKernel((kernelNDFT_1d<<<gridDim,blockDim>>>
				(plan->x_gpu,plan->fhat_gpu,plan->f_gpu,plan->M_total,
						plan->N_total)));
		break;
	case 2:
		cudaVerifyKernel((kernelNDFT_2d<<<gridDim,blockDim>>>
				(plan->x_gpu,plan->fhat_gpu,plan->f_gpu,plan->M_total,
						plan->N_total)));
		break;
	default:
		cudaVerifyKernel((kernelNDFT_3d<<<gridDim,blockDim>>>
				(plan->x_gpu,plan->fhat_gpu,plan->f_gpu,plan->M_total,
						plan->N_total)));
		break;
	}
}

void cuNDFT_adjoint(cunfft_plan *plan)
{
	dim3 blockDim,gridDim;
	getComputationGridParams_Array(&blockDim, &gridDim,	plan->N_total);

	CONFIG_OUT("kernelndft_adjoint", gridDim, blockDim,plan->d);
	switch (plan->d) {
	case 1:
		cudaVerifyKernel((kernelNDFT_adjoint_1d<<<gridDim,blockDim>>>
				(plan->x_gpu,plan->fhat_gpu,plan->f_gpu,plan->M_total,
						plan->N_total)));
		break;
	case 2:
		cudaVerifyKernel((kernelNDFT_adjoint_2d<<<gridDim,blockDim>>>
				(plan->x_gpu,plan->fhat_gpu,plan->f_gpu,plan->M_total,
						plan->N_total)));
		break;
	default:
		cudaVerifyKernel((kernelNDFT_adjoint_3d<<<gridDim,blockDim>>>
				(plan->x_gpu,plan->fhat_gpu,plan->f_gpu,plan->M_total,
						plan->N_total)));
		break;
	}
}



