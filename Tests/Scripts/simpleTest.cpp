/*
 * runtimeTest.cpp
 *
 *  Created on: 28.08.2015
 *      Author: sukunis
 */
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
 * $Id: simpleTest.c 2012-06-01 11:36:00Z sukunis $
 */

/**
 * @file simpleTest.cpp
 * @brief Example file for using CUNFFT1.
 */

//#include "myTest.h"

#include <stdarg.h>
#include <unistd.h> // get current dir
#include "cunfft_util.h"
#include "cunfft.h"
#include "ndft.h"
#include <limits.h>
#include <float.h>

// test against ndft on gpu, output: l infinity error
#define CHECK_AGAINST_CUNDFT 1
// test against ndft on cpu, output: l infinity error
#define CHECK_AGAINST_NDFT 0

#define CHECK_AGAINST_NFFT 0

double RuntimeMinCUNFFT;
double RuntimeCUNFFT;
double RuntimeMinNFFT;
double RuntimeNFFT;
double RuntimeCUNDFT;
double RunTimeMedNFFT[10];
double RunTimeMedCUNFFT[10];
double err1,err2;
int counter;



double simple_test_cunfft(uint_t* N, uint_t* n, uint_t M, int m, int dim)
{
	CPUTimer t0;
	GPUTimer tg;
	double t;
	cunfft_plan p;

	cunfft_init(&p,dim,N,M);

	printf("# N_total= " PRINT_FORMAT ", n_total= " PRINT_FORMAT ",M_total= " PRINT_FORMAT ", m= %d\n",
				p.N_total,p.n_total,p.M_total,p.m);

	showCPUMemUse(&p);

//	getExampleData_null(&p);
//		p.f_hat[1].x=1.0; p.f_hat[1].y=1.0;
	getExampleData_uniDistr(&p);

	// copy Data to gpu device memory
	copyDataToDevice(&p);

	// CUNDFT transformation
#if CHECK_AGAINST_CUNDFT
//	RuntimeCUNDFT=0.0;
	gpuComplex *cundft;
	cundft = (gpuComplex *)malloc(p.M_total*sizeof(gpuComplex));
	tg=getTimeGPU();//t0=getTimeCPU();
	cundft_transform(&p);
	RuntimeCUNDFT+=elapsedGPUTime(tg,getTimeGPU());//RuntimeCUNDFT+=elapsedCPUTime(t0,getTimeCPU());
	printf("+ CUNDFT (%dD) \t took %.2e seconds.\n",p.d,RuntimeCUNDFT);
	//copy result to host memory
	copy_f_ToHost(&p,cundft);


	// reset device data
	cunfft_reinit(&p);
	copyDataToDevice(&p);
#endif

	// NDFT transformation
#if CHECK_AGAINST_NDFT
	gpuComplex *myndft;
	myndft = (gpuComplex *)malloc(p.M_total*sizeof(gpuComplex));
	t0=getTimeCPU();
	ndft(&p,myndft);
	printf("+ NDFT (%dD) \t took %.2e seconds.\n",p.d,elapsedCPUTime(t0,getTimeCPU()));
#endif

#if CHECK_AGAINST_NFFT
//	RuntimeNFFT=0.0;
	gpuComplex *mynfft;
	mynfft = (gpuComplex *)malloc(p.M_total*sizeof(gpuComplex));
//	t=run_nfft(N, n, M, m,dim,p.x,p.f_hat,mynfft);
	t=nfft(&p,mynfft);
	printf("+ NFFT (%dD) \t took %.2e seconds.\n",p.d,t);
	if(t<= RuntimeMinNFFT) RuntimeMinNFFT=t;
	RunTimeMedNFFT[counter]=t;
	RuntimeNFFT+=t;
#endif
//RuntimeCUNFFT=0.0;
	tg=getTimeGPU();

	cunfft_transform(&p);
	t=elapsedGPUTime(tg,getTimeGPU());
	printf("+ CUNFFT (%dD) \t took %.2e seconds.\n",p.d,t);
	if(t<= RuntimeMinCUNFFT) RuntimeMinCUNFFT=t;
	RunTimeMedCUNFFT[counter]=t;
	RuntimeCUNFFT+=t;
	p.CUNFFTTimes.runTime=t;

	// copy result to host memory
	copyDataToHost(&p);
	showTimes(&p.CUNFFTTimes,1);

	// print result to stdout
#ifdef SHOW_RESULTS
	showCoeff_cuComplex(p.f,32,"cunfft , vector f (first few entries)");
#if CHECK_AGAINST_CUNDFT
	showCoeff_cuComplex(cundft,32,"cundft , vector f (first few entries)");
#endif
#if CHECK_AGAINST_NDFT
	showCoeff_cuComplex(myndft,32,"ndft , vector f");
#endif
#if CHECK_AGAINST_NFFT
	showCoeff_cuComplex(mynfft,32,"nfft , vector f (first few entries)");
//	NFFT(vpr_complex)(p.f,32,"nfft, vector f");
#endif
#endif

	// compare transformation results

#if CHECK_AGAINST_CUNDFT
	err1=compute_error_l_infty(cundft,p.f,p.M_total);
	printf("\nFEHLER cundft - cunfft =%e \n",err1);
	free(cundft);
#endif

#if CHECK_AGAINST_NDFT
	err1=compute_error_l_infty(myndft,p.f,p.M_total);
	printf("\nFEHLER ndft - cunfft = %e\n",err1);
	free(myndft);
#endif

#if CHECK_AGAINST_NFFT
	err2=compute_error_l_infty(mynfft,p.f,p.M_total);
	printf("\nFEHLER nfft - cunfft = %e\n",err2);
	free(mynfft);
#endif


	// finalize host and device
	cunfft_finalize(&p);
	cudaDeviceReset();

	return err1;//p.CUNFFTTimes.runTime;
}

double simple_test_adjoint(uint_t* N, uint_t* n, uint_t M, int m, int dim)
{
	CPUTimer t0;
	GPUTimer tg;
	double t;
	cunfft_plan p;

	cunfft_init(&p,dim,N,M);
	printf("N_total= " PRINT_FORMAT ", n_total= " PRINT_FORMAT ",M_total= " PRINT_FORMAT ", m= %d\n",
			p.N_total,p.n_total,p.M_total,p.m);

	getExampleDataAd_uniDistr(&p);

	copyDataToDeviceAd(&p);


#if CHECK_AGAINST_CUNDFT
	gpuComplex *cundft;
	cundft = (gpuComplex *)malloc(p.N_total*sizeof(gpuComplex));

	tg=getTimeGPU();
	cundft_adjoint(&p);
	t=elapsedGPUTime(tg,getTimeGPU());
	printf("CUNDFT ADJOINT (%dD) \t took %.2e seconds.\n",p.d,
			t);
	RuntimeCUNDFT+=t;
	//copy result
	copy_f_hat_ToHost(&p,cundft);
	//reset device
	cunfft_reinitAd(&p);
	copyDataToDeviceAd(&p);
#endif


#if CHECK_AGAINST_NDFT
	gpuComplex *myndft;
	myndft = (gpuComplex *)malloc(p.N_total*sizeof(gpuComplex));

	t0=getTimeCPU();
	ndft_adjoint(&p,myndft);
	printf("NDFT ADJOINT (%dD) \t took %.2e seconds.\n",p.d,
				elapsedCPUTime(t0,getTimeCPU()));
#endif
#if CHECK_AGAINST_NFFT
//	t=run_nfft_adjoint(N, n, M, m,dim,p.x,p.f);
	gpuComplex *mynfft;
	mynfft = (gpuComplex *)malloc(p.N_total*sizeof(gpuComplex));
	t=nfft_adjoint(&p,mynfft);
	printf("NFFT ADJOINT (%dD) \t took %.2e seconds.\n",p.d,t);
	if(t<= RuntimeMinNFFT) RuntimeMinNFFT=t;
		RunTimeMedNFFT[counter]=t;
	RuntimeNFFT+=t;
#endif


	tg=getTimeGPU();
	cunfft_adjoint(&p);
	t=elapsedGPUTime(tg,getTimeGPU());
	printf("CUNFFT ADJOINT (%dD) \t took %.2e seconds.\n",p.d,t);
	if(t<= RuntimeMinCUNFFT) RuntimeMinCUNFFT=t;
		RunTimeMedCUNFFT[counter]=t;
	RuntimeCUNFFT+=t;
	p.CUNFFTTimes.runTime=t;

	copyDataToHostAd(&p);
	showTimes(&p.CUNFFTTimes,1);

#ifdef SHOW_RESULTS
#if CHECK_AGAINST_CUNDFT
	showCoeff_cuComplex(cundft,32,"cundft adjoint , vector f_hat (first few entries)");
#endif
#if CHECK_AGAINST_NDFT
	showCoeff_cuComplex(myndft,32,"ndft adjoint , vector f_hat");
#endif
	showCoeff_cuComplex(p.f_hat,32,"cunfft adjoint , vector f_hat (first few entries)");
#endif


#if CHECK_AGAINST_NDFT
	err1=compute_error_l_infty(myndft,p.f_hat,p.N_total);
	printf("\nFEHLER ndft - cunfft = %e\n",err1);
	free(myndft);
#endif
#if CHECK_AGAINST_NFFT
	err2=compute_error_l_infty(mynfft,p.f_hat,p.M_total);
	printf("\nFEHLER nfft - cunfft = %e\n",err2);
	free(mynfft);
#endif
#if CHECK_AGAINST_CUNDFT
	err1=compute_error_l_infty(cundft,p.f_hat,p.N_total);
	printf("\nFEHLER cundft - cunfft =%e \n",err1);
	free(cundft);
#endif

	cunfft_finalize(&p);
	return p.CUNFFTTimes.runTime;
}


int compare (const void * a, const void * b)
{
  double fa = *(double*) a;
  double fb = *(double*) b;
  return (fa > fb) - (fa < fb);
}

double median(double *array,int n)
{
	qsort(array, n, sizeof(double), compare);

	if(n%2==0)
		return (array[n/2]+array[n/2-1])/2;
	else
		return array[n/2];
}

void printHelp()
{
	printf("NAME\n\tsimpleTest - Test vor NFFT on gpu device\n");
	printf("OPTIONS\n\t -h \tPrint usage message\n");
	printf("\t-i \tPrint device informations\n");
	printf("\t-d <val> \tSelect dimension for NFFT. Available dimensions are 1, 2 and 3\n");
printf("\t-N <val1D> <val2D> <val3D> \t Size N of input vector as exponent x of 2^x. Specify only val1D for d=1, val1D and val2D for d=2 and all for d=3\n");
			printf("\t-M <val> \t Size of output\n");
	printf("\nSee also parseInput function in myTest.cpp\n");
}

/** Scan input of format: ./simpleTest -d <d> -N <N1> <N2> <N3> -M <M_total>*/
int parseInput(int argc, char**argv,uint_t* N, uint_t *M, int *d)
{
	int i;
	int res=1;

	for(i=0; i< argc; i++){
		// check for option character
		switch((int)argv[i][0]){
		case '-':
			switch((int)argv[i][1]){
			case 'd': *d=atoi(argv[i+1]); break;
			case 'N': // N
				N[0]=1<<atoi(argv[i+1]);
				N[1]=1<<atoi(argv[i+2]); //printf("N_1= " PRINT_FORMAT "\n",N[1]);
				N[2]=1<<atoi(argv[i+3]);
				break;
//			case 'm'://filterradius
//				*m=atoi(argv[i+1]);
//				break;
			case 'M':
				*M=atoi(argv[i+1]);
				break;
			case 'i': //info
				getGPUMemProps_ToStdout();
				res=0;
				break;
			case 'h':printHelp();
				res=0;break;
			default:printf("no valid argument");
				printHelp();
				res=0;	break;
			}
			break;
		}
	}
	switch(*d){
	case 1: if(N[0]==1){ printf("ERROR: wrong argument for N\n"); res=0;}break;
	case 2: if(N[0]==1 || N[1]==1){
		printf("ERROR: wrong argument for N\n"); res=0;}break;
	case 3: if(N[0]==1 || N[1]==1 || N[2]==1){
		printf("ERROR: wrong argument for N\n"); res=0;}break;
	default: printf("ERROR: wrong argument for dimension!\n"); break;
	}


	return res;
}

#define OUTPUT_FILE "test_runtime.txt"
#define LATEX_1 "\verb+"
#define LATEX_2 "+& \verb+"
#define LATEX_3 "+\\"

int main(int argc, char** argv)
{
	if(1){
		RuntimeMinCUNFFT=1000;
		RuntimeMinNFFT=1000;
		uint_t N[3],n[3],M;
		uint_t N_total;
		int dim;

		int res=parseInput(argc,argv,N,&M,&dim);
		if(!res){return 0;}
		switch (dim) {
		case 1:N_total=N[0];break;
		case 2:	N_total=N[0]*N[1];break;
		default:N_total=N[0]*N[1]*N[2];break;
		}


		resetDevice();
		findDevice();
		getGPUMemProps_ToStdout();

		double err=0.0;
		int maxLoop=1;
		RuntimeCUNDFT=0.0; RuntimeCUNFFT=0.0; RuntimeNFFT=0.0;
		RuntimeMinCUNFFT=1000;RuntimeMinNFFT=1000;
		memset(RunTimeMedCUNFFT,0.0,maxLoop);
		memset(RunTimeMedNFFT,0.0,maxLoop);

		for(counter=0; counter<maxLoop; counter++){
			err+= simple_test_cunfft(N,n,M,CUT_OFF,dim);
//			err+=simple_test_adjoint(N,n,M,CUT_OFF,dim);
		}


		if(1){
			FILE *file;
			file =fopen(OUTPUT_FILE,"a+");
			if(file==NULL){
				printf("\n ERROR: can't open log file for system infos\n");
			}else{
				// print error nfft/cunfft
				//		fprintf(file,"m = %d [d=%d; N_total= " PRINT_FORMAT ";M_total= " PRINT_FORMAT "] %.4e\n",CUT_OFF,dim,N_total,M,err);

				//print runtimes
				//		fprintf(file,"m = %d [d=%d; N_total= " PRINT_FORMAT ";M_total= " PRINT_FORMAT "] %.4e   %.4e\n",
				//				CUT_OFF,dim,N_total,M,RuntimeNFFT,RuntimeCUNFFT);

				//print config
				//		fprintf(file,"m = %d [d=%d; N_total= " PRINT_FORMAT ";M_total= " PRINT_FORMAT "] [%d x %d]  %.4e   %.4e %.4e\n",
				//						CUT_OFF,dim,N_total,M,THREAD_DIM_X,THREAD_DIM_Y,RuntimeNFFT,RuntimeCUNDFT,RuntimeCUNFFT);

				//computation time
#if CHECK_AGAINST_CUNDFT
				fprintf(file," [%d, %d x %d]  %d  %d \t%.4e %.4e %.4e | %.4e %.4e\n",
						CUT_OFF,THREAD_DIM_X,THREAD_DIM_Y,dim,
						(int)(log((double)N_total)/log(2)),
						RuntimeNFFT/maxLoop,
						RuntimeCUNDFT/maxLoop,
						RuntimeCUNFFT/maxLoop,
						err1,err2);
#else

				fprintf(file,"[%d, %d x %d] %d  %d \t%.4e (%.4e) (%.4e) %.4e (%.4e) (%.4e)  | %.4e \n",
						CUT_OFF,THREAD_DIM_X,THREAD_DIM_Y,dim,
						(int)(log((double)N_total)/log(2)),
						RuntimeNFFT/maxLoop,RuntimeMinNFFT,median(RunTimeMedNFFT,maxLoop),
						RuntimeCUNFFT/maxLoop,RuntimeMinCUNFFT,median(RunTimeMedCUNFFT,maxLoop),
						err2);
#endif

			}
			fclose(file);
		}
	}
}






