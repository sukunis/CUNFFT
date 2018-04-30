/*
 * simpleTest.cpp
 * Examples for usinf CUNFFT lib
 *  Created on: 23.07.2015
 *      Author: sukunis
 */

#include <stdarg.h>
#include <float.h>
#include <unistd.h> // get current dir
#include "cunfft_util.h"
#include "cunfft.h"

#include <limits.h>
#include <float.h>


void simple_test_cunfft_1d()
{
	resetDevice();
	uint_t N[3],M;
	N[0]=1<<5;N[1]=1<<0;N[2]=1<<0;
	M=N[0];
	int numOfRuns=1;
	cunfft_plan p;

	cunfft_init(&p,1,N,M);
	getExampleData_uniDistr(&p);

printf("CUNDFT(%dD):\n",p.d);
	GPUTimer t=getTimeGPU();
	copyDataToDevice(&p);
	cundft_transform(&p);
	copyDataToHost(&p);
	double runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f,32,"vector f (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");
	
printf("CUNFFT(%dD):\n",p.d);
	cunfft_reinit(&p);
	t=getTimeGPU();
	copyDataToDevice(&p);
	cunfft_transform(&p);
	copyDataToHost(&p);
	runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f,32,"vector f (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");

	// CUNDFT adjoint
printf("CUNDFT(%dD) adjoint:\n",p.d);
	cunfft_reinitAd(&p);
	t=getTimeGPU();
	copyDataToDeviceAd(&p);
	cundft_adjoint(&p);
	copyDataToHostAd(&p);
	runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f_hat,32,"vector f_hat (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");

	//CUNFFT adjoint
printf("CUNFFT(%dD) adjoint:\n",p.d);
	cunfft_reinitAd(&p);
	t=getTimeGPU();
	copyDataToDeviceAd(&p);
	cunfft_adjoint(&p);
	copyDataToHostAd(&p);
	runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f_hat,32,"vector f_hat (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");

	cunfft_finalize(&p);
}

void simple_test_cunfft_2d()
{
	resetDevice();
	uint_t N[3],M;
	N[0]=32;N[1]=32;
	M=N[0]*N[1];
	cunfft_plan p;
int numOfRuns=1;
	cunfft_init(&p,2,N,M);
	getExampleData_uniDistr(&p);


	// CUNDFT
printf("CUNDFT(%dD):\n",p.d);
	GPUTimer t=getTimeGPU();
	copyDataToDevice(&p);
	cundft_transform(&p);
	copyDataToHost(&p);
	double runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f,32,"vector f (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");


	//CUNFFT
printf("CUNFFT(%dD):\n",p.d);
	cunfft_reinit(&p);
	t=getTimeGPU();
	copyDataToDevice(&p);
	cunfft_transform(&p);
	copyDataToHost(&p);
	runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f,32," vector f (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");

	// CUNDFT adjoint
printf("CUNDFT(%dD) adjoint:\n",p.d);
	cunfft_reinitAd(&p);
	t=getTimeGPU();
	copyDataToDeviceAd(&p);
	cundft_adjoint(&p);
	copyDataToHostAd(&p);
	runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f_hat,32,"vector f_hat (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");

	//CUNFFT adjoint
printf("CUNFFT(%dD) adjoint:\n",p.d);
	cunfft_reinitAd(&p);
	t=getTimeGPU();
	copyDataToDeviceAd(&p);
	cunfft_adjoint(&p);
	copyDataToHostAd(&p);
	runTime = elapsedGPUTime(t,getTimeGPU());
	showCoeff_cuComplex(p.f_hat,32,"vector f_hat (first few entries)");
	p.CUNFFTTimes.runTime=runTime;
	showTimes(&p.CUNFFTTimes,numOfRuns);
	printf("\n\n\n");


	cunfft_finalize(&p);
}

int main(int argc, char** argv)
{
	printf("1) computing a one dimensional cundft, cunfft and an adjoint cunfft\n\n");
	simple_test_cunfft_1d();

	getc(stdin);

	printf("2) computing a two dimensional cundft, cunfft and an adjoint cunfft\n\n");
	simple_test_cunfft_2d();

	return EXIT_SUCCESS;

}
