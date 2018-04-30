/*
 * timeMeasurement.h
 *
 *  Created on: 07.07.2015
 *      Author: sukunis
 */

#ifndef TIMEMEASUREMENT_H_
#define TIMEMEASUREMENT_H_


#include <cuda_runtime_api.h>
#include <time.h>

/** Timer struct for wallclock time */
typedef struct timespec Timer;

typedef cudaEvent_t GPUTimer;
typedef Timer CPUTimer;

/** Timer struct for several transformation steps */
typedef struct{
	/** runtime of whole transformation*/
	double runTime;
	/** runtime of ROC step */
	double time_ROC;
	/** runtime of Convolution step */
	double time_CONV;
	/** runtime of FFT step */
	double time_FFT;
	/** runtime of copy data host -> device */
	double time_COPY_IN;
	/** runtime of copy data device -> host */
	double time_COPY_OUT;
	double time_else;/**< placeholder for measurement user specific time */
}NFFTTimeSpec;

#ifdef MEASURED_TIMES
#define T_CPU(t) t=getTimeCPU()
#define T_GPU(t) t=getTimeGPU()
#define T_GPU_DIFF(t) elapsedGPUTime(t,getTimeGPU());
#define T_CPU_DIFF(t) elapsedCPUTime(t,getTimeCPU());
#else
#define T_CPU(t) t=NULL
#define T_GPU(t) t=NULL
#define T_GPU_DIFF(t) 0.0
#define T_CPU_DIFF(t) 0.0
#endif


static __inline__ void resetNFFTTimeSpec(NFFTTimeSpec* timerSpec)
{
	timerSpec->runTime=0.0;
	timerSpec->time_ROC=0.0;
	timerSpec->time_CONV=0.0;
	timerSpec->time_FFT=0.0;
	timerSpec->time_COPY_IN=0.0;
	timerSpec->time_COPY_OUT=0.0;
	timerSpec->time_else=0.0;
}

/** timing with gettimeofday uses the system time for timing.
 * Accurate within 10us on average.
 */
//typedef struct timeval CPUTimer;
//static __inline__ CPUTimer getTimeCPU()
//{
//	CPUTimer time;
//	gettimeofday(&time,NULL);
//	return time;
//}
///** Return time difference in seconds*/
//static __inline__ double elapsedCPUTime(CPUTimer start, CPUTimer stop)
//{
//	return (stop.tv_sec-start.tv_sec) +
//				(stop.tv_nsec-start.tv_nsec) *1e-6;
//}


/** timing with clock_gettime uses the number of cycles that have passed on the
 * CPU for timing. Accurate within 1ns on average (clock rate of the CPU).
 * Must link with the realtime library when compiling (-lrt).
 */
static __inline__ CPUTimer getTimeCPU()
{
	CPUTimer time;
	clock_gettime(CLOCK_MONOTONIC,&time);
	return time;
}

/**Use Profiling for timing:
 *
 * enable by setting COMPUTE_PROFILE environment variable to 1
 * - export COMPUTE_PROFILE=1 # bash
 * - setenv COMPUTE_PROFILE 1 # csh
 * Execute your code normally.
 * One or more profile logs will be generated.
 *
 */

/** Events are special kernels that can be invoked for precise timing
 * on the GPU. Resolution 0.5 us (operating-system-independent)
 *
 * The device will record a timestamp for the event when it
 * reaches that event in the stream.
 */
static __inline__ GPUTimer getTimeGPU()
{
	GPUTimer time;
	cudaEventCreate(&time);
	// place the start or end events into the default stream
	cudaEventRecord(time,0);
	return time;
}

/** Return time difference in seconds*/
static __inline__ double elapsedCPUTime(CPUTimer start, CPUTimer stop)
{
	return (stop.tv_sec-start.tv_sec) +
				(stop.tv_nsec-start.tv_nsec) *1e-9;
}

/** Return time difference in seconds*/
static __inline__ double elapsedGPUTime(GPUTimer start, GPUTimer stop)
{
	float diffTime=0.0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&diffTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return double(diffTime)/1000;
}


#endif /* TIMEMEASUREMENT_H_ */
