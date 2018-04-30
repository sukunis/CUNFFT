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
 * $Id: c_util.c 2012-05-31 11:36:00Z sukunis $
 */

/**
 * @file c_util.cpp
 * @brief Utilities for C
 */

#include "cunfft_util.h"


void showCPUMemUse(cunfft_plan *p)
{
	cufftType type;
	#ifdef CUNFFT_DOUBLE_PRECISION
		type=CUFFT_Z2Z;
	#else
		type=CUFFT_C2C;
	#endif

		uint_t size_g=/*pow(2,p->d)**/sizeof(gpuComplex)*p->n_total;
		uint_t size_f=1*sizeof(gpuComplex)*p->M_total;
		uint_t size_f_hat=1*sizeof(gpuComplex)*p->N_total;
		uint_t size_x=p->d*sizeof(dTyp)*p->M_total;
		uint_t neededMem = size_g+size_f+size_f_hat+size_x+sizeof(type)*size_g*2;
		printf("# CPU Mem min used : " PRINT_FORMAT " bytes (" PRINT_FORMAT " KB) (" PRINT_FORMAT " MB)\n", neededMem,inKB(neededMem),inMB(neededMem));
}


//------------------------------------------------------------------------------
// 		TIMER OUTPUT FUNCTION
//------------------------------------------------------------------------------

void showTimes(NFFTTimeSpec *times,int tfac)
{
#ifdef MEASURED_TIMES
		showTime_t("\n\tKERNEL RollOf",times->time_ROC/tfac);
		showTime_t("\tKERNEL FFT",times->time_FFT/tfac);
		showTime_t("\tKERNEL Conv",times->time_CONV/tfac);
		showTime_t("\tCOPY IN",times->time_COPY_IN/tfac);
		showTime_t("\tCOPY OUT",times->time_COPY_OUT/tfac);
#endif
		showTime_t("\n\tprocess took",times->runTime/tfac);
}

//------------------------------------------------------------------------------
// 		Memory OUTPUT FUNCTION
//------------------------------------------------------------------------------

void printStats(unsigned long free, unsigned long total)
{
   printf("\tFree : %lu bytes (%lu KB) (%lu MB)\n", free,inKB(free),inMB(free));
   printf("\tTotal: %lu bytes (%lu KB) (%lu MB)\n",total,inKB(total),inMB(total));
   printf("\t%f%% free, %f%% used\n", 100.0*free/(double)total,
		   100.0*(total - free)/(double)total);
}

void printStatsToFile(unsigned long free, unsigned long total,FILE *file,int device)
{
   fprintf(file,"Use Device: %d\t Free : %lu MB (%f%%)\t Total : %lu MB\n",
	device,inMB(free),100.0*free/(double)total,inMB(total));
}


//------------------------------------------------------------------------------
//			 OUTPUT FUNCTIONS
//------------------------------------------------------------------------------

void showCoeff_double(const dTyp *x, int n, const char* text)
{
	if(text != NULL){
		printf("\n %s\n",text);
	}

	int k;
	for (k = 0; k < n; k++){
		if (k%4 == 0) printf("%6d.\t", k);

		printf("%+.1lE,", x[k]);

		if (k%4 == 3) printf("\n");
	}

	if (n%4 != 0) printf("\n");

	printf("\n");
}

//------------------------------------------------------------------------------
//			 MATH HELPER FUNCTION
//------------------------------------------------------------------------------
/** Computes \f$\prod_{t=0}^{d-1} v_t\f$.*/
uint_t prod_int(uint_t *vec, int d)
{
	int t;

	uint_t prod=1;
	for(t=0; t<d; t++){
		prod *= vec[t];
	}

	return prod;
}

/** Computes \f$n\ge N\f$ such that \f$n=2^j,\, j\in\mathhb{N}_0\f$.*/
//TODO berechnung mit shift: siehe mathematisches Paper/link
uint_t next_power_of_2(uint_t N)
{
	uint_t n,i,logn;
	uint_t N_is_not_power_of_2=0;

	if (N == 0){
		return 1;
	}else{
		n=N;
		logn=0;
		while (n != 1){
			if (n%2 == 1){
				N_is_not_power_of_2=1;
			}
			n = n/2; //TODO n= n>>1
			logn++;
		}

		if (!N_is_not_power_of_2){
			logn--;
		}

		for (i = 0; i <= logn; i++){
			n = n*2; //TODO n= n<<1
		}

		return n;
	}
}

