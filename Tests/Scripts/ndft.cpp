/*
 * ndft.c
 *
 *  Created on: 05.05.2015
 *      Author: sukunis
 */

#include "ndft.h"


double ndft(cunfft_plan *myplan, gpuComplex *f)
{
	CPUTimer t0;
	int dim=myplan->d;
	int tM=myplan->M_total;
	int tN[dim];
	int tn[dim];
	for(int i=0; i<dim; i++){
		tN[i]=myplan->N[i];
		tn[i]=myplan->n[i];
	}

	NFFT(plan) p;
	/** init an one dimensional plan */
#ifdef COM_FG_PSI
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			FG_PSI | MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#else
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			/*PRE_PHI_HUT| PRE_FULL_PSI|*/ MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#endif
	// set x nodes
	for(int i=0; i<p.d*p.M_total; i++){
		p.x[i]=myplan->x[i];
	}
	// set fhat
	for(int j=0; j<p.N_total;j++){
		__real__(p.f_hat[j])=myplan->f_hat[j].x;
		__imag__(p.f_hat[j])=myplan->f_hat[j].y;
	}

	t0 = getTimeCPU();
	 NFFT(trafo_direct)(&p);
	double t=elapsedCPUTime(t0,getTimeCPU());

	for(int j=0; j<p.M_total;j++){
		f[j].x=creal(p.f[j]);
		f[j].y=cimag(p.f[j]);
	}
	/** finalise the one dimensional plan */
	NFFT(finalize)(&p);

	return t;
}

double ndft_adjoint(cunfft_plan *myplan, gpuComplex* f_hat)
{
	int tM=myplan->M_total;
	int dim=myplan->d;
	int tN[dim];
	int tn[dim];
	for(int i=0; i<dim; i++){
		tN[i]=myplan->N[i];
		tn[i]=myplan->n[i];
	}
	CPUTimer t0, t1;

	NFFT(plan) p;
	/** init an one dimensional plan */
#ifdef COM_FG_PSI
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			FG_PSI | MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#else
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			/*PRE_PHI_HUT| PRE_FULL_PSI|*/ MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#endif
	// set x nodes
	for(int i=0; i<p.d*p.M_total; i++){
		p.x[i]=myplan->x[i];
	}
	// set fhat
	for(int j=0; j<p.M_total;j++){
		__real__(p.f[j])=myplan->f[j].x;
		__imag__(p.f[j])=myplan->f[j].y;
	}


	/** approx. trafo and show the result */
	t0 = getTimeCPU();
	 NFFT(adjoint_direct)(&p);
	t1 = getTimeCPU();
	//	NFFT(vpr_complex)(p.f_hat,32,"nfft adjoint, vector f_hat");
	double t=elapsedCPUTime(t0,t1);
	//	printf(" took %.2f seconds.\n",t);

	// copy result
	for(int j=0; j<p.M_total;j++){
		f_hat[j].x=creal(p.f_hat[j]);
		f_hat[j].y=cimag(p.f_hat[j]);
	}
	/** finalise the one dimensional plan */
	NFFT(finalize)(&p);

	return t;
}

double nfft(cunfft_plan *myplan, gpuComplex *f)
{
	int dim=myplan->d;
	int tM=myplan->M_total;
	int tN[dim];
	int tn[dim];
	for(int i=0; i<dim; i++){
		tN[i]=myplan->N[i];
		tn[i]=myplan->n[i];
	}
	CPUTimer t0;

	NFFT(plan) p;
	/** init an one dimensional plan */
#ifdef COM_FG_PSI
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			FG_PSI | MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#else
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			/*PRE_PHI_HUT| PRE_FULL_PSI|*/ MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#endif
	// set x nodes
	for(int i=0; i<p.d*p.M_total; i++){
		p.x[i]=myplan->x[i];
	}
	// set fhat
	for(int j=0; j<p.N_total;j++){
		__real__(p.f_hat[j])=myplan->f_hat[j].x;
		__imag__(p.f_hat[j])=myplan->f_hat[j].y;
	}

	/** precompute psi, the entries of the matrix B */
	if (p.flags & PRE_ONE_PSI)
		NFFT(precompute_one_psi)(&p);

	/** approx. trafo and show the result */
	t0 = getTimeCPU();
	NFFT(trafo)(&p);

	double t=elapsedCPUTime(t0,getTimeCPU());

	//	NFFT(vpr_complex)(p.f,32,"nfft, vector f");
	// copy result
	for(int j=0; j<p.M_total;j++){
		f[j].x=creal(p.f[j]);
		f[j].y=cimag(p.f[j]);
	}
	/** finalise the one dimensional plan */
	NFFT(finalize)(&p);

	return t;
}


double nfft_adjoint(cunfft_plan *myplan, gpuComplex* f_hat)
{
	int dim=myplan->d;
	int tM=myplan->M_total;
	int tN[dim];
	int tn[dim];
	for(int i=0; i<dim; i++){
		tN[i]=myplan->N[i];
		tn[i]=myplan->n[i];
	}
	CPUTimer t0, t1;

	NFFT(plan) p;
	/** init an one dimensional plan */
#ifdef COM_FG_PSI
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			FG_PSI | MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#else
	NFFT(init_guru)(&p, dim, tN, tM, tn, CUT_OFF,
			/*PRE_PHI_HUT| PRE_FULL_PSI|*/ MALLOC_F_HAT| MALLOC_X| MALLOC_F |
			FFTW_INIT| FFT_OUT_OF_PLACE,
			FFTW_ESTIMATE| FFTW_DESTROY_INPUT);
#endif
	// set x nodes
	for(int i=0; i<p.d*p.M_total; i++){
		p.x[i]=myplan->x[i];
	}
	// set fhat
	for(int j=0; j<p.M_total;j++){
		__real__(p.f[j])=myplan->f[j].x;
		__imag__(p.f[j])=myplan->f[j].y;
	}

	/** precompute psi, the entries of the matrix B */
	if (p.flags & PRE_ONE_PSI)
		NFFT(precompute_one_psi)(&p);

	/** approx. trafo and show the result */
	t0 = getTimeCPU();
	NFFT(adjoint)(&p);
	t1 = getTimeCPU();
	//	NFFT(vpr_complex)(p.f_hat,32,"nfft adjoint, vector f_hat");
	double t=elapsedCPUTime(t0,t1);
	//	printf(" took %.2f seconds.\n",t);

	// copy result
	for(int j=0; j<p.M_total;j++){
		f_hat[j].x=creal(p.f_hat[j]);
		f_hat[j].y=cimag(p.f_hat[j]);
	}
	/** finalise the one dimensional plan */
	NFFT(finalize)(&p);

	return t;
}

