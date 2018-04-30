#ifndef CONFIG_H
#define CONFIG_H

/**\def DEBUG 
* Enable verification of CUDA function calls (cmakedefine)*/
#define DEBUG

/**\def PRINT_CONFIG 
* Launch kernel configurations to stdout (cmakedefine)*/
#define PRINT_CONFIG 

/**\def MEASURED_TIMES 
* Measure time for several CUDA kernels (cmakedefine) */
/* #undef MEASURED_TIMES */

/**\def MILLI_SEC 
* Measure time in milli seconds (else seconds) (cmakedefine)*/
#define MILLI_SEC 

/**\def COM_FG_PSI 
* Enable fast gaussian gridding (cmakedefine)*/
#define COM_FG_PSI 

/**\def CUNFFT_DOUBLE_PRECISION 
* Enable double precision computation (cmakedefine)*/
#define CUNFFT_DOUBLE_PRECISION 

/**\def LARGE_INPUT 
* Use LONG INT instead of INT (cmakedefine)*/
#define LARGE_INPUT

/**\def CUT_OFF 
* Cut-off parameter for window function.  (cmakedefine)*/
#define CUT_OFF 6

/**\def THREAD_DIM_X 
* Maximum numbers of threads (power of two) per block in x direction for 
* 2D launch grid. Default 16. For configuration see also your device properties. (cmakedefine)*/
#define THREAD_DIM_X 16

/**\def THREAD_DIM_Y 
* Maximum numbers of threads (power of two) per block in y direction for 
* 2D launch grid. Default 16. For configuration see also your device properties. (cmakedefine)*/
#define THREAD_DIM_Y 16

/**\def THREAD_DIM 
* Maximum numbers of threads (power of two) per block in x direction for 
* 1D launch grid. Default 512. For configuration see also your device properties. (cmakedefine)*/
#define THREAD_DIM 512

/**\def MAX_NUM_THREADS 
* Maximum numbers of threads (power of two) per block. 
* Default 1024. For configuration see also your device properties. (cmakedefine)*/
#define MAX_NUM_THREADS 1024

/**\def MAX_BLOCK_DIM_X 
* Maximum numbers of threads (power of two) per block in x direction. 
* Default 1024. For configuration see also your device properties. (cmakedefine)*/
#define MAX_BLOCK_DIM_X 1024

/**\def MAX_BLOCK_DIM_Y 
* Maximum numbers of threads (power of two) per block in y direction. 
* Default 1024. For configuration see also your device properties. (cmakedefine)*/
#define MAX_BLOCK_DIM_Y 1024

/**\def MAX_BLOCK_DIM_Z 
* Maximum numbers of threads (power of two) per block in z direction. 
* Default 64. For configuration see also your device properties. (cmakedefine)*/
#define MAX_BLOCK_DIM_Z 64

/**\def MAX_GRID_DIM_X 
* Maximum numbers of threads (power of two) per grid in x direction. 
* Default 65535. For configuration see also your device properties. (cmakedefine)*/
#define MAX_GRID_DIM_X 65535

/**\def MAX_GRID_DIM_Y 
* Maximum numbers of threads (power of two) per grid in y direction. 
* Default 65535. For configuration see also your device properties. (cmakedefine)*/
#define MAX_GRID_DIM_Y 65535

/**\def MAX_GRID_DIM_Z 
* Maximum numbers of threads (power of two) per grid in z direction. 
* Default 1. For configuration see also your device properties. (cmakedefine)*/
#define MAX_GRID_DIM_Z 1

#endif //CONFIG_H
