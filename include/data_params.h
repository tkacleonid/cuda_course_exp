#pragma once

//Trace flag
#define TRACE true
//Data class 
#define CL 4

//Block sizes
#define BS1 32
#define BS2 32
#define BS3 1

//DATA classes
#if CL == 0
	#define P_SIZE        12
	#define NITER_DEFAULT 100
	#define DT_DEFAULT    0.015
#elif CL == 1
	#define P_SIZE        36
	#define NITER_DEFAULT 400
	#define DT_DEFAULT    0.0015
#elif CL == 2
	#define P_SIZE        64
	#define NITER_DEFAULT 400
	#define DT_DEFAULT    0.0015
#elif CL == 3
	#define P_SIZE        102
	#define NITER_DEFAULT 400
	#define DT_DEFAULT    0.001
#elif CL == 4
	#define P_SIZE        162
	#define NITER_DEFAULT 400
	#define DT_DEFAULT    0.00067
#elif CL == 5
	#define P_SIZE        408
	#define NITER_DEFAULT 500
	#define DT_DEFAULT    0.0003
#elif CL == 6
	#define P_SIZE        1020
	#define NITER_DEFAULT 500
	#define DT_DEFAULT    0.0001
#endif	
