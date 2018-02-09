#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


#ifdef _WIN32

#include <sys/timeb.h>
#include <time.h>
#include <float.h>
#include <windows.h>

double gettime() { // granularity about 50 microsecs on my machine
	static LARGE_INTEGER freq, start;
	LARGE_INTEGER count;
	if (!QueryPerformanceCounter(&count))
		// FatalError("QueryPerformanceCounter");
		fprintf(stderr, "QueryPerformanceCounter");
	if (!freq.QuadPart) { // one time initialization
		if (!QueryPerformanceFrequency(&freq))
			//FatalError("QueryPerformanceFrequency");
			fprintf(stderr, "QueryPerformanceCounter");
		start = count;
	}
	return (double)(count.QuadPart - start.QuadPart) / freq.QuadPart;
}


int     opterr = 1,             /* if error message should be printed */
optind = 1,             /* index into parent argv vector */
optopt,                 /* character checked for validity */
optreset;               /* reset getopt */
char    *optarg;                /* argument associated with option */

#define BADCH   (int)'?'
#define BADARG  (int)':'
#define EMSG    ""

/*
* getopt --
*      Parse argc/argv argument vector.
*/
int
getopt(int nargc, char * const nargv[], const char *ostr)
{
	static char *place = EMSG;              /* option letter processing */
	const char *oli;                        /* option letter list index */

	if (optreset || !*place) {              /* update scanning pointer */
		optreset = 0;
		if (optind >= nargc || *(place = nargv[optind]) != '-') {
			place = EMSG;
			return (-1);
		}
		if (place[1] && *++place == '-') {      /* found "--" */
			++optind;
			place = EMSG;
			return (-1);
		}
	}                                       /* option letter okay? */
	if ((optopt = (int)*place++) == (int)':' ||
		!(oli = strchr(ostr, optopt))) {
		/*
		* if the user didn't specify '-' as an option,
		* assume it means -1.
		*/
		if (optopt == (int)'-')
			return (-1);
		if (!*place)
			++optind;
		if (opterr && *ostr != ':')
			(void)printf("illegal option -- %c\n", optopt);
		return (BADCH);
	}
	if (*++oli != ':') {                    /* don't need argument */
		optarg = NULL;
		if (!*place)
			++optind;
	}
	else {                                  /* need an argument */
		if (*place)                     /* no white space */
			optarg = place;
		else if (nargc <= ++optind) {   /* no arg */
			place = EMSG;
			if (*ostr == ':')
				return (BADARG);
			if (opterr)
				(void)printf("option requires an argument -- %c\n", optopt);
			return (BADCH);
		}
		else                            /* white space */
			optarg = nargv[optind];
		place = EMSG;
		++optind;
	}
	return (optopt);                        /* dump back option letter */
}
#else

#include <sys/time.h>
#include <getopt.h>
#define _isnan(a) isnan(a)

double gettime() { // returns 0 seconds first time called
	static struct timeval t0;
	struct timeval tv;
	gettimeofday(&tv, 0);
	if (!t0.tv_sec)
		t0 = tv;
	return tv.tv_sec - t0.tv_sec + (tv.tv_usec - t0.tv_usec) / 1000000.;
}

/*
double gettime() {
timespec ts;
clock_gettime(CLOCK_REALTIME, &ts);
return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}
/**/

#endif



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, 
int line, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(1);
	}
}
