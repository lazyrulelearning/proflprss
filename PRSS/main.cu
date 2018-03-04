#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <map>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>
#include <string>
#include <omp.h>
#include <signal.h>

#include <smmintrin.h>
#ifdef _MSC_VER
#  include <intrin.h>

//#  define __popcounter _mm_popcnt_u64
#  define __popcounter __popcnt64 
//#  define __popcounter __popcnt
#else

//#  define __popcounter __builtin_popcount
#  define __popcounter __builtin_popcountll

#endif

#define __gpu_pop __popcll


#include "limits.cuh"
#include "utils.cu"


using namespace std;

//struct for an itemset
typedef struct {
	int count;
	vector<int> list;
}itemset_t;

//struct for a rule antecedent support and consequent support for various classes
typedef struct {
	int 	ant_count;
	int	consq_countRel[NUM_CLASSES];
} rule_t;

//stores the bitmap of a rule
typedef struct {
	rule_t rules;
	TYPE 	bitmap[MAX_BITMAP_SIZE];
} content_t;

struct Dev_doc{
	short features[MAX_FEATURES];
};

struct Device_Variables{
	short *d_docs;
	int *d_rulecount;
	TYPE *d_bitmaps;
};


itemset_t ITEMSETS[MAX_ITEMS];

unsigned int bitmap_size;

int count_target[NUM_CLASSES], option = 0;
int N_ITEMSETS = 0, COUNT_TARGET[NUM_CLASSES],
TARGET_ID[NUM_CLASSES], MAX_SIZE = MAX_RULE_SIZE;
int N_TRAINING = 0, N_TESTS = 0;
int MIN_SUP = 1;
float MIN_CONF = 0.001f;
int gpu_cnt = 0;

map<string, int> ITEM_MAP;
map<int, string> SYMBOL_TABLE;
map<string, int> CLASS_NAME;


vector< vector<short> > UNLABELED; //Stores the documents from input
vector<int> qids; //query/document id

int *h_rulecount = 0;
short *h_projected = 0;


#include "file_read.cu"
#include "common.cu"
#include "pssarp.cu"
#include "prss.cu"

//Starts the dataset reduction process SSARP by using multithreading or GPU
void SSARP(char *training, int partitions, char * file_features){

	double s;
	double tt = 0;

	memset(&ITEMSETS, 0, MAX_ITEMS * sizeof(itemset_t));
	s = gettime();

	read_unlabeled_set(training);

	fprintf(stderr, "\nInput time: %lf\n", gettime() - s);

	s = gettime();
	if (option == 1) selective_sampling(partitions, file_features);
	if (option == 2) selective_samplingCUDA(partitions, file_features);

	tt = gettime() - s;

	fprintf(stderr, "Process Time: %lf\n", tt);

}


void sig_handler(int signo)
{
	if (signo == SIGINT){
		fprintf(stderr, "received SIGINT\n");
		exit(EXIT_FAILURE);
		gpuErrchk(cudaDeviceReset());
		fprintf(stderr, "Cuda Device Reseted\n");

	}
}


int main(int argc, char** argv) {

	if (signal(SIGINT, sig_handler) == SIG_ERR)
	{
		fputs("An error occurred while setting a signal handler.\n", stderr);
		return EXIT_FAILURE;
	}
	//freopen("output.txt", "w", stdout);

	gpu_cnt = 1;
	cudaGetDeviceCount(&gpu_cnt);

	int c, partitions = 1;
	char *file_in = NULL,  *file_features = NULL;

	while ((c = getopt(argc, argv, "o:k:f:i:l:g:j:p:a:c:d:s:v:x:t:n:m:e:r:h")) != -1) {
		switch (c) {
		case 'i': file_in = strdup(optarg);
			break;		
		case 'f': file_features = strdup(optarg);
			break;
		case 'c': MIN_CONF = (float)atof(optarg);
			break;
		case 's': MIN_SUP = atoi(optarg);
			break;
		case 'g': gpu_cnt = atoi(optarg);
			if (gpu_cnt < 1 || gpu_cnt > 4)
				gpu_cnt = 1;
			break;
		case 'm': MAX_SIZE = atoi(optarg);
			if (MAX_SIZE > MAX_RULE_SIZE)
				MAX_SIZE = MAX_RULE_SIZE;
			break;
		case 'o': option = atoi(optarg); // 1 SSAR/SSARP, 2 PRSS
			if (option > 2 || option < 1)
				option = 1;
			break;
		case 'p': partitions = atoi(optarg); // partitions for SSARP / 1 equals SSAR
			if (partitions < 0)
				partitions = 1;
			break;

		}
	}

	if (option == 1){
		fprintf(stderr, "Using %d cores\n", omp_get_max_threads());		
	}
	
	if (option == 2){
		fprintf(stderr, "Using %d devices\n", gpu_cnt);
		omp_set_num_threads(gpu_cnt);
	}

	fprintf(stderr, "Lazy Active mode: ");
	if (partitions == 1)
		fprintf(stderr, "SSAR\n");
	else
		fprintf(stderr, "SSARP\n");
	SSARP(file_in, partitions, file_features);	

	gpuErrchk(cudaDeviceReset());
	fflush(stdout);

	return 0;
}
