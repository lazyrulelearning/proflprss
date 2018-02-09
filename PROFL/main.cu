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

using namespace std;

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#include <smmintrin.h>
#ifdef _MSC_VER
#include <intrin.h>

//#define __popcounter _mm_popcnt_u64
//#define __popcounter __popcnt64 
#define __popcounter __popcnt
#else

#  define __popcounter __builtin_popcount
//#  define __popcounter __builtin_popcountll

#endif


#define C_SIZE MAX_BITMAP_SIZE/4
#define BLK_SIZE  n_tests

#include "limits.cuh"

//struct where a test document is stored
typedef struct  list_node{

	int label;
	int size;
	int instance[MAX_FEATURES];

} listTest_t;

//struct for an itemset
typedef struct {
	int count;
	vector<int> list;
}itemset_t;

//struct for a rule antecedent support and consequent support for various classes
typedef struct MY_ALIGN(16){
	int 	ant_count;
	int 	consq_countRel[NUM_CLASSES];
} rule_t;

//stores the bitmap of a rule
typedef struct MY_ALIGN(16){
	rule_t rules;
	TYPE 	bitmap[MAX_BITMAP_SIZE];
} content_t;

//auxiliary hash table that stores  indexes on which index the bitmap hash table is located
typedef struct {
	rule_t rules[HASH_SIZE];
	char is_frequent[HASH_SIZE];
	unsigned long long key[HASH_SIZE];
	int size[HASH_SIZE];
	int htable_index[HASH_SIZE];
} fhash_t;

typedef struct {
	int ordered_labels[NUM_CLASSES];
	float points[NUM_CLASSES];
} score_t;

typedef struct {
	int correct;
	int label;
	int rules;
	score_t score;
} prediction_t;

typedef struct {
	int *total_predictions, *correct_predictions, *true_labels;
	float acc, mf1, *precision, *recall, *f1;
} evaluation_t;

struct DeviceVariables{

	int d_COUNT_TARGET[NUM_CLASSES];
	int *d_primes;
	int d_N_TRAINING;
	int d_N_ITEMSETS;
	int d_MIN_SUP;
	float d_MIN_CONF;
	int d_bitmap_size;
	int d_MAX_SIZE;
	int *d_indextable;

};


int primes[MAX_ITEMS];
itemset_t ITEMSETS[MAX_ITEMS];
content_t contClasses[NUM_CLASSES];
rule_t *rules1k;

unsigned int bitmap_size;

int count_target[NUM_CLASSES];
int N_ITEMSETS = 0, COUNT_TARGET[NUM_CLASSES],
TARGET_ID[NUM_CLASSES], MAX_SIZE = MAX_RULE_SIZE;
int N_TRAINING = 0, N_TESTS = 0;
int MIN_SUP = 1;
float MIN_CONF = 0.001f;
int gpu_cnt = 0;

map<string, int> ITEM_MAP;
map<string, int> CLASS_NAME;

listTest_t *TEST;

#include "utils.cu"
#include "comb.cu"
#include "prime_gen.cu"

#include "file_read.cu"
#include "scores.cu"
#include "profl.cu"

int lazy_supervised_ranking();

void intersect_itemsets(content_t *a, content_t *b, content_t* result);

int fillBitmaps(vector<int>& list, content_t* cont);

prediction_t* lazy_supervised_ranking_GPU();


int fillBitmaps(vector<int>& list, content_t* cont)
{

	for (unsigned int i = 0; i < bitmap_size; i++)
		(*cont).bitmap[i] = 0;

	//preenche o bitmap do content_t, usando a list de ocorrencia de ids de transações
	for (unsigned int i = 0; i < list.size(); i++){
		int indSlot = list[i] / BITMAP_SLOT_SIZE;
		int indBit = list[i] % BITMAP_SLOT_SIZE;

		(*cont).bitmap[indSlot] |= 1u << indBit;
	}

	return (int)list.size();
}

//create a new itemset with intersection of two others
void intersect_itemsets(content_t *a, content_t *b, content_t* result) {

	(*result).rules.ant_count = 0;
	
	if ((*a).rules.ant_count != 0 && (*b).rules.ant_count != 0) {

		for (unsigned int i = 0; i < bitmap_size; i++)
		{
			(*result).bitmap[i] = (*a).bitmap[i] & (*b).bitmap[i];
			(*result).rules.ant_count += __popcounter((*result).bitmap[i]);
		}
	}
}

//
prediction_t* lazy_supervised_ranking_GPU(TYPE *bitmaps){

	
	prediction_t *predictions = (prediction_t*)calloc(N_TESTS, sizeof(prediction_t));

	vector<double> gptimes(gpu_cnt);

	fprintf(stderr, "Tests number: %d\n", N_TESTS);

	omp_set_num_threads(gpu_cnt);
	//Launches a cpu thread for each GPU
#pragma omp parallel for
	for (int gpu_id = 0; gpu_id < gpu_cnt; gpu_id++)
	{
		gpuErrchk(cudaSetDevice(gpu_id));
		DeviceVariables dev_vars;

		//create stream
		cudaStream_t stream;
		gpuErrchk(cudaStreamCreate(&stream));

		//Distribute an even amount of tests for each GPU
		int TEST_CNT = ((int)ceil(N_TESTS / (double)gpu_cnt));
		int n_tests = (gpu_id + 1) * TEST_CNT > N_TESTS ? N_TESTS - gpu_id * TEST_CNT : TEST_CNT;
		int offset =  gpu_id * TEST_CNT;
		fprintf(stderr, "Tests number for gpu %d: %d\n", gpu_id, n_tests);

		listTest_t *d_TEST = 0;
		float *d_finalpoints = 0;
		int *d_nruls = 0;
		prediction_t *d_predictions = 0;
		rule_t *d_rules1k = 0;
		int *d_temps = 0;
		TYPE *d_bitmaps = 0;
		content_t* hash = 0;
		fhash_t *d_fhash = 0;

		int dimx;
		int num_bytes = 0, *d_indextable;

		dimx = (int)GetBinCoeff_l(TEST[0].size, MAX_SIZE - 1);
		num_bytes += dimx * sizeof(int) * (MAX_SIZE - 1);
		gpuErrchk(cudaMalloc((void**)&d_indextable, num_bytes));		

		if (MAX_SIZE > 2){
			gpuErrchk(cudaMalloc((void**)&d_fhash, sizeof(fhash_t)));
			gpuErrchk(cudaMalloc((void**)&hash, sizeof(content_t) * HASH_SIZE));
		}
		

		if (MAX_SIZE > 2){
			gpuErrchk(cudaMalloc((void**)&d_bitmaps, bitmap_size * N_ITEMSETS * sizeof(TYPE)));
		}
		gpuErrchk(cudaMalloc((void**)&d_rules1k, sizeof(content_t) * N_ITEMSETS));
		gpuErrchk(cudaMalloc((void**)&d_TEST, sizeof(listTest_t) * n_tests));
		gpuErrchk(cudaMalloc((void**)&d_finalpoints, NUM_CLASSES * n_tests * sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_nruls, NUM_CLASSES * n_tests * sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_predictions, n_tests * sizeof(prediction_t)));
		
		if (MAX_SIZE > 2){
			gpuErrchk(cudaMemsetAsync(hash, -1, HASH_SIZE * sizeof(content_t), stream));
			gpuErrchk(cudaMemsetAsync(d_fhash, 0, sizeof(fhash_t), stream));
			gpuErrchk(cudaMalloc((void**)&dev_vars.d_primes, N_ITEMSETS * sizeof(int)));
			gpuErrchk(cudaMemcpyAsync(dev_vars.d_primes, &primes[0], N_ITEMSETS * sizeof(int), cudaMemcpyHostToDevice));
		}

		gpuErrchk(cudaMemcpyAsync(d_TEST, TEST + offset, sizeof(listTest_t) * n_tests, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpyAsync(d_rules1k, rules1k, N_ITEMSETS * sizeof(rule_t), cudaMemcpyHostToDevice));

		if (MAX_SIZE > 2){
			gpuErrchk(cudaMemcpyToSymbolAsync(d_contClasses, contClasses, NUM_CLASSES * sizeof(content_t), 0, cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpyAsync(d_bitmaps, bitmaps, bitmap_size * N_ITEMSETS * sizeof(int), cudaMemcpyHostToDevice));
		}

		gpuErrchk(cudaMemsetAsync(d_finalpoints, 0, NUM_CLASSES * n_tests * sizeof(float)));
		gpuErrchk(cudaMemsetAsync(d_nruls, 0, NUM_CLASSES * n_tests * sizeof(float)));
		gpuErrchk(cudaMemsetAsync(d_predictions, 0, n_tests * sizeof(prediction_t)));

		memcpy(dev_vars.d_COUNT_TARGET, COUNT_TARGET, NUM_CLASSES * sizeof(int));
		dev_vars.d_N_TRAINING = N_TRAINING;
		dev_vars.d_N_ITEMSETS = N_ITEMSETS;
		dev_vars.d_MIN_SUP = MIN_SUP;
		dev_vars.d_MIN_CONF = MIN_CONF;
		dev_vars.d_bitmap_size = bitmap_size;
		dev_vars.d_MAX_SIZE = MAX_SIZE;
		dev_vars.d_indextable = d_indextable;		

		dim3 grid, block;
		block.x = num_threads;
		grid.x = BLK_SIZE; 

		gpuErrchk(cudaFuncSetCacheConfig(lazy_supervised_ranking_kernel, cudaFuncCachePreferShared));

		gptimes[gpu_id] = gettime();
		//Calls kernel for each k
		for (int k = 1; k < MAX_SIZE; k++){
						
			COMB_VARS vars;
			//initialize the combinadic on the CPU
			BinCoeff(TEST[0].size, k, &vars);
			gpuErrchk(cudaMemcpyToSymbol(indexes_d, vars.indexes, vars.IndexTabNum * sizeof(ARRAY), 0, cudaMemcpyHostToDevice));

			combkernel << <8,  1024>> >(TEST[0].size, k, d_indextable);
			gpuErrchk(cudaDeviceSynchronize());
			
			fprintf(stderr, "Size %d\n", k);

			lazy_supervised_ranking_kernel << <grid, block >> >(d_TEST, k, d_finalpoints, d_nruls,
				d_predictions, d_rules1k, d_temps, d_bitmaps, hash, d_fhash, n_tests,  dev_vars);

			gpuErrchk(cudaDeviceSynchronize());

			fprintf(stderr, "Done %d\n", k);

			free(vars.indexes);

		}
		gptimes[gpu_id] = gettime() - gptimes[gpu_id];		

		gpuErrchk(cudaStreamDestroy(stream));

		//prediction: tells whether or not the label was predicted correctly for the document, and keep its score
		gpuErrchk(cudaMemcpy(predictions + offset, d_predictions, n_tests * sizeof(prediction_t), cudaMemcpyDeviceToHost));
		
		if (MAX_SIZE > 2){
			gpuErrchk(cudaFree(dev_vars.d_primes));
		}

		gpuErrchk(cudaFree(d_TEST));
		gpuErrchk(cudaFree(d_finalpoints));
		gpuErrchk(cudaFree(d_nruls));
		if (MAX_SIZE > 2){
			gpuErrchk(cudaFree(d_bitmaps));
		}
		gpuErrchk(cudaFree(d_predictions));
		gpuErrchk(cudaFree(d_rules1k));


		if (MAX_SIZE > 2){
			gpuErrchk(cudaFree(hash));
			gpuErrchk(cudaFree(d_fhash));
		}		

	}

	sort(gptimes.begin(), gptimes.end());

	fprintf(stderr,"Kernel time: %lf\n", gptimes[gpu_cnt - 1]);	

	return predictions;
}

int lazy_supervised_ranking()
{
	evaluation_t evaluator;
	prediction_t *predictions;
	double s, tt = 0;

	s = gettime();
	initialize_evaluation(&evaluator);	
	TYPE *bitmaps;

	gpuErrchk(cudaHostAlloc(&rules1k, N_ITEMSETS * sizeof(rule_t), cudaHostAllocWriteCombined | cudaHostAllocPortable));
	gpuErrchk(cudaHostAlloc(&bitmaps, bitmap_size * N_ITEMSETS * sizeof(TYPE), cudaHostAllocWriteCombined | cudaHostAllocPortable));

	//builds the classes' bitmaps
	fprintf(stderr, "Preprocessing data.\n");
	for (int i = 0; i < NUM_CLASSES; i++)
	{
		contClasses[i].rules.ant_count = ITEMSETS[TARGET_ID[i]].count;
		
		int test_count = fillBitmaps(ITEMSETS[TARGET_ID[i]].list, &contClasses[i]);
				
		rules1k[i] = contClasses[i].rules;
		//reassign the bitmaps content so that it allows coalesced access on the GPU, while using the vectorize int4 type
		//Assigns the first 4 ints of each item bitmap together, then the seconds ones, and so forth
		for (unsigned int j = 0; j < bitmap_size / 4; j++){
			for (int k = 0; k < 4; k++)
				bitmaps[(j * 4) * N_ITEMSETS + k + i * 4] = contClasses[i].bitmap[j * 4 + k];
		}
	}
	//Calculate rules of size 2 on the CPU, since it's fast
	for (int i = NUM_CLASSES; i < N_ITEMSETS; i++){
		content_t contTemp, cont;
		cont.rules.ant_count = ITEMSETS[i].count;
		fillBitmaps(ITEMSETS[i].list, &cont);//fill the bitmap cont by using the item occurrence list

		if (NUM_CLASSES == 2){
			intersect_itemsets(&cont, &contClasses[1], &contTemp);
			cont.rules.consq_countRel[1] = contTemp.rules.ant_count;
		}
		else
			for (int j = 0; j < NUM_CLASSES; j++)
			{
				intersect_itemsets(&cont, &contClasses[j], &contTemp);
				cont.rules.consq_countRel[j] = contTemp.rules.ant_count;
			}

		rules1k[i] = cont.rules;
		for (unsigned int j = 0; j < bitmap_size / 4; j++){
			for (int k = 0; k < 4; k++)
				bitmaps[(j * 4) * N_ITEMSETS + k + i * 4] = cont.bitmap[j * 4 + k];
		}

	}

	predictions = lazy_supervised_ranking_GPU(bitmaps);
	gpuErrchk(cudaFreeHost(bitmaps));	

	stringstream stream;
	//iterting over each test result
	for (int i = 0; i < N_TESTS; i++)
	{
		update_evaluation(&evaluator, predictions[i].label, TEST[i].label);
		print_statistics(i + 1, TEST[i].label, i + 1, predictions[i], evaluator, stream);
	}

	stream << endl;

	for (int i = 0; i < NUM_CLASSES; i++)
		stream << "CLASS(" << i << ")= " << evaluator.true_labels[i] << " Prec= " << evaluator.precision[i] << " Rec= " << evaluator.recall[i] << " F1= " << evaluator.f1[i] << "  ";
	
	stream << "Acc= " << evaluator.acc << "  MF1= " << evaluator.mf1 << " \n";	

	std::cout << stream.str();
	free(predictions);

	finalize_evaluation(&evaluator);

	tt += gettime() - s;
	fprintf(stderr,"Process time: %lf\n", tt);	

	return 0;
}


int profl(char* training, char* test) {

	memset(&ITEMSETS, 0, MAX_ITEMS * sizeof(itemset_t));

	double t = gettime();

	read_training_set(training);

	t = gettime() - t;

	fprintf(stderr, "Input Train time: %lf\n", t);

	t = gettime();

	read_test_set(test);

	t = gettime() - t;

	fprintf(stderr, "Input Test time: %lf\n", t);

	//Correct bitmap size with the usage of int4 type
	bitmap_size = (bitMAP_SIZE % 4 != 0) ? 4 * (int)ceil(bitMAP_SIZE / (double)4) : bitMAP_SIZE;
	fprintf(stderr, "# itemsets %d \n", N_ITEMSETS);

	vector<int> prime_vec = generate_primes();
	vector<int>::iterator it = upper_bound(prime_vec.begin(), prime_vec.end(), N_ITEMSETS);

	if (it + N_ITEMSETS >= prime_vec.end()){
		fprintf(stderr, "Not enough prime numbers. Change prime_gen MAX variable.\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i <= N_ITEMSETS; it++, i++){
		primes[i] = *it;
	}

	lazy_supervised_ranking();
	gpuErrchk(cudaFreeHost(rules1k));
	gpuErrchk(cudaFreeHost(TEST));

	return 0;
}

void sig_handler(int signo)
{
	if (signo == SIGINT){
		fprintf(stderr, "received SIGINT\n");
		exit(EXIT_FAILURE);
		//gpuErrchk(cudaDeviceReset());
		//fprintf(stderr, "Cuda Device Reseted\n");

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
	gpuErrchk(cudaGetDeviceCount(&gpu_cnt));

	int c;
	char *train_file = NULL, *test_file = NULL;

	while ((c = getopt(argc, argv, "k:f:i:l:g:j:p:a:c:d:s:v:x:t:n:m:e:r:o:h")) != -1) {
		switch (c) {
		case 'i': train_file = strdup(optarg);
			break;
		case 't': test_file = strdup(optarg);
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

		}
	}
	fprintf(stderr, "Using %d devices\n", gpu_cnt);

	fprintf(stderr, "Lazy Supervised Ranking\n");
	profl(train_file, test_file);	

	gpuErrchk(cudaDeviceReset());
	fflush(stdout);

	return 0;
}
