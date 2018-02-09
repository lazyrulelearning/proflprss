

///'Combinadic' / mth - lexicographic combination adapted from
// http://tablizingthebinomialcoeff.wordpress.com/2011/08/09/how-to-create-and-access-a-table-based-upon-the-binomial-coefficient/
//, designed and originally written by Robert G. Bryan, specifically the functions GetIndexes, BinCoeff, GetBinCoeff_l, GetKIndexes_d



typedef struct _array{
	int v[MAX_FEATURES];
	int capacity;
}ARRAY;

typedef struct {

	int  NumItems;                  // Total number of items.  Equal to N.
	int  GroupSize;                 // # of items in a group.  Equal to K.
	int  IndexTabNum;               // Total number of index tables.  Equal to K - 1.
	int  IndexTabNumM1;             // Total number of index tables minus 1.  Equal to K - 2.
	int  IndexTabNumM2;             // Total number of index tables minus 2.  Equal to K - 3.
	int  TotalCombos;               // Total number of unique combinations.
	ARRAY *indexes;

}COMB_VARS;

void GetIndexes(COMB_VARS *vars);
void BinCoeff(int N, int K, COMB_VARS *vars);

__host__ __device__ int  GetBinCoeff_l(int   N, int   K);
__device__ bool d_next_combination(short* n_begin, short* n_end, short* r_begin, short* r_end);
__device__ void GetKIndexes_d(int  Index, short * KIndexes, int  K, int N);

__device__ ARRAY indexes_d[6];

__device__  bool d_next_combination(short* n_begin, short* n_end, short* r_begin, short* r_end) {
	bool boolmarked = false;
	short* r_marked = 0;
	short* n_it1 = n_end;
	--n_it1;
	short* tmp_r_end = r_end;
	--tmp_r_end;
	short* tmp_r_begin = r_begin;
	--tmp_r_begin;

	for (short* r_it1 = tmp_r_end; r_it1 != tmp_r_begin; --r_it1, --n_it1) {
		if (*r_it1 == *n_it1) {
			if (r_it1 != r_begin) {
				boolmarked = true;
				r_marked = (--r_it1);
				++r_it1;
				continue;
			}
			else return false;
		}
		else {
			if (boolmarked == true) {
				short* n_marked = 0;
				for (short* n_it2 = n_begin; n_it2 != n_end; ++n_it2) {
					if (*r_marked == *n_it2) {
						n_marked = n_it2;
						break;
					}
				}
				short* n_it3 = ++n_marked;
				for (short* r_it2 = r_marked; r_it2 != r_end; ++r_it2, ++n_it3) {
					*r_it2 = *n_it3;
				}
				return true;
			}
			for (short* n_it4 = n_begin; n_it4 != n_end; ++n_it4) {
				if (*r_it1 == *n_it4) {
					*r_it1 = *(++n_it4);
					return true;
				}
			}
		}
	}
	return true;
}


void BinCoeff(int  N, int  K, COMB_VARS *vars)
{
	// This constructor builds the index tables used to retrieve the index to the binomial coefficient table.
	// N is the number of items and K is the number of items in a group.
	
	int   N1, K1, TotalCombosL;
	// Validate the inputs.
	if (K < 1)
	{
		printf("K < 1\n");
		exit(-1);
	}
	if (N <= K)
	{
		printf("N <= K\n");
		exit(-2);
	}
	// Get the total number of unique combinations.
	vars->IndexTabNum = K - 1;
	vars->IndexTabNumM1 = vars->IndexTabNum - 1;
	vars->IndexTabNumM2 = vars->IndexTabNumM1 - 1;
	vars->NumItems = N;
	vars->GroupSize = K;
	vars->IndexTabNum = vars->GroupSize - 1;
	N1 = (long long)N;
	K1 = (long long)K;
	TotalCombosL = GetBinCoeff_l(N1, K1);
	if (TotalCombosL > (1u << 31) - 1)
	{
		printf("BinCoeff:BinCoeff - Total # of combos > 2GB.\n");
		//exit(-3);
	}
	vars->TotalCombos = (int)TotalCombosL;
	GetIndexes(vars);
}

void GetIndexes(COMB_VARS *vars)
{
	// This function creates each index that is used to obtain the index to the binomial coefficient
	// table based upon the underlying K indexes.
	//
	int  LoopIndex, Loop, Value, IncValue, StartIndex, EndIndex;

	vars->indexes = (ARRAY*)calloc(vars->IndexTabNum, sizeof(ARRAY));

	// Create the arrays used for each index.
	for (Loop = 0; Loop < vars->IndexTabNum; Loop++)
	{
		vars->indexes[Loop].capacity = vars->NumItems - Loop;
	}
	// Get the indexes values for the least significant index.

	ARRAY *index_array_least = &vars->indexes[vars->IndexTabNumM1];
	Value = 1;
	IncValue = 2;

	for (Loop = 2; Loop < (int)index_array_least->capacity/*IndexArrayLeast.capacity()*/; Loop++)
	{
		index_array_least->v[Loop] = Value;

		Value += IncValue++;
	}
	// Get the index values for the remaining indexes.
	Value = 1;
	IncValue = 2;
	StartIndex = 3;
	EndIndex = vars->NumItems - vars->IndexTabNumM2;
	for (LoopIndex = vars->IndexTabNumM2; LoopIndex >= 0; LoopIndex--)
	{

		ARRAY* index_array_prev = &vars->indexes[(LoopIndex + 1)];
		ARRAY* index_array = &vars->indexes[LoopIndex];

		index_array->v[StartIndex] = 1;

		for (Loop = StartIndex + 1; Loop < EndIndex; Loop++)
		{

			index_array->v[Loop] = index_array->v[Loop - 1] + index_array_prev->v[Loop - 1];
		}

		StartIndex++;
		EndIndex++;
	}
}

__host__ __device__ int   GetBinCoeff_l(int   N, int   K)
{
	// This function gets the total number of unique combinations based upon N and K.
	// N is the total number of items.
	// K is the size of the group.
	// Total number of unique combinations = N! / ( K! (N - K)! ).
	// This function is less efficient, but is more likely to not overflow when N and K are large.
	// Taken from:  http://blog.plover.com/math/choose.html
	//
	int   r = 1;
	int   d;
	if (K > N) return 0;
	for (d = 1; d <= K; d++)
	{
		r *= N--;
		r /= d;
	}
	return r;
}

//The function that returns the Nth combination of size K
__device__ void GetKIndexes_d(int  Index, short * KIndexes, int  K, int N)
{
	// This function returns the proper K indexes from an index to the sorted binomial coefficient table.
	// This is the reverse of the GetIndex function.  The correct K indexes are returned in descending order
	// in KIndexes.
	int LoopIndex, Loop, End, RemValue = Index;

	ARRAY* index_array;
	for (LoopIndex = 0; LoopIndex < K - 1; LoopIndex++)
	{
		index_array = &indexes_d[LoopIndex];

		End = index_array->capacity - 1;
		for (Loop = End; Loop >= 0; Loop--)
		{
			if (RemValue >= index_array->v[Loop])
			{
				KIndexes[LoopIndex] = N - Loop;
				RemValue -= index_array->v[Loop];
				break;
			}
		}
	}
	KIndexes[K - 1] = N - RemValue;
}
///

//Create an index table with the lexicographic combinations, so that every thread can reuse it,
//and also not having the need for each thread to call GetKIndexes_d during the main kernel.
//It's also filled in a way that allows coalesced access
__global__ void combkernel(int n, int k, int *idxtable)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ short ele[MAX_FEATURES];

	short v[10];

	if (threadIdx.x < n && blockDim.x >= n)
	{
		ele[threadIdx.x] = threadIdx.x;
	}
	else{
		for (int i = 0; i < n; i++)
			ele[i] = i;
	}
	__syncthreads();

	int max = (int)GetBinCoeff_l(n, k);
	int range = (int)ceil(max / (double)blockDim.x * gridDim.x);
	int ran = tid * range;

	range = (ran + range > max) ? max : (ran + range);

	if (ran < max){

		GetKIndexes_d(max - ran - 1, v, k, n - 1);

		for (int i = 0; i < k; i++)
			idxtable[i * max + ran] = v[i];

		ran++;

		while (ran < range){

			d_next_combination(ele, ele + n, v, v + k);

			for (int i = 0; i < k; i++)
				idxtable[i * max + ran] = v[i];

			ran++;
		}

	}
}