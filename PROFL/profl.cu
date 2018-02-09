__device__ rule_t get_from_hash_table(int* itemset, int size, content_t *d_mhash, fhash_t *d_fhash
	, rule_t *d_rules1k, TYPE* d_bitmaps, uint4 classes[NUM_CLASSES][C_SIZE], DeviceVariables *vars);

__device__ void calculate_rule(int* items, int level, float *points, int *nrules,
	content_t *d_mhash, fhash_t *d_fhash, DeviceVariables *vars);

__device__ void generate_rules(int items[MAX_FEATURES], int elements[MAX_FEATURES], int n_items, int k,
	int* nrules, float* points, content_t *d_mhash, fhash_t *d_fhash, DeviceVariables *vars);

__global__ void lazy_supervised_ranking_kernel(listTest_t *tests, int k,
	float* finalpoints, int* gn_rules, prediction_t *d_predictions, rule_t *d_rules1k, 
	int *d_temps, TYPE *d_bitmaps, content_t *d_mhash, fhash_t *d_fhash, int n_tests, DeviceVariables vars);

//Global variable of the classes' bitmaps, so that it can be stored at shared memory later
__device__ content_t d_contClasses[NUM_CLASSES];


//Max number of iterations of the linear probing
#define iter (7)



__device__ rule_t get_from_hash_table(int* itemset, int size, content_t *d_mhash, fhash_t *d_fhash
	, rule_t *d_rules1k, TYPE* d_bitmaps, uint4 classes[NUM_CLASSES][C_SIZE], DeviceVariables *vars)
{

	content_t cont;
	int j = 0, cp = 0;
	unsigned long long  key = 1, mt = 0, mt2 = 0;
	double FP = 0;
	uint4 u1[2];

	int d_MIN_SUP = vars->d_MIN_SUP, d_N_ITEMSETS = vars->d_N_ITEMSETS, d_bitmap_size = vars->d_bitmap_size / 4;

	//Calculates the fingerprint of the itemset
	for (int i = 0; i < size; i++) {

		FP += itemset[i] / (double)vars->d_primes[itemset[i]];
	}

	double FP_first = FP;

	cont.rules.ant_count = 0;

	//For the itemset of size 3
	if (size == 3)
	{
		key = *(unsigned long long*)&FP_first;
		mt = key % HASH_SIZE;
		mt2 = mt;
		cp = 1;

		//Checks if it is in the hash table
		while (d_fhash->key[mt] != 0 && d_fhash->key[mt] != key && cp < iter)
		{
			mt = ((mt + 1)) % HASH_SIZE;
			cp++;
		}
		//return if found
		if (key == d_fhash->key[mt]){

			return d_fhash->rules[mt];
		}

		FP_first = 0;
		//If not found, calculate the fingerprint of the prefix of size 2
		for (int i = 0; i < size - 1; i++)
		{
			FP_first += itemset[i] / (double)vars->d_primes[itemset[i]];
		}
		j = 0;

	}

	key = *(unsigned long long*)&FP_first;
	mt = key % HASH_SIZE;

	if (size == 2)
		mt2 = mt;

	//Searches for a itemset of size 2
	cp = 1;
	while (d_fhash->key[mt] != 0 && d_fhash->key[mt] != key  && cp < iter)
	{
		mt = ((mt + 1)) % HASH_SIZE;
		cp++;
	}

	if (key == d_fhash->key[mt])
	{
		//If the original itemset being searched was of size 2, return it
		if (size == 2)
		{
			return d_fhash->rules[mt];
		}
		//Else, the prefix was found, so keep its bitmap
		cont = d_mhash[d_fhash->htable_index[mt]];
		j = 1;
	}
	else
	{
		if (size > 2){
			cont.rules.ant_count = -1;
			//return cont.rules;
		}

	}
	if (d_rules1k[itemset[0]].ant_count >= d_MIN_SUP)
	{
		if (size == 2){

			//vectorized intersection of bitmaps with the bitwise AND operation
			for (int i = 0; i < d_bitmap_size; i++){

				u1[0] = ((uint4*)d_bitmaps)[i * d_N_ITEMSETS + itemset[0]];
				u1[1] = ((uint4*)d_bitmaps)[i * d_N_ITEMSETS + itemset[1]];

				cont.rules.ant_count += __popc(u1[0].w &= u1[1].w)
					+ __popc(u1[0].x &= u1[1].x)
					+ __popc(u1[0].y &= u1[1].y)
					+ __popc(u1[0].z &= u1[1].z);

				((uint4*)cont.bitmap)[i] = u1[0];
			}
		}

		else{
			//j is the flag if the prefix was used or not for the itemset of size 3
			if (j == 0){
				cont.rules.ant_count = 0;
				for (int i = 0; i < d_bitmap_size; i++){

					u1[0] = ((uint4*)d_bitmaps)[i * d_N_ITEMSETS + itemset[0]];
					u1[1] = ((uint4*)d_bitmaps)[i * d_N_ITEMSETS + itemset[1]];

					cont.rules.ant_count += __popc(u1[0].w &= u1[1].w)
						+ __popc(u1[0].x &= u1[1].x)
						+ __popc(u1[0].y &= u1[1].y)
						+ __popc(u1[0].z &= u1[1].z);

					((uint4*)cont.bitmap)[i] = u1[0];
				}
				j++;

			}
			//If the prefix is frequent, then complete the intersection with the last item
			if (j == 1 && cont.rules.ant_count >= d_MIN_SUP){
				cont.rules.ant_count = 0;
				for (int i = 0; i < d_bitmap_size; i++){

					u1[0] = ((uint4*)cont.bitmap)[i];
					u1[1] = ((uint4*)d_bitmaps)[i * d_N_ITEMSETS + itemset[2]];

					cont.rules.ant_count += __popc(u1[0].w &= u1[1].w)
						+ __popc(u1[0].x &= u1[1].x)
						+ __popc(u1[0].y &= u1[1].y)
						+ __popc(u1[0].z &= u1[1].z);

					((uint4*)cont.bitmap)[i] = u1[0];
				}

			}

		}
	}


	if (cont.rules.ant_count >= d_MIN_SUP){

		//With 2 classes, the intersection of only one is needed, since the confidence value is the complement of the other class
		if (NUM_CLASSES == 2){
			cont.rules.consq_countRel[1] = 0;
			for (int i = 0; i < d_bitmap_size; i++){

				u1[0] = ((uint4*)cont.bitmap)[i];
				u1[1] = classes[1][i];

				cont.rules.consq_countRel[1] += __popc(u1[0].w & u1[1].w)
					+ __popc(u1[0].x & u1[1].x)
					+ __popc(u1[0].y & u1[1].y)
					+ __popc(u1[0].z & u1[1].z);

			}
		}
		else

			//Intersect with the class items, the consequents, to build the rules
#pragma unroll
			for (int j = 0; j < NUM_CLASSES; j++)
			{
				cont.rules.consq_countRel[j] = 0;
				for (int i = 0; i < d_bitmap_size; i++){

					u1[0] = ((uint4*)cont.bitmap)[i];
					u1[1] = classes[j][i];

					cont.rules.consq_countRel[j] += __popc(u1[0].w & u1[1].w)
						+ __popc(u1[0].x & u1[1].x)
						+ __popc(u1[0].y & u1[1].y)
						+ __popc(u1[0].z & u1[1].z);
				}
			}
	}

	//Treats the fingerprint as a 64 bit integer to use in the hash calc
	key = *(unsigned long long*)&FP;
	mt = mt2;
	//Iteratively tries to find an available position, because of parallel concurrency
	while (1){
		cp = 1;

		while (d_fhash->key[mt2] != 0 && cp < iter)
		{
			mt2 = (mt2 + 1) % HASH_SIZE;
			cp++;

		}
		if (d_fhash->size[mt2] == 2 && size > 2)return cont.rules;

		d_fhash->rules[mt2] = cont.rules;
		d_fhash->key[mt2] = key;
		d_fhash->size[mt2] = size;
		d_fhash->is_frequent[mt2] = (cont.rules.ant_count >= d_MIN_SUP);
		d_fhash->htable_index[mt2] = &d_fhash[mt2] - &d_fhash[0];

		//If data is stored is the same as the one we want, then done
		if (d_fhash->key[mt2] == key && d_fhash->is_frequent[mt2] == (cont.rules.ant_count >= d_MIN_SUP)
			&& d_fhash->htable_index[mt2] == (&d_fhash[mt2] - &d_fhash[0])){

			if (size < vars->d_MAX_SIZE - 1)
				d_mhash[d_fhash->htable_index[mt2]] = cont;
			return cont.rules;
		}

		d_fhash->key[mt2] = 0;
		mt2 = mt = (mt + 1) % HASH_SIZE;
	}


	return cont.rules;
}


__device__ void calculate_rule(int* items, int level, float *points,
	int *nrules, rule_t *d_rules1k, int *d_temps, TYPE* d_bitmaps, uint4 classes[NUM_CLASSES][C_SIZE],
	content_t *d_mhash, fhash_t *d_fhash, DeviceVariables *vars){

	rule_t rule;

	int d_MIN_SUP = vars->d_MIN_SUP;
	float d_N_TRAINING = vars->d_N_TRAINING, d_MIN_CONF = vars->d_MIN_CONF;

	//Only items of size 2 (rules of size 3) and above are retrieved and inserted in the hash table
	if (level > 1)
		rule = get_from_hash_table(items, level, d_mhash, d_fhash, d_rules1k, d_bitmaps, classes, vars);

	//Rules of size 2 are already preprocessed on the CPU, so the confidence and support values are already calculated
	else
		rule = d_rules1k[items[0]];

	//If the itemset / antecedent is frequent
	if (rule.ant_count >= d_MIN_SUP){

		//With only two classes, the confidence of other class is just 1 subtracted from the current class confidence value
		if (NUM_CLASSES == 2){

			float conf = (rule.consq_countRel[1] / d_N_TRAINING) / (rule.ant_count / d_N_TRAINING);

			if (rule.consq_countRel[1] >= d_MIN_SUP && conf >= d_MIN_CONF) {
				points[threadIdx.x + blockDim.x] += conf;
				nrules[threadIdx.x + blockDim.x]++; //class 1
			}
			if ((rule.ant_count - rule.consq_countRel[1]) >= d_MIN_SUP && (1 - conf) >= d_MIN_CONF) {
				points[threadIdx.x] += 1 - conf;
				nrules[threadIdx.x]++; // class 0
			}
		}
		else
		{
			//Calculate the confidence value for each class
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++)
				if (rule.consq_countRel[i] >= d_MIN_SUP && (rule.consq_countRel[i] / (float)rule.ant_count) >= d_MIN_CONF) {
					points[threadIdx.x + i * blockDim.x] += (rule.consq_countRel[i] / d_N_TRAINING) / (rule.ant_count / d_N_TRAINING);
					nrules[threadIdx.x + i * blockDim.x]++;
				}
		}
	}
}

__device__ void generate_rules(short items[MAX_FEATURES], int n_items,
	int k, int* nrules, float* points, rule_t *d_rules1k,
	int *d_temps, TYPE *d_bitmaps, uint4 classes[NUM_CLASSES][C_SIZE],
	content_t *d_mhash, fhash_t *d_fhash, DeviceVariables *vars)
{

	if (k > n_items) return;

	int itemset[5];
	int * d_indextable = vars->d_indextable;

	//Get the number of combinations of size k with n_items elements
	int max = GetBinCoeff_l(n_items, k);

	//Each thread calculates a balanced amount of rules, getting the lexicographical combination from the index table
	for (int tid = threadIdx.x; tid < max; tid += blockDim.x){
		for (int i = 0; i < k; i++)
			itemset[i] = items[d_indextable[tid + i * max]];

		calculate_rule(itemset, k, points, nrules, d_rules1k, d_temps, d_bitmaps, classes, d_mhash, d_fhash, vars);
	}

}

//Adapted block reduction from Mark Harris
__device__ void rules_points_reduction(float points[num_threads * NUM_CLASSES], int nrules[num_threads * NUM_CLASSES]){

	float psum[NUM_CLASSES];
	int nsum[NUM_CLASSES];
#pragma unroll
	for (int i = 0; i < NUM_CLASSES; i++){
		psum[i] = points[threadIdx.x + i * blockDim.x];
		nsum[i] = nrules[threadIdx.x + i * blockDim.x];
	}

	__syncthreads();

	if (blockDim.x >= 1024) {
		if (threadIdx.x < 512) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				points[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + points[threadIdx.x + 512 + i * blockDim.x];
				nrules[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + nrules[threadIdx.x + 512 + i * blockDim.x];
			}

		} __syncthreads();
	}
	if (blockDim.x >= 512) {
		if (threadIdx.x < 256) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				points[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + points[threadIdx.x + 256 + i * blockDim.x];
				nrules[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + nrules[threadIdx.x + 256 + i * blockDim.x];
			}
		} __syncthreads();
	}
	if (blockDim.x >= 256) {
		if (threadIdx.x < 128) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				points[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + points[threadIdx.x + 128 + i * blockDim.x];
				nrules[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + nrules[threadIdx.x + 128 + i * blockDim.x];
			}
		} __syncthreads();
	}
	if (blockDim.x >= 128) {
		if (threadIdx.x <  64) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				points[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + points[threadIdx.x + 64 + i * blockDim.x];
				nrules[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + nrules[threadIdx.x + 64 + i * blockDim.x];
			}
		} __syncthreads();
	}

	if (threadIdx.x < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile int* smem = nrules;
		volatile float* smemf = points;

		if (blockDim.x >= 64) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				smemf[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + smemf[threadIdx.x + 32 + i * blockDim.x];
				smem[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + smem[threadIdx.x + 32 + i * blockDim.x];
			}
		}
		if (blockDim.x >= 32) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				smemf[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + smemf[threadIdx.x + 16 + i * blockDim.x];
				smem[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + smem[threadIdx.x + 16 + i * blockDim.x];
			}
		}
		if (blockDim.x >= 16) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				smemf[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + smemf[threadIdx.x + 8 + i * blockDim.x];
				smem[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + smem[threadIdx.x + 8 + i * blockDim.x];
			}
		}
		if (blockDim.x >= 8) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				smemf[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + smemf[threadIdx.x + 4 + i * blockDim.x];
				smem[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + smem[threadIdx.x + 4 + i * blockDim.x];
			}
		}
		if (blockDim.x >= 4) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				smemf[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + smemf[threadIdx.x + 2 + i * blockDim.x];
				smem[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + smem[threadIdx.x + 2 + i * blockDim.x];
			}
		}
		if (blockDim.x >= 2) {
#pragma unroll
			for (int i = 0; i < NUM_CLASSES; i++){
				smemf[threadIdx.x + i * blockDim.x] = psum[i] = psum[i] + smemf[threadIdx.x + 1 + i * blockDim.x];
				smem[threadIdx.x + i * blockDim.x] = nsum[i] = nsum[i] + smem[threadIdx.x + 1 + i * blockDim.x];
			}
		}
	}

}


//Gives a ranking score to each test document, where a thread block process a single document
__global__ void lazy_supervised_ranking_kernel(listTest_t *tests, int k,
	float* finalpoints, int* gn_rules, prediction_t *d_predictions,
	rule_t *d_rules1k, int *d_temps, TYPE *d_bitmaps, content_t *d_mhash, fhash_t *d_fhash,
	int n_tests, DeviceVariables vars)
{

	short true_label, n_items, default_prediction = 0;

	__shared__ float points[num_threads * NUM_CLASSES]; //points for each class
	__shared__ int nrules[num_threads * NUM_CLASSES]; //Number of strong rules that each thread generates
	__shared__ short items[MAX_FEATURES]; //the document's feature values that all threads of a block work on it
	__shared__ uint4 classes[NUM_CLASSES][C_SIZE]; //The classes bitmap are used on every rule, since the class is always the consequent

	int ridx = vars.d_bitmap_size / 4;

	//Stores the classes bitmaps in the shared memory
#pragma unroll
	for (int j = 0; j < NUM_CLASSES; j++){
		if (NUM_CLASSES == 2) j++;
		for (int i = threadIdx.x; i < ridx; i += blockDim.x){
			classes[j][i] = ((uint4*)d_contClasses[j].bitmap)[i];
		}
	}

	//initialize the vectors
#pragma unroll
	for (int i = 0; i < NUM_CLASSES; i++)
	{
		if (vars.d_COUNT_TARGET[i] >= vars.d_COUNT_TARGET[default_prediction])
			default_prediction = i;

		points[threadIdx.x + i * blockDim.x] = 0;
		nrules[threadIdx.x + i * blockDim.x] = 0;
	}

	true_label = tests[blockIdx.x].label;
	n_items = tests[blockIdx.x].size;

	//copy the document's content to shared memory for this block
	for (int i = threadIdx.x; i < n_items; i += blockDim.x)
	{
		items[i] = tests[blockIdx.x].instance[i];
	}

	//To avoid division by zero on score calculation
	if (k == 1){
		nrules[0] = 1;
		nrules[blockDim.x] = 1;
	}

	__syncthreads();

	//generate rules for this document, with all threads of this block
	generate_rules(items, n_items, k, nrules, points, d_rules1k, d_temps, d_bitmaps, classes, d_mhash, d_fhash, &vars);

	//do a reduction operation to sum all scores and rule quantity that each thread got
	rules_points_reduction(points, nrules);

	//First thread updates the scores
	if (threadIdx.x == 0){

		//Since the kernel is launched for each rule size, store in the global memory the current result
		for (int i = 0; i < NUM_CLASSES; i++){
			gn_rules[blockIdx.x + i * gridDim.x] += nrules[i * blockDim.x];
			finalpoints[blockIdx.x + i * gridDim.x] += points[i * blockDim.x];
		}

		//If it's the last rule size, do the score calculation by using the final values in global memory
		if (k + 1 == vars.d_MAX_SIZE){
			d_predictions[blockIdx.x].score = get_total_score(finalpoints, gn_rules, n_tests, blockIdx.x);//pega o score de cada classe, de forma ordenada
			d_predictions[blockIdx.x].label = d_predictions[blockIdx.x].score.ordered_labels[0];//o primeiro da ordenação (maior score) é usado como o label/classe previsto

			for (int i = 0; i < NUM_CLASSES; i++)
				d_predictions[blockIdx.x].rules += gn_rules[blockIdx.x + i * gridDim.x] - 1;

			if (d_predictions[blockIdx.x].rules == 0)
				d_predictions[blockIdx.x].label = default_prediction;

			d_predictions[blockIdx.x].correct = (d_predictions[blockIdx.x].label == true_label);

		}

	}

}

