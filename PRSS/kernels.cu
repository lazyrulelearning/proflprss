#define  D_MIN_CONF 0.001f // set the confidence here


__device__ rule_t rules[MAX_ITEMS];
__device__ char citem_set[MAX_ITEMS];

//Check the threshold of the pre computed rules of size 2
__device__ int gen_rules_s2(short *doc, int size, int MAX_SIZE, unsigned int bmpsize){

	int r_count = 0;

	rule_t rule;

	for (int i = 0; i < size; i++){

		rule = rules[doc[i]];
		for (int m = 0; m < NUM_CLASSES; m++){

			if ((rule.consq_countRel[m] / (float)rule.ant_count) >= D_MIN_CONF){
				r_count++;
			}
		}
	}
	return r_count;
}

//Makes rules of size 3 by intersecting 2 items
__device__ int gen_rules_s3(short *doc, int size, TYPE* items, int MAX_SIZE, unsigned int bmpsize,
	int N_ITEMS, TYPE classes[NUM_CLASSES*(sizeof(content_t) / sizeof(TYPE))]){

	int r_count = 0;

	content_t cont1;

	for (int i = 0; i < size - 1; i++){

		for (int j = i + 1; j < size; j++){

			cont1.rules.ant_count = 0;
			for (unsigned int b = 0; b < bmpsize; b++){
				cont1.bitmap[b] = items[doc[i] + b * N_ITEMS] & items[doc[j] + b * N_ITEMS];
				cont1.rules.ant_count += __gpu_pop(cont1.bitmap[b]);
			}

			if (cont1.rules.ant_count >= 1){
				for (int m = 0; m < NUM_CLASSES; m++){

					cont1.rules.consq_countRel[m] = 0;

					for (unsigned int b = 0; b < bmpsize; b++){
						cont1.rules.consq_countRel[m] += __gpu_pop(cont1.bitmap[b] & classes[m + NUM_CLASSES * b]);
					}

					if ((cont1.rules.consq_countRel[m] / (float)cont1.rules.ant_count) >= D_MIN_CONF){
						r_count++;

					}
				}
			}
		}
	}

	return r_count;
}

//Makes rules of size 4 by intersecting 3 items
__device__ int gen_rules_s4(short *doc, int size, TYPE* items, int MAX_SIZE, unsigned int bmpsize,
	int N_ITEMS, TYPE classes[NUM_CLASSES*(sizeof(content_t) / sizeof(TYPE))]){

	int r_count = 0;

	content_t cont1;
	content_t cont2;

	for (int i = 0; i < size - 2; i++){

		for (int j = i + 1; j < size - 1; j++){

			cont1.rules.ant_count = 0;
			for (unsigned int b = 0; b < bmpsize; b++){
				cont1.bitmap[b] = items[doc[i] + b * N_ITEMS] & items[doc[j] + b * N_ITEMS];
				cont1.rules.ant_count += __gpu_pop(cont1.bitmap[b]);
			}

			if (cont1.rules.ant_count >= 1){
				for (int k = j + 1; k < size; k++){

					cont2.rules.ant_count = 0;
					for (unsigned int b = 0; b < bmpsize; b++){
						cont2.bitmap[b] = cont1.bitmap[b] & items[doc[k] + b * N_ITEMS];
						cont2.rules.ant_count += __gpu_pop(cont2.bitmap[b]);
					}
					if (cont2.rules.ant_count >= 1){
						for (int m = 0; m < NUM_CLASSES; m++){

							cont2.rules.consq_countRel[m] = 0;
							for (unsigned int b = 0; b < bmpsize; b++){
								cont2.rules.consq_countRel[m] += __gpu_pop(cont2.bitmap[b] & classes[m + b * NUM_CLASSES]);
							}

							if ((cont2.rules.consq_countRel[m] / (float)cont2.rules.ant_count) >= D_MIN_CONF){
								r_count++;
							}

						}
					}
				}
			}
		}
	}

	return r_count;
}

//Makes intersections of each item from the document, producing rules of size up to MAX_SIZE
__device__ __forceinline__ int gen_rules(short *doc, int size, TYPE*  items, int MAX_SIZE, unsigned int bmpsize,
	int N_ITEMS, TYPE classes[NUM_CLASSES*(sizeof(content_t) / sizeof(TYPE))]){


	int r_count = 0;

	r_count += gen_rules_s2(doc, size, MAX_SIZE, bmpsize);

	if (MAX_SIZE > 2)
		r_count += gen_rules_s3(doc, size, items, MAX_SIZE, bmpsize, N_ITEMS, classes);

	if (MAX_SIZE > 3)
		r_count += gen_rules_s4(doc, size, items, MAX_SIZE, bmpsize, N_ITEMS, classes);


	return r_count;
}

//Kernel that computes the rules for each document. A thread per document
__global__ void rule_counter_kernel(short *docs, TYPE* bitmaps,
	int* rulecount, int N, int MAX_SIZE, unsigned int bmpsize, int N_ITEMS, int doc_size){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	//A third is enough, given the suggested number of partitions is usually greater than 5
	short doc[MAX_FEATURES / 3];

	//Classes bitmaps are stored in shared memory since they are always used
	__shared__ TYPE classes[NUM_CLASSES * (sizeof(content_t) / sizeof(TYPE))];

	for (int m = 0; m < NUM_CLASSES; m++){
		for (int i = threadIdx.x; i < bmpsize; i += blockDim.x){
			classes[m + i *NUM_CLASSES] = bitmaps[m + i * N_ITEMS];
		}
	}
	__syncthreads();
	//Process more than one document if possible
	for (; idx < N; idx += blockDim.x * gridDim.x){

		int  size = 0, f;

		for (int i = 0; i < doc_size; i++){
			f = docs[idx + i * N];
			if (citem_set[f])//project the document
				doc[size++] = f;
		}

		if (!size)
			rulecount[idx] = 0;
		else
			rulecount[idx] = gen_rules(doc, size, bitmaps, MAX_SIZE, bmpsize, N_ITEMS, classes);
	}
}

//Kernel that updates the bitmaps according to the new documents items. A thread per item
__global__ void bitmap_updater(int N, TYPE *d_bitmaps, Dev_doc new_doc, int N_TRAINING, int N_ITEMS){

	int i = threadIdx.x;
	if (i < N){

		int idx = new_doc.features[i];
		citem_set[idx] = 1;// turn on the current item

		++rules[idx].ant_count;
		d_bitmaps[idx + N_ITEMS * (N_TRAINING / BITMAP_SLOT_SIZE)] |= 1ULL << (N_TRAINING % BITMAP_SLOT_SIZE);

		unsigned int bmp_sz = bitMAP_SIZE;

		//Computes the rule of size 2 again
		for (int m = 0; m < NUM_CLASSES; m++){

			rules[idx].consq_countRel[m] = 0;

			for (unsigned int b = 0; b < bmp_sz; b++){
				rules[idx].consq_countRel[m] += __gpu_pop(d_bitmaps[b * N_ITEMS + idx] & d_bitmaps[b * N_ITEMS + m]);
			}
		}
	}
}

//Resets the current item vector
__global__ void citem_memset(int  N_ITEMSETS){

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N_ITEMSETS; i += blockDim.x * gridDim.x)
		citem_set[i] = 0;
}

