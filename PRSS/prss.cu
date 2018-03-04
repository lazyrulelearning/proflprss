
#include "kernels.cu"

//Chooses the document that generates the fewest rules, by using the current bitmaps and current items
int find_least_representativeCUDA(vector< vector<short> >&unlabeled_partition,
	vector<int>& doc_occurs, vector<Device_Variables>& dev_vars){
	
	dim3 block, thread;

	int N_DOCS = (int)unlabeled_partition.size();
#pragma omp parallel for 
	for (int i = 0; i < gpu_cnt; i++){

		int gpu_id = omp_get_thread_num();

		gpuErrchk(cudaSetDevice(gpu_id));

		int TEST_CNT = ((int)ceil(N_DOCS / (double)gpu_cnt));
		int n_docs = (gpu_id + 1) * TEST_CNT > N_DOCS ? N_DOCS - gpu_id * TEST_CNT : TEST_CNT;
		int offset = gpu_id * TEST_CNT;

		short *d_docs = dev_vars[gpu_id].d_docs;
		int *d_rulecount = dev_vars[gpu_id].d_rulecount;
		TYPE *d_bitmaps = dev_vars[gpu_id].d_bitmaps;

		rule_counter_kernel << < (n_docs / num_threads) + 1, num_threads >> >(d_docs, d_bitmaps, d_rulecount,
			n_docs, MAX_SIZE, bitmap_size, N_ITEMSETS, (int)(unlabeled_partition[0].size() - 1));

		gpuErrchk(cudaMemcpy(h_rulecount + offset, d_rulecount, sizeof(int) * n_docs, cudaMemcpyDeviceToHost));
	}

	int min_rules = 1 << 30, idx = 0;

	for (int j = 0; j < unlabeled_partition.size(); j++){

		int rules_qnt = h_rulecount[j];		

		if (min_rules == rules_qnt){

			if (doc_occurs[j] < doc_occurs[idx]){
				idx = j;
			}
			else {
				if (doc_occurs[j] == doc_occurs[idx]){
					if (idx < j){
						idx = j;
					}
				}
			}
		}
		else{
			if (min_rules > rules_qnt){
				min_rules = rules_qnt;
				idx = j;
			}
		}
	}

	return idx;
}

//Update the bitmaps with the new document and its items
void update_bitmap_in_device(vector<short>& new_doc, vector<Device_Variables>& dev_vars){


	++N_TRAINING;

	Dev_doc doc;

	for (unsigned int i = 0; i < new_doc.size(); i++){

		doc.features[i] = new_doc[i];
	}

	bitmap_size = bitMAP_SIZE;

#pragma omp parallel for 
	for (int i = 0; i < gpu_cnt; i++){

		int gpu_id = omp_get_thread_num();
		gpuErrchk(cudaSetDevice(gpu_id));

		bitmap_updater << <1, 512 >> >((int)new_doc.size(), dev_vars[gpu_id].d_bitmaps, doc, N_TRAINING, N_ITEMSETS);
		//gpuErrchk(cudaDeviceSynchronize());
	}
}

//multi GPU version of SSARP, called PRSS (Parallel Rule-based Selective Sampling)
void selective_samplingCUDA(int partitions, char * file_features){

	vector< vector< vector<short> > > unlabeled_partitions, reduced_trainings;
	vector< vector<int> > reduced_trs_ids;
	vector< set<int> > reduced_trs_sets;
	vector<int> doc_occurs;
	int occurrences[MAX_ITEMS];
	int N_DOCS;

	unlabeled_partitions.resize(partitions);
	reduced_trainings.resize(partitions);
	reduced_trs_ids.resize(partitions);
	reduced_trs_sets.resize(partitions);

	//distribute features to partitions
	partitioner(unlabeled_partitions, file_features);

	build_occurrences(occurrences);

	N_DOCS = (int)unlabeled_partitions[0].size();

	int DOC_SIZE = MAX_FEATURES / partitions + 1;

	size_t mem = sizeof(int) * N_DOCS;
	mem += sizeof(short) *N_DOCS * DOC_SIZE
		+ N_ITEMSETS * sizeof(content_t);

	cerr << "Memory used: " << mem / (float)(1 << 20) << "MB" << endl;
	
	vector<Device_Variables> dev_vars(gpu_cnt);

	//Allocate the variables that will store the rule count of each document and the projected documents
	gpuErrchk(cudaHostAlloc(&h_rulecount, N_DOCS * sizeof(int), cudaHostAllocPortable));
	gpuErrchk(cudaHostAlloc(&h_projected, sizeof(short) * N_DOCS * DOC_SIZE, cudaHostAllocPortable));


#pragma omp parallel for 
	//Launch a thread for each GPU
	for (int i = 0; i < gpu_cnt; i++){

		int gpu_id = omp_get_thread_num();

		gpuErrchk(cudaSetDevice(gpu_id));

		//Each GPU works with a part of the documents
		int TEST_CNT = ((int)ceil(N_DOCS / (double)gpu_cnt));
		int n_docs = (gpu_id + 1) * TEST_CNT > N_DOCS ? N_DOCS - gpu_id * TEST_CNT : TEST_CNT;
	
		//Allocate memory in GPU according to the assigned number of documents
		gpuErrchk(cudaMalloc(&dev_vars[gpu_id].d_rulecount, sizeof(int) * n_docs));
		gpuErrchk(cudaMalloc(&dev_vars[gpu_id].d_docs, sizeof(short) * n_docs * DOC_SIZE));
		gpuErrchk(cudaMalloc(&dev_vars[gpu_id].d_bitmaps, N_ITEMSETS * sizeof(content_t)));

	}

	for (int p = 0; p < partitions; p++){

		//Compute in CPU the DFs and first document
		docs_DF_sum(unlabeled_partitions[p], occurrences, doc_occurs);
		int id = find_most_representative(unlabeled_partitions[p], doc_occurs);
		reduced_trs_ids[p].push_back(id);
		reduced_trainings[p].push_back(unlabeled_partitions[p][id]);

		int iterations = 1;
		N_TRAINING = 0;
		int size = (int) unlabeled_partitions[p][0].size();

		//Each cpu thread assigns the data so that it allows coalesced access in their respective GPU
#pragma omp parallel 
		{
			int gpu_id = omp_get_thread_num();

			int TEST_CNT = ((int)ceil(N_DOCS / (double)gpu_cnt));
			int n_docs = (gpu_id + 1) * TEST_CNT > N_DOCS ? N_DOCS - gpu_id * TEST_CNT : TEST_CNT;
			int offset = gpu_id * TEST_CNT;

			for (int i = 0; i < n_docs; i++){

				for (int j = 1; j < size; j++){

					h_projected[offset *(size - 1) + i + (j - 1) * n_docs] = unlabeled_partitions[p][i + offset][j];
				}
			}
		}

#pragma omp parallel for 
		//Initialize and copy data to each GPU
		for (int i = 0; i < gpu_cnt; i++){

			int gpu_id = omp_get_thread_num();

			gpuErrchk(cudaSetDevice(gpu_id));

			int TEST_CNT = ((int)ceil(N_DOCS / (double)gpu_cnt));
			int n_docs = (gpu_id + 1) * TEST_CNT > N_DOCS ? N_DOCS - gpu_id * TEST_CNT : TEST_CNT;
			int offset = gpu_id * TEST_CNT;

			citem_memset << <16, 64 >> >(N_ITEMSETS);
			gpuErrchk(cudaMemsetAsync(dev_vars[gpu_id].d_bitmaps, 0, N_ITEMSETS * sizeof(content_t)));

			gpuErrchk(cudaMemcpyAsync(dev_vars[gpu_id].d_docs, h_projected + offset * (size - 1), sizeof(short) *
				n_docs * (size - 1), cudaMemcpyHostToDevice));
		}

		//Iterate until convergence
		while (1){

			//Launch a kernel to update the bitmap in the GPU
			update_bitmap_in_device(reduced_trainings[p][N_TRAINING], dev_vars);

			//Launch a kernel to find the document that generates the fewest rules
			int idx = find_least_representativeCUDA(unlabeled_partitions[p], doc_occurs, dev_vars);

			//If already inserted, convergence has been reached
			if (reduced_trs_sets[p].find(idx) != reduced_trs_sets[p].end())
			{
				fprintf(stderr, "Partition %d: %d is already in reduced set\n\tReduced in %d iterations\n", p, idx + 1, iterations);
				break;
			}
			//Insert the new document in the current reduced set
			iterations++;
			reduced_trs_ids[p].push_back(idx);
			reduced_trs_sets[p].insert(idx);
			reduced_trainings[p].push_back(unlabeled_partitions[p][idx]);

		}

	}

#pragma omp parallel for 
	//Free the GPUs variables
	for (int i = 0; i < gpu_cnt; i++){

		gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaFree(dev_vars[i].d_rulecount));
		gpuErrchk(cudaFree(dev_vars[i].d_docs));
		gpuErrchk(cudaFree(dev_vars[i].d_bitmaps));
		//gpuErrchk(cudaFreeHost(dev_vars[i].h_projected));
		
	}
	gpuErrchk(cudaFreeHost(h_projected));
	gpuErrchk(cudaFreeHost(h_rulecount));
	//merge the reduced sets
	merge(reduced_trs_ids, partitions);
}