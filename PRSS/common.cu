
//Fills the vector that counts the occurrences of each feature/item (document frequency - DF) 
void build_occurrences(int* occurrences){

	vector< vector<short> >& unlabeled_partition = UNLABELED;
	memset(occurrences, 0, sizeof(int) * MAX_ITEMS);

#pragma omp parallel for	
	for (int i = 0; i < unlabeled_partition.size(); i++){
		for (int j = 1; j < unlabeled_partition[i].size(); j++){
			int x = unlabeled_partition[i][j];
#pragma omp atomic
			occurrences[x]++;
		}
	}
}

//Fills the vector that store the sum of the DFs of the items that compose the document, for all documents
void docs_DF_sum(vector< vector<short> >& unlabeled_partition, int* occurrences, vector<int>& doc_occurs){

	doc_occurs.resize(unlabeled_partition.size());

#pragma omp parallel for
	for (int i = 0; i < unlabeled_partition.size(); i++){

		int cnt = 0;

		for (int j = 1; j < unlabeled_partition[i].size(); j++){
			cnt += occurrences[unlabeled_partition[i][j]];
		}

		doc_occurs[i] = cnt;
	}
}

//First chooses the document that has the greatest DF sum. On a tie, choose the one with greater id
int find_most_representative(vector< vector<short> >& unlabeled_partition, vector<int>& doc_occurs){

	int max = -1, idx = 0;

	for (int i = 0; i < unlabeled_partition.size(); i++){

		int cnt = doc_occurs[i];

		if (max < cnt){
			max = cnt;
			idx = i;
		}
		else if (max == cnt && idx < i)
			idx = i;
	}

	return idx;
}

//Merge the reduced sets produced from each partition, with a set union operation
void merge(vector< vector<int> >& reduced_trs_ids, int partitions){

	fprintf(stderr, "Merging reduced sets\n");
	set<int> reduced_uniq;

	//merge with their ids and a set container
	for (int p = 0; p < partitions; p++){
		for (int i = 0; i < reduced_trs_ids[p].size(); i++)
			reduced_uniq.insert(reduced_trs_ids[p][i]);
	}

	fprintf(stderr, "Reduced set has %u lines.\n", reduced_uniq.size());

	//Then print the reduced set according to the original data that was read
	for (set<int>::iterator it = reduced_uniq.begin(); it != reduced_uniq.end(); it++){

		printf("%d CLASS=%d ", *it + 1, UNLABELED[*it][0]);
		for (int i = 1; i < UNLABELED[*it].size(); i++)
			printf("%s ", SYMBOL_TABLE[UNLABELED[*it][i]].data());
		putchar('\n');
	}
}

//Creates the partitions by using the ordered features file and the read data.
void partitioner(vector< vector< vector<short> > >& unlabeled_partitions, char *file_features){

	vector<short> ordered_features;
	read_ordered_features(ordered_features, file_features);
	int partitions = (int)unlabeled_partitions.size();

	for (int i = 0; i < UNLABELED.size(); i++){

		for (int j = 0; j < partitions; j++){
			unlabeled_partitions[j].push_back(vector<short>());
			unlabeled_partitions[j][i].push_back(UNLABELED[i][0]); //class
		}

		int k = 0;
		//Assign with a round-robin manner
		for (int j = 0; j < ordered_features.size(); j++){

			unlabeled_partitions[k % partitions][i].push_back(UNLABELED[i][ordered_features[j]]);
			k++;
		}
	}
}

