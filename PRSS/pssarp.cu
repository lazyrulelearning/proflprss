
//Makes intersections of each item from the document, producing rules of size up to MAX_SIZE
void gen_rules(vector<short>& pdoc, content_t *items, set<short>& curr_items, int *r_qnt){

	int rules_qnt = 0;
	vector<short> projected;

	projected.clear();
	//The document is first projected according to the items that exist in the reduced set.
	//The ones that have a bitmap with at least one bit 1.
	for (int i = 1; i < pdoc.size(); i++){
		if (curr_items.find(pdoc[i]) != curr_items.end()){
			projected.push_back(pdoc[i]);
		}
	}
	if (projected.empty()){
		*r_qnt = 0;
		return;
	}

	int proj_size = (int)projected.size();
	//Check the rules of size 2 satisfies the support and confidence thresholds
	for (int i = 0; i < proj_size; i++){
		rule_t rule;
		for (int m = 0; m < NUM_CLASSES; m++){
			rule = items[projected[i]].rules;
			if ((rule.consq_countRel[m] / (float)rule.ant_count) >= MIN_CONF){
				rules_qnt++;
			}
		}
	}

	content_t temp, temp2;
	rule_t rule;

	//Intersects two items and a class item to produce a rule of size 3
	if (MAX_SIZE > 2)
		for (int i = 0; i < proj_size - 1; i++){

			for (int j = i + 1; j < proj_size; j++){

				rule.ant_count = 0;
				for (unsigned int b = 0; b < bitmap_size; b++){
					temp.bitmap[b] = items[projected[i]].bitmap[b] & items[projected[j]].bitmap[b];
					rule.ant_count += __popcounter(temp.bitmap[b]);
				}

				for (int m = 0; m < NUM_CLASSES; m++){

					rule.consq_countRel[m] = 0;
					if (rule.ant_count >= MIN_SUP){

						for (unsigned int b = 0; b < bitmap_size; b++){
							rule.consq_countRel[m] += __popcounter(temp.bitmap[b] & items[m].bitmap[b]);
						}

						if ((rule.consq_countRel[m] / (float)rule.ant_count) >= MIN_CONF){
							rules_qnt++;

						}
					}
				}
			}
		}

	//Intersects three items and a class item to produce a rule of size 4
	if (MAX_SIZE > 3)
		for (int i = 0; i < proj_size - 2; i++){

			for (int j = i + 1; j < proj_size - 1; j++){

				rule.ant_count = 0;
				for (unsigned int b = 0; b < bitmap_size; b++){
					temp.bitmap[b] = items[projected[i]].bitmap[b] & items[projected[j]].bitmap[b];
					rule.ant_count += __popcounter(temp.bitmap[b]);
				}

				if (rule.ant_count >= MIN_SUP){
					for (int k = j + 1; k < proj_size; k++){

						rule.ant_count = 0;
						for (unsigned int b = 0; b < bitmap_size; b++){
							temp2.bitmap[b] = temp.bitmap[b] & items[projected[k]].bitmap[b];
							rule.ant_count += __popcounter(temp2.bitmap[b]);
						}

						for (int m = 0; m < NUM_CLASSES; m++){

							rule.consq_countRel[m] = 0;
							if (rule.ant_count >= MIN_SUP){

								for (unsigned int b = 0; b < bitmap_size; b++){
									rule.consq_countRel[m] += __popcounter(temp2.bitmap[b] & items[m].bitmap[b]);
								}

								if ((rule.consq_countRel[m] / (float)rule.ant_count) >= MIN_CONF){
									rules_qnt++;
								}
							}
						}
					}
				}

			}
		}

	*r_qnt = rules_qnt;
}

//Chooses the document that generates the fewest rules, by using the current bitmaps and current items
int find_least_representative(vector< vector<short> >&unlabeled_partition, content_t *items,
	set<short>& curr_items, vector<int>& doc_occurs){

	int min_rules = 1 << 30, idx = 0;

#pragma omp parallel for
	//iterate over the documents of this partition
	for (int j = 0; j < unlabeled_partition.size(); j++){
		int  rules_qnt;
		//generate its rules
		gen_rules(unlabeled_partition[j], items, curr_items, &rules_qnt);

		//Choose the document with fewest rules. In case of a tie, choose the document with greater occurrences/DF sum or greater ID.
#pragma omp critical
		{
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
	}

	return idx;
}

//Update the bitmaps with the new document and its items
void update_bitmap(vector<short>& new_doc, set<short>& curr_items, content_t *items){

	//Increase the number of documents in the current reduced set
	++N_TRAINING;

	//Turns on the bits of the items that compose this document
	for (unsigned int i = 0; i < new_doc.size(); i++){

		++items[new_doc[i]].rules.ant_count;
		items[new_doc[i]].bitmap[N_TRAINING / BITMAP_SLOT_SIZE] |= 1ULL << (N_TRAINING % BITMAP_SLOT_SIZE);
		curr_items.insert(new_doc[i]);
	}

	bitmap_size = bitMAP_SIZE;

	//Recalculates the rule of size 2 with each class, by performing the intersection with AND operation
	for (int i = 0; i < new_doc.size(); i++){

		for (int j = 0; j < NUM_CLASSES; j++){

			items[new_doc[i]].rules.consq_countRel[j] = 0;
			for (unsigned int b = 0; b < bitmap_size; b++){
				items[new_doc[i]].rules.consq_countRel[j] += __popcounter(items[new_doc[i]].bitmap[b] & items[j].bitmap[b]);
			}

		}
	}


}

//Multithreaded version of SSARP
void selective_sampling(int partitions, char * file_features){

	vector< vector< vector<short> > > unlabeled_partitions, reduced_trainings;
	vector< vector<int> > reduced_trs_ids;
	vector< set<int> > reduced_trs_sets;
	int occurrences[MAX_ITEMS];
	vector<int> doc_occurs;

	unlabeled_partitions.resize(partitions);
	reduced_trainings.resize(partitions);
	reduced_trs_ids.resize(partitions);
	reduced_trs_sets.resize(partitions);

	//distribute features to partitions
	partitioner(unlabeled_partitions, file_features);

	build_occurrences(occurrences);

	content_t *items = (content_t *)calloc(N_ITEMSETS, sizeof(content_t));

	//Iterate over the partitions
	for (int p = 0; p < partitions; p++){

		docs_DF_sum(unlabeled_partitions[p], occurrences, doc_occurs);
		int id = find_most_representative(unlabeled_partitions[p], doc_occurs);
		reduced_trs_ids[p].push_back(id);
		reduced_trainings[p].push_back(unlabeled_partitions[p][id]);
		int iterations = 1;
		set<short> curr_items;
		curr_items.clear();
		memset(items, 0, N_ITEMSETS * sizeof(content_t));
		N_TRAINING = 0;

		//Iterate until convergence
		while (1){
			//Update the bitmaps with the newly inserted document
			update_bitmap(reduced_trainings[p][N_TRAINING], curr_items, items);
			//Find the document that generates the fewest rules
			int idx = find_least_representative(unlabeled_partitions[p], items, curr_items, doc_occurs);
			//If this document is already in the reduced set, convergence has been reached
			if (reduced_trs_sets[p].find(idx) != reduced_trs_sets[p].end())
			{
				fprintf(stderr, "Partition %d: %d is already in reduced set\n\tReduced in %d iterations\n", p, idx + 1, iterations);
				break;
			}

			//insert the new document
			iterations++;
			reduced_trs_ids[p].push_back(idx);
			reduced_trs_sets[p].insert(idx);
			reduced_trainings[p].push_back(unlabeled_partitions[p][idx]);

		}

	}

	free(items);
	//Merge all reduced sets
	merge(reduced_trs_ids, partitions);

}
