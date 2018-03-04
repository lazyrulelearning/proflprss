
//Reads the file that has the features ordered by their Chi-squared values, which is used during the partition process
void read_ordered_features(vector<short>& ordered_features, char *file_features){

	if (!file_features){
		fprintf(stderr, "No feature file specified.\n");
		exit(EXIT_FAILURE);
	}

	FILE *in = fopen(file_features, "r");
	int x;
	while (fscanf(in, "%d", &x) != EOF){
		ordered_features.push_back(x);
	}

	fclose(in);
}

//Reads the file that contains the training documents, where the inverted lists of each feature will be built.
//It can be adapted to read an actual unlabeled dataset, and act in a semi-supervised manner with annotators
void read_unlabeled_set(char *training){

	fprintf(stderr, "Reading training data.\n");

	int target = -1;
	char** target_name = (char**)malloc(sizeof(char*)*NUM_CLASSES);

	for (int i = 0; i < NUM_CLASSES; i++)
		target_name[i] = (char*)malloc(sizeof(char) * 100);

	set<string> proc_items;

	FILE* file = fopen(training, "r");

	if (file == NULL) {
		fprintf(stderr, "Training set %s not found.\n\n", training);
		exit(-1);
	}

	N_ITEMSETS = 0;
	N_TRAINING = 0;
	ITEM_MAP.clear();
	SYMBOL_TABLE.clear();
	CLASS_NAME.clear();

	//We use the class label as an item too, so that it allows feature intersection to build the association rules,
	//with the class always being the consequent.
	for (int i = 0; i < NUM_CLASSES; i++) {
		sprintf(target_name[i], "CLASS=%d", i);
		CLASS_NAME[target_name[i]] = i;
		COUNT_TARGET[i] = 0;
		TARGET_ID[i] = N_ITEMSETS;
		SYMBOL_TABLE[N_ITEMSETS] = strdup(target_name[i]);
		ITEM_MAP[target_name[i]] = N_ITEMSETS;
		++N_ITEMSETS;
	}

	while (1) {

		char line[200 * KB];
		fgets(line, 200 * KB, file);

		if (feof(file)) break;

		N_TRAINING++;
		vector<int> doc;

		proc_items.clear();
		target = -1;

		char* item = strtok(line, " \t\n");

		qids.push_back(atoi(item));

		while ((item = strtok(NULL, " \t\n")) != NULL) {

			//Counts occurrence of each class
			if (CLASS_NAME.find(item) != CLASS_NAME.end()) {
				target = (int)CLASS_NAME[item];
				COUNT_TARGET[target]++;
			}

			if (proc_items.find(item) == proc_items.end()) {

				proc_items.insert(item);

				if (ITEM_MAP.find(item) == ITEM_MAP.end()) {//If an item does not exists, create it				

					//Keep a map of its string value, so that we can output the reduced version later
					SYMBOL_TABLE[N_ITEMSETS] = strdup(item);
					ITEM_MAP[item] = N_ITEMSETS;
					doc.push_back(N_ITEMSETS);

					++N_ITEMSETS;

					if (MAX_ITEMS < N_ITEMSETS)	{
						fprintf(stderr, "The number of features/items in Training set is lower than  MAX_ITEMS. Please change it in limits.cuh.\n\n");
					}
				}
				else{
					//insert item into the doc vector
					doc.push_back(ITEM_MAP[item]);
				}
			}
		}

		vector<short> vdoc(doc.begin(), doc.end());
		UNLABELED.push_back(vdoc);//insert the document in the list of documents
	}

	fprintf(stderr, "done\n");
	fprintf(stderr, "%d Transactions read.\n", N_TRAINING);

	for (int i = 0; i < NUM_CLASSES; i++)
		free(target_name[i]);

	free(target_name);
	fclose(file);
}
