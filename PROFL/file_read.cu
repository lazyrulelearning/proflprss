

//Reads the file that contains the training documents, where the inverted lists of each feature will be built
int read_training_set(char* training) {
	
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

	int n_lines = 0;
	N_ITEMSETS = 0;
	N_TRAINING = 0;
	ITEM_MAP.clear();
	CLASS_NAME.clear();

	while (1) {
		char line[200 * KB];
		fgets(line, 200 * KB, file);
		if (feof(file)) break;
		n_lines++;
	}

	rewind(file);

	//We use the class label as an item too, so that it allows feature intersection to build the association rules,
	//with the class always being the consequent.
	for (int i = 0; i < NUM_CLASSES; i++) {
		sprintf(target_name[i], "CLASS=%d", i);
		CLASS_NAME[target_name[i]] = i;
		COUNT_TARGET[i] = 0;
		TARGET_ID[i] = N_ITEMSETS;
		ITEMSETS[N_ITEMSETS].count = 0;
		ITEMSETS[N_ITEMSETS].list.clear();
		ITEM_MAP[target_name[i]] = N_ITEMSETS;
		++N_ITEMSETS;
	}

	while (1) {

		char line[200 * KB];
		fgets(line, 200 * KB, file);

		if (feof(file)) break;

		N_TRAINING++;

		if (N_TRAINING > MAX_LINE_TRAINING)
		{
			fprintf(stderr, "The number of TRANSACTIONS in Training set is lower than MAX_LINE_TRAINING. Please change it in limits.cuh.\n\n");
			return 1;

		}
		proc_items.clear();
		target = -1;

		char* item = strtok(line, " \t\n");

		while ((item = strtok(NULL, " \t\n")) != NULL) {
			
			//Counts occurrence of each class		
			if (CLASS_NAME.find(item) != CLASS_NAME.end()) {

				target = (int)CLASS_NAME[item];
				COUNT_TARGET[target]++;				
			}		
			
			if (proc_items.find(item) == proc_items.end()) {

				proc_items.insert(item);

				if (ITEM_MAP.find(item) != ITEM_MAP.end()) {//If an item already exists, update its occurrence count

					int index = (int)(ITEM_MAP[item]);
					
					//Transaction/document list of where this item occurs
					ITEMSETS[index].list.push_back(N_TRAINING);//Occurs at the N-TRAINING-th transaction
					ITEMSETS[index].count++;					
				}
				else {

					ITEMSETS[N_ITEMSETS].list.clear();
					ITEMSETS[N_ITEMSETS].count = 1;

					ITEMSETS[N_ITEMSETS].list.push_back(N_TRAINING);//Number of the first transaction where this item occurs

					ITEM_MAP[item] = N_ITEMSETS;

					++N_ITEMSETS;

					if (MAX_ITEMS < N_ITEMSETS)
					{
						fprintf(stderr, "The number of features/items in Training set is lower than  MAX_ITEMS. Please change it in limits.cuh.\n\n");
						return 1;
					}
				}
			}
		}
	}

	fprintf(stderr, "done\n");
	fprintf(stderr, "%d Transactions read.\n", N_TRAINING);

	for (int i = 0; i < NUM_CLASSES; i++)
		free(target_name[i]);

	free(target_name);
	fclose(file);

	return 0;
}


//Reads the file that contains the documents that are going to be ranked
int read_test_set(char* test)
{
	fprintf(stderr, "Reading test set.\n");
	
	int target = 0;
	FILE* file = fopen(test, "r");
	if (file == NULL)
	{
		fprintf(stderr, "Test file %s not found.\n\n", test);
		exit(-1);
	}

	N_TESTS = 0;
	int n_lines = 0;
	TEST = 0;
	while (1)
	{
		char line[200 * KB];
		fgets(line, 200 * KB, file);
		if (feof(file)) break;
		n_lines++;
	}
	rewind(file);
	set<int> instance;

	gpuErrchk(cudaHostAlloc(&TEST, sizeof(listTest_t) * n_lines, cudaHostAllocWriteCombined | cudaHostAllocPortable));
	if (!TEST){
		exit(EXIT_FAILURE);
	}

	for (int k = 0; k < n_lines; k++)
	{
		char line[200 * KB];
		fgets(line, 200 * KB, file);
		if (feof(file)) break;
		target = -1;
		int id = atoi(line);
		if (id == 0) break; 

		char* item = strtok(line, " \t\n");

		short countFeatures = 0;

		while ((item = strtok(NULL, " \t\n")) != NULL)
		{
			if (target == -1 && CLASS_NAME.find(item) != CLASS_NAME.end())
				target = (int)CLASS_NAME[item];

			//If the item didn't exist during the training data read, create it here			
			if (ITEM_MAP.find(item) == ITEM_MAP.end())
			{
				ITEM_MAP[item] = N_ITEMSETS;

				ITEMSETS[N_ITEMSETS].count = 0;

				ITEMSETS[N_ITEMSETS].list.clear();
				ITEMSETS[N_ITEMSETS].list.push_back(N_ITEMSETS);
				++N_ITEMSETS;

				if (MAX_ITEMS < N_ITEMSETS)
				{
					fprintf(stderr, "The number of items in Test set is higher than MAX_ITEMS.\n\n");
					return 1;
				}
			}
			//Only keep normal items, the ones that are not a class item
			if (CLASS_NAME.find(item) == CLASS_NAME.end())
				instance.insert(ITEM_MAP[item]);

			if (++countFeatures > MAX_FEATURES)
			{
				fprintf(stderr, "A document has more features than MAX_FEATURES parameters. Please change it in limits_h.cu.\n\n");
				return 1;
			}
		}

		TEST[N_TESTS].size = 0;
		
		//Fill the TEST array with the tests features values
		for (set<int>::iterator it = instance.begin(); it != instance.end(); it++)
			TEST[N_TESTS].instance[TEST[N_TESTS].size++] = *it;

		TEST[N_TESTS].label = target;//True class of the test document

		instance.clear();
		N_TESTS++;		

	}

	fprintf(stderr, "done\n%d itemsets.\n", N_ITEMSETS);

	fclose(file);

	return 0;
}
