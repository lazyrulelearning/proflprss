/**
THIS MODULE IMPLEMENTS THE EVALUATION METRICS SUCH AS
PRECISION, RECALL AND ACCURACY.
*/

int initialize_evaluation(evaluation_t* evaluator);
int update_evaluation(evaluation_t *evaluator, int predicted_label, int true_label);
void finalize_evaluation(evaluation_t *evaluator);


int initialize_evaluation(evaluation_t* evaluator) {

	(*evaluator).total_predictions = (int*)calloc(NUM_CLASSES, sizeof(int));
	(*evaluator).correct_predictions = (int*)calloc(NUM_CLASSES, sizeof(int));
	(*evaluator).true_labels = (int*)calloc(NUM_CLASSES, sizeof(int));
	(*evaluator).precision = (float*)calloc(NUM_CLASSES, sizeof(float));
	(*evaluator).recall = (float*)calloc(NUM_CLASSES, sizeof(float));
	(*evaluator).f1 = (float*)calloc(NUM_CLASSES, sizeof(float));
	
	return 0;
}

/**
UPDATES THE EVALUATOR AFTER THE PROGRAM PERFORMS A PREDICTION.
*/
int update_evaluation(evaluation_t *evaluator, int predicted_label, int true_label) {
	
	(*evaluator).total_predictions[predicted_label]++;
	(*evaluator).true_labels[true_label]++;

	if (predicted_label == true_label)
		(*evaluator).correct_predictions[predicted_label]++;

	int total1 = 0, total2 = 0;

	for (int i = 0; i < NUM_CLASSES; i++) {
		(*evaluator).precision[i] = (*evaluator).correct_predictions[i] / (float)(*evaluator).total_predictions[i];
		(*evaluator).recall[i] = (*evaluator).correct_predictions[i] / (float)(*evaluator).true_labels[i];
		(*evaluator).f1[i] = (2 * (*evaluator).precision[i] * (*evaluator).recall[i]) / (float)((*evaluator).precision[i] + (*evaluator).recall[i]);
		total1 += (*evaluator).correct_predictions[i];
		total2 += (*evaluator).true_labels[i];
	}
	(*evaluator).acc = total1 / (float)total2;
	(*evaluator).mf1 = 0;
	int k = 0;
	//Average of the F1 measure
	for (int i = 0; i < NUM_CLASSES; i++) {
		if (!_isnan((*evaluator).f1[i])) {
			(*evaluator).mf1 += (*evaluator).f1[i];
			k++;
		}
	}
	(*evaluator).mf1 /= (float)k;

	return 0;
}

/**
FINALIZE THE EVALUATOR AND RELEASE THE MEMORY.
*/
void finalize_evaluation(evaluation_t *evaluator) {
	
	free((*evaluator).total_predictions);
	free((*evaluator).correct_predictions);
	free((*evaluator).true_labels);
	free((*evaluator).precision);
	free((*evaluator).recall);
	free((*evaluator).f1);

}


/**
RETURNS THE SCORE ASSOCIATED WITH EACH CLASS, ACCORDING TO A CERTAIN CRITERION.
*/
__device__ score_t get_total_score(float* finalpoints, int* gn_rules, int n_tests, int idx) {

	score_t score;
	float t_points = 0;

	#pragma unroll
	for (int i = 0; i < NUM_CLASSES; i++) {
		score.ordered_labels[i] = i;
		score.points[i] = 0;
	}
	//Average the confidence values of the rules that belongs to each class.
	#pragma unroll
	for (int i = 0; i < NUM_CLASSES; i++){
		finalpoints[idx + i * n_tests] /= gn_rules[idx + i * n_tests];
	}
	#pragma unroll
	for (int i = 0; i < NUM_CLASSES; i++) {
		score.points[i] += finalpoints[idx + i * n_tests];
		t_points += finalpoints[idx + i * n_tests];
	}

	if (t_points > 0) {
		//Normalize the scores of each class with the sum of all classes' scores.
		#pragma unroll
		for (int i = 0; i < NUM_CLASSES; i++)
			score.points[i] = score.points[i] / (float)t_points;
	}
	float points[NUM_CLASSES];
	#pragma unroll
	for (int i = 0; i < NUM_CLASSES; i++)
		points[i] = score.points[i];

	//Sort classes by largest score
	for (int i = 0; i < NUM_CLASSES; i++) {
		int largest = i;
		float p = -1;
		for (int j = 0; j < NUM_CLASSES; j++) {
			if (points[j] > p) {
				largest = j;
				p = points[j];
			}
		}
		score.ordered_labels[i] = largest;
		points[largest] = -1;
	}

	return score;
}

/**
PRINTS ALL STATISTICS ASSOCIATED WITH A PREDICTION.
*/
void print_statistics(int n_tests, int true_label, int id, prediction_t prediction, evaluation_t evaluator, stringstream& stream)
{
	float ranking = 0, entropy = 0; 

	for (int i = 0; i < NUM_CLASSES; i++)
		ranking += i * prediction.score.points[i];

	for (int i = 0; i < NUM_CLASSES; i++)
		entropy -= prediction.score.points[i] * log2(prediction.score.points[i]);

	stream << "id= " << id << " label= " << true_label << " prediction= " << prediction.label;

	stream << " ranking= " << ranking << " entropy= " << entropy << " rules= " << prediction.rules << endl;	
}
