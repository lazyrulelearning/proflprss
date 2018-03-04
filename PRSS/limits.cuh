#ifndef _LIMITS_H_
#define _LIMITS_H_

#define KB 1024

//Adjust here the number of classes that the dataset have.
#define NUM_CLASSES 2

//Only works up to 4, and a minimum of 2.
#define MAX_RULE_SIZE 4

//Maximum number of features/items that a document can have. THe closer to the current file, the better.
#define MAX_FEATURES 65 

//Maximum quantity of different features/items that the files can have. '10' is the number of bins used
//during discretization. If the file was discretized with a different number of bins, '10' should be changed.
#define MAX_ITEMS (MAX_FEATURES * 10 + 1) 

//Number of threads in a block
#define num_threads 256

//Adjust the number of lines for the training file. Closer to the actual number is better,
//since that gives smaller bitmaps, and more positions in the hash table.
#define MAX_LINE_TRAINING 1200 

//It will only work with a 64 bit integer.
#define TYPE  unsigned long long 
#define BITMAP_SLOT_SIZE (sizeof(TYPE) * 8)

//Calculating the maximum size of the bitmap 
#define MAX_BITMAP_SIZE  ( ( MAX_LINE_TRAINING / BITMAP_SLOT_SIZE ) + 1)

//The current bitmap size, according to the number of training documents
#define bitMAP_SIZE (N_TRAINING / BITMAP_SLOT_SIZE +1)


#endif
