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

//It will only work with a 32 bit integer.
#define TYPE  unsigned int
#define BITMAP_SLOT_SIZE (sizeof(TYPE) * 8)

//Calculating the maximum size of the bitmap by taking into account the vectorized int4 type.
#define DUMMY  ((MAX_LINE_TRAINING / BITMAP_SLOT_SIZE ) + 1)
#define MAX_BITMAP_SIZE  (((DUMMY / 4) + 1) * 4)

//Adjust the memory for the hash table here. It should be around 80% of the max available memory,
//so that there is enough memory for the other structures.
#define GPU_MEM 4500ULL*KB*KB 

//Number of positions in the hash table
#define HASH_SIZE  (GPU_MEM / sizeof(content_t) + 1)

//The current bitmap size, according to the number of training documents
#define bitMAP_SIZE ((int)ceil(N_TRAINING / (float)(BITMAP_SLOT_SIZE) ))


#endif
