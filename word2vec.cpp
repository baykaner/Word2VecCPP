#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <valarray>
#include <vector>

#include "tensor.hpp"
#include "w2v_cbow_dataloader.hpp"
#include "unigram_table.hpp"

using namespace fetch::ml;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;                    // Precision of float numbers

CBOWLoader<uint64_t> global_loader(5);
UnigramTable unigram_table(0);
 
struct vocab_word
{
  long long cn;
  unsigned int unique_id;
  char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

long long vocab_max_size = 1000, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

real alpha = 0.025, starting_alpha, sample = 1e-3;

std::vector<std::valarray<real>> syn0; // word vector
std::vector<std::valarray<real>> syn1neg; // Weights
real *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

std::string readFile(std::string const &path)
{
  std::ifstream t(path);
  return std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
}

void InitUnigramTable()
{
  std::vector<uint64_t> frequencies(global_loader.VocabSize());
  for (auto const &kvp : global_loader.GetVocab())
    {
      frequencies[kvp.second.first] = kvp.second.second;
    }
  unigram_table.Reset(table_size, frequencies);
}

void InitNet()
{
  while (syn0.size() < global_loader.VocabSize())
    syn0.emplace_back(std::valarray<real>(layer1_size));

  while (syn1neg.size() < global_loader.VocabSize())
    syn1neg.emplace_back(std::valarray<real>(layer1_size));

  for (auto &w : syn0)
    for (auto &e : w)
      e = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) / layer1_size;
}

/**
 * ======== TrainModelThread ========
 * This function performs the training of the model.
 */
void *TrainModelThread(void *id)
{
  // Make a copy of the global loader for thread
  CBOWLoader<uint64_t> thread_loader(global_loader);
  thread_loader.SetOffset(thread_loader.Size() / (long long)num_threads * (long long)id);
  
  /*
   * word - Stores the index of a word in the vocab table.
   * word_count - Stores the total number of training words processed.
   */
  long long a, b, d, cw, word, last_word;
  long long c, target, label;
  real f, g;
  
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  auto sample = thread_loader.GetNext();
  unsigned int iterations = global_loader.Size() / num_threads;
  for (unsigned int i(0) ; i < iter * iterations ; ++i)
    {
      if (i % 10000 == 0)
	{
	  if (id == 0)
	    {
	      alpha = starting_alpha * (((float)iter * iterations - i) / (iter * iterations));
	      if (alpha < starting_alpha * 0.0001)
		alpha = starting_alpha * 0.0001;	  
	      std::cout << i << " / " << iter * iterations << " (" << (int)(100.0 * i / (iter * iterations)) << ") -- " << alpha << std::endl;
	    }
	}
    
      if (thread_loader.IsDone())
	{
	  std::cout << id << " -- Reset" << std::endl;
	  thread_loader.Reset();
	}
      sample = thread_loader.GetNext(sample);
      
      word = sample.second.Get(0);
    
      for (c = 0; c < layer1_size; c++)
	neu1[c] = 0;
      for (c = 0; c < layer1_size; c++)
	neu1e[c] = 0;
      
      b = rand() % window;
    
      if (cbow)
	{
	  cw = 0;
	  for (a = 0 ; a < window * 2; a++)
	    {
	      last_word = sample.first.Get(a);
	      if (last_word >= 0)
		{
		  for (c = 0; c < layer1_size; c++)
		    neu1[c] += syn0[last_word][c];
		  cw++;
		}
	    }
      
	  if (cw)
	    {        
	      // neu1 was the sum of the context word vectors, and now becomes
	      // their average. 
	      for (c = 0; c < layer1_size; c++)
		neu1[c] /= cw;
                
	      // NEGATIVE SAMPLING
	      // Rather than performing backpropagation for every word in our 
	      // vocabulary, we only perform it for the positive sample and a few
	      // negative samples (the number of words is given by 'negative').
	      // These negative words are selected using a "unigram" distribution, 
	      // which is generated in the function InitUnigramTable.
	      if (negative > 0)
		{
		  for (d = 0; d < negative + 1; d++)
		    {
		      // On the first iteration, we're going to train the positive sample.
		      if (d == 0)
			{
			  target = word;
			  label = 1;
			}
		      else
			{
			  target = unigram_table.Sample();
			  if (target == word) continue;
			  label = 0;
			}
		      
		      f = 0;
		      for (c = 0; c < layer1_size; c++)
			f += neu1[c] * syn1neg[target][c];
		
		      if (f > MAX_EXP)
			{
			  g = (label - 1) * alpha;
			}
		      else if (f < -MAX_EXP)
			{
			  g = (label - 0) * alpha;
			}			
		      else
			{
			  g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
			}
		
		      for (c = 0; c < layer1_size; c++)
			neu1e[c] += g * syn1neg[target][c];
		
		      for (c = 0; c < layer1_size; c++)
			syn1neg[target][c] += g * neu1[c];
		    }
		}
	      
	      for (a = 0 ; a < window * 2 ; a++)
		{
		  last_word = sample.first.Get(a);
		  if (last_word >= 0)
		    for (c = 0; c < layer1_size; c++)
		      syn0[last_word][c] += neu1e[c];
		}
	    }
	} 
    }
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

/**
 * ======== TrainModel ========
 * Main entry point to the training process.
 */
void TrainModel()
{
  long a, b;
  FILE *fo;
  
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  
  printf("Starting training using file %s\n", train_file);
  
  starting_alpha = alpha;
    
  // Stop here if no output_file was specified.
  if (output_file[0] == 0) return;
  
  // Allocate the weight matrices and initialize them.
  InitNet();

  // If we're using negative sampling, initialize the unigram table, which
  // is used to pick words to use as "negative samples" (with more frequent
  // words being picked more often).  
  if (negative > 0) InitUnigramTable();
  
  // Record the start time of training.
  start = clock();
  
  // Run training, which occurs in the 'TrainModelThread' function.
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  
  
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lu %lld\n", global_loader.VocabSize(), layer1_size);
    auto vocab = global_loader.GetVocab();
    for (auto kvp : vocab) // for (a = 0; a < vocab_size; a++)
      {
	fprintf(fo, "%s ", kvp.first.c_str()); //	fprintf(fo, "%s ", vocab[a].word);
	if (binary)
	  {
	    for (b = 0; b < layer1_size; b++)
	      {
		fwrite(&syn0[kvp.second.first][b], sizeof(real), 1, fo);
	      }
	  }
	else
	  {
	    for (b = 0; b < layer1_size; b++)
	      {
		fprintf(fo, "%lf ", syn0[kvp.second.first][b]);
	      }
	  }
	fprintf(fo, "\n");
      }
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    {
      if (!strcmp(str, argv[a]))
	{
	  if (a == argc - 1)
	    {
	      printf("Argument missing for %s\n", str);
	      exit(1);
	    }
	  return a;
	}
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1)
    return 0;

  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);


  global_loader.AddData(readFile(train_file));
  global_loader.RemoveInfrequent(5);
  std::cout << "Dataloader Vocab Size : " << global_loader.VocabSize() << std::endl;
  
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  std::cout << "All done" << std::endl;
  return 0;
}
