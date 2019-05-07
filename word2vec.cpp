#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <vector>

#include "tensor.hpp"
#include "w2v_cbow_dataloader.hpp"
#include "unigram_table.hpp"

#include "averaged_embeddings.hpp"

using namespace fetch::ml;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;                    // Precision of float numbers

CBOWLoader<real> global_loader(5);
UnigramTable unigram_table(0);
 
struct vocab_word
{
  long long cn;
  unsigned int unique_id;
  char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;

unsigned long long vocab_max_size = 1000, layer1_size = 200;
unsigned long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

real alpha = 0.025, starting_alpha, sample = 1e-3;

std::vector<fetch::math::Tensor<real, 1>> syn0; // word vector
std::vector<fetch::math::Tensor<real, 1>> syn1neg; // Weights
real *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

fetch::ml::ops::AveragedEmbeddings<fetch::math::Tensor<float, 2>> embeddings_module(1, 1);

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
    syn0.emplace_back(fetch::math::Tensor<real, 1>({layer1_size}));

  while (syn1neg.size() < global_loader.VocabSize())
    syn1neg.emplace_back(fetch::math::Tensor<real, 1>({layer1_size}));

  for (auto &w : syn0)
    for (auto &e : w)
      e = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) / layer1_size;

  fetch::math::Tensor<real, 2> word_embeding_matrix({global_loader.VocabSize(), layer1_size});
  for (auto &e : word_embeding_matrix)
    e = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) / layer1_size;
  embeddings_module.SetData(word_embeding_matrix);
}

/**
 * ======== TrainModelThread ========
 * This function performs the training of the model.
 */
void *TrainModelThread(void *id)
{
  // Make a copy of the global loader for thread
  CBOWLoader<float> thread_loader(global_loader);
  thread_loader.SetOffset(thread_loader.Size() / (long long)num_threads * (long long)id);
  
  /*
   * word - Stores the index of a word in the vocab table.
   * word_count - Stores the total number of training words processed.
   */
  long long d, word;
  long long target, label;
  real f, g;

  fetch::math::Tensor<real, 2> neu1_unsqueezed({1, layer1_size});
  fetch::math::Tensor<real, 1> neu1 = neu1_unsqueezed.Slice(0);
  fetch::math::Tensor<real, 2> neu1e_unsqueezed({1, layer1_size});
  fetch::math::Tensor<real, 1> neu1e = neu1e_unsqueezed.Slice(0);

  auto sample = thread_loader.GetNext();
  std::vector<std::reference_wrapper<fetch::math::Tensor<real, 2> const>> inputs;
  inputs.push_back(std::cref(sample.first));
  
  unsigned int iterations = global_loader.Size() / num_threads;
  auto start = std::chrono::system_clock::now();
  unsigned int last_count(0);
  for (unsigned int i(0) ; i < iter * iterations ; ++i)
    {
      if (id == 0 && i % 10000 == 0)
	{
	  alpha = starting_alpha * (((float)iter * iterations - i) / (iter * iterations));
	  if (alpha < starting_alpha * 0.0001)
	    alpha = starting_alpha * 0.0001;
	  //	  std::cout << i << " / " << iter * iterations << " (" << (int)(100.0 * i / (iter * iterations)) << ") -- " << alpha << std::endl;
	}

      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end-start;    

      if (elapsed_seconds.count() > 1)
	{
	  std::cout << "Word / sec " << i - last_count << std::endl;
	  last_count = i;
	  start = std::chrono::system_clock::now();
	}
      
      
      if (thread_loader.IsDone())
	{
	  std::cout << id << " -- Reset" << std::endl;
	  thread_loader.Reset();
	}
      thread_loader.GetNext(sample);
      word = sample.second.Get(0, 0);
    
      //      neu1.Fill(0);
      neu1e.Fill(0);

      embeddings_module.Forward(inputs, neu1_unsqueezed);
      		
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
	      for (int fi(0) ; fi < layer1_size ; ++fi) // Dot Product
		f += syn1neg[target].Get(fi) * neu1.Get(fi);
	      
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
	      
	      neu1e.InlineAdd(syn1neg[target], g);
	      syn1neg[target].InlineAdd(neu1, g);
	    }
	}
      
      embeddings_module.Backward(inputs, neu1e_unsqueezed);
      embeddings_module.Step(alpha);
    }
  pthread_exit(NULL);
}

/**
 * ======== TrainModel ========
 * Main entry point to the training process.
 */
void TrainModel()
{
  long a;
  unsigned long long b;
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

  fetch::math::Tensor<real, 2> index({1, 1});
  std::vector<std::reference_wrapper<fetch::math::Tensor<real, 2> const>> in_index({std::cref(index)});
  fetch::math::Tensor<real, 2> vector({1, layer1_size}); 
  
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lu %lld\n", global_loader.VocabSize(), layer1_size);
    auto vocab = global_loader.GetVocab();
    for (auto kvp : vocab) // for (a = 0; a < vocab_size; a++)
      {
	fprintf(fo, "%s ", kvp.first.c_str()); //	fprintf(fo, "%s ", vocab[a].word);

	index.Set(0, 0, kvp.second.first);
	embeddings_module.Forward(in_index, vector);

	if (binary)
	  {
	    for (b = 0; b < layer1_size; b++)
	      {
		real v = vector.Get(0, b); // syn0[kvp.second.first].Get(b);
		fwrite(&v, sizeof(real), 1, fo);
	      }
	  }
	else
	  {
	    for (b = 0; b < layer1_size; b++)
	      {
		fprintf(fo, "%lf ", syn0[kvp.second.first].Get(b));
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
