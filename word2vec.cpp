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
#include "embeddings.hpp"
#include "matrix_multiply.hpp"
#include "inplace_transpose.hpp"
#include "placeholder.hpp"

#include "graph.hpp"

using namespace fetch::ml;
using namespace fetch::ml::ops;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;                    // Precision of float numbers

CBOWLoader<real> global_loader(5);
UnigramTable unigram_table(0);
 
char train_file[MAX_STRING], output_file[MAX_STRING];

int window = 5, min_count = 5, num_threads = 1, min_reduce = 1;

unsigned long long vocab_max_size = 1000, layer1_size = 200;
unsigned long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

real alpha = 0.025, starting_alpha, sample = 1e-3;

std::vector<fetch::math::Tensor<real, 1>> syn1neg; // Weights
real *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

fetch::ml::Graph<fetch::math::Tensor<real, 2>> graph;
fetch::ml::ops::AveragedEmbeddings<fetch::math::Tensor<float, 2>> word_vectors_embeddings_module(1, 1);
fetch::ml::ops::Embeddings<fetch::math::Tensor<float, 2>> word_weights_embeddings_module(1, 1);
fetch::ml::ops::MatrixMultiply<fetch::math::Tensor<float, 2>> dot_module;

// Keep out for easy saving
fetch::math::Tensor<real, 2> word_embeding_matrix({1, 1});

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
  fetch::math::Tensor<real, 2> weight_embeding_matrix({global_loader.VocabSize(), layer1_size});
  word_embeding_matrix = fetch::math::Tensor<real, 2>({global_loader.VocabSize(), layer1_size});
  for (auto &e : word_embeding_matrix)
    e = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) / layer1_size;
  word_vectors_embeddings_module.SetData(word_embeding_matrix);
  word_weights_embeddings_module.SetData(weight_embeding_matrix);


  graph.AddNode<PlaceHolder<fetch::math::Tensor<real, 2>, 2>>("Context", {});
  graph.AddNode<AveragedEmbeddings<fetch::math::Tensor<real, 2>>>("Words", {"Context"}, word_embeding_matrix); //global_loader.VocabSize(), layer1_size);
  graph.AddNode<PlaceHolder<fetch::math::Tensor<real, 2>, 2>>("Target", {});
  graph.AddNode<Embeddings<fetch::math::Tensor<real, 2>>>("Weights", {"Target"}, weight_embeding_matrix); // global_loader.VocabSize(), layer1_size);
  graph.AddNode<InplaceTranspose<fetch::math::Tensor<real, 2>>>("WeightsTranspose", {"Weights"});  
  graph.AddNode<MatrixMultiply<fetch::math::Tensor<real, 2>>>("DotProduct", {"Words", "WeightsTranspose"});

  
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


  fetch::math::Tensor<real, 2> f_tensor({1, 1}); // Prediction
  fetch::math::Tensor<real, 2> g_tensor({1, 1}); // Error
  fetch::math::Tensor<real, 2> label_tensor({1, 1}); // Ground truth
  
  auto sample = thread_loader.GetNext();
  
  unsigned int iterations = global_loader.Size() / num_threads;
  //  auto start = std::chrono::system_clock::now();
  unsigned int last_count(0);
  for (unsigned int i(0) ; i < iter * iterations ; ++i)
    {
      if (id == 0 && i % 10000 == 0)
	{
	  alpha = starting_alpha * (((float)iter * iterations - i) / (iter * iterations));
	  if (alpha < starting_alpha * 0.0001)
	    alpha = starting_alpha * 0.0001;
	  std::cout << i << " / " << iter * iterations << " (" << (int)(100.0 * i / (iter * iterations)) << ") -- " << alpha << std::endl;
	}

      if (thread_loader.IsDone())
	{
	  std::cout << id << " -- Reset" << std::endl;
	  thread_loader.Reset();
	}
      thread_loader.GetNext(sample);
      word = sample.second.Get(0, 0);
    
      graph.SetInput("Context", sample.first);      		
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
	      target = unigram_table.SampleNegative(target);
	      if (target == word) continue;
	      label = 0;
	    }
	  
	  f = 0;

	  label_tensor.Set(0, 0, target);
	  graph.SetInput("Target", label_tensor);

	  auto graphF = graph.Evaluate("DotProduct");

	  f = graphF.Get(0, 0);
	  
	  if (f > MAX_EXP)
	    {
	      g = (label - 1); //  * alpha; // alpha multiplication has moved to the step call
	    }
	  else if (f < -MAX_EXP)
	    {
	      g = (label - 0); //  * alpha; // alpha multiplication has moved to the step call
	    }
	  else
	    {
	      g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]); // * alpha;  // alpha multiplication has moved to the step call
	    }				      
	  
	  g_tensor.Set(0, 0, g);
	  graph.BackPropagate("DotProduct", g_tensor);
	}
      graph.Step(alpha);
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
  
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lu %lld\n", global_loader.VocabSize(), layer1_size);
    auto vocab = global_loader.GetVocab();
    for (auto kvp : vocab) // for (a = 0; a < vocab_size; a++)
      {
	fprintf(fo, "%s ", kvp.first.c_str()); //	fprintf(fo, "%s ", vocab[a].word);
	for (b = 0; b < layer1_size; b++)
	  {
	    real v = word_embeding_matrix.Get(kvp.second.first, b);
	    fwrite(&v, sizeof(real), 1, fo);
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
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  alpha = 0.05; // Initial learning rate
  
  global_loader.AddData(readFile(train_file));
  global_loader.RemoveInfrequent(min_count);
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
