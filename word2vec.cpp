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

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

typedef float real;                    // Precision of float numbers

CBOWLoader<real> global_loader(5, 25);
UnigramTable unigram_table(0);
 
std::string train_file, output_file;

int window = 5, min_count = 5;

unsigned long long layer1_size = 200, iter = 1;
real alpha = 0.025, starting_alpha;

real *expTable;
clock_t start;

int negative = 25;

fetch::ml::Graph<fetch::math::Tensor<real, 2>> graph;
// Add a sigmoid module here -- Not done for now as it would require bringing all the math library


// Keep out for easy saving
fetch::math::Tensor<real, 2> word_embeding_matrix({1, 1});

std::string readFile(std::string const &path)
{
  std::ifstream t(path);
  return std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
}

void InitNet()
{
  word_embeding_matrix = fetch::math::Tensor<real, 2>({global_loader.VocabSize(), layer1_size});
  for (auto &e : word_embeding_matrix)
    e = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) / layer1_size;

  graph.AddNode<PlaceHolder<fetch::math::Tensor<real, 2>, 2>>("Context", {});
  graph.AddNode<AveragedEmbeddings<fetch::math::Tensor<real, 2>>>("Words", {"Context"}, word_embeding_matrix);
  graph.AddNode<PlaceHolder<fetch::math::Tensor<real, 2>, 2>>("Target", {});
  graph.AddNode<Embeddings<fetch::math::Tensor<real, 2>>>("Weights", {"Target"}, global_loader.VocabSize(), layer1_size);
  graph.AddNode<InplaceTranspose<fetch::math::Tensor<real, 2>>>("WeightsTranspose", {"Weights"});  
  graph.AddNode<MatrixMultiply<fetch::math::Tensor<real, 2>>>("DotProduct", {"Words", "WeightsTranspose"});  
}

/**
 * ======== TrainModelThread ========
 * This function performs the training of the model.
 */
void TrainModelThread()
{
  // Make a copy of the global loader for thread
  CBOWLoader<float> thread_loader(global_loader);
  
  /*
   * word - Stores the index of a word in the vocab table.
   * word_count - Stores the total number of training words processed.
   */
  real f;


  fetch::math::Tensor<real, 2> f_tensor({1, uint64_t(negative)}); // Prediction
  fetch::math::Tensor<real, 2> g_tensor({1, uint64_t(negative)}); // Error
  fetch::math::Tensor<real, 2> label_tensor({1, uint64_t(negative)}); // Target Word input
  
  auto sample = thread_loader.GetNext();
  
  unsigned int iterations = global_loader.Size();
  for (unsigned int i(0) ; i < iter * iterations ; ++i)
    {
      if (i % 10000 == 0)
	{
	  alpha = starting_alpha * (((float)iter * iterations - i) / (iter * iterations));
	  if (alpha < starting_alpha * 0.0001)
	    alpha = starting_alpha * 0.0001;
	  std::cout << i << " / " << iter * iterations << " (" << (int)(100.0 * i / (iter * iterations)) << ") -- " << alpha << std::endl;
	}

      if (thread_loader.IsDone())
	{
	  thread_loader.Reset();
	}

      thread_loader.GetNext(sample);
      graph.SetInput("Context", sample.first);
      graph.SetInput("Target", sample.second);
      auto graphF = graph.Evaluate("DotProduct");


      // This block computes sigmoid activation + MSE and store error signal in g_tensor 
      for (int d = 0; d < negative ; d++)
	{
      	  f = graphF.Get(0, d);
	  float label = (d == 0) ? 1 : 0;
	  if (f > MAX_EXP)
	    {
	      g_tensor.Set(0, d, label - 1);
	    }
	  else if (f < -MAX_EXP)
	    {
	      g_tensor.Set(0, d, label - 0);
	    }
	  else
	    {
	      g_tensor.Set(0, d, label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);
	    }				      	  
	}
      graph.BackPropagate("DotProduct", g_tensor);
      graph.Step(alpha);
    }

  std::cout << "Done" << std::endl;
}

/**
 * ======== TrainModel ========
 * Main entry point to the training process.
 */
void TrainModel()
{
  unsigned long long b;
  FILE *fo;
  
  starting_alpha = alpha;
    
  if (output_file.empty())
    return;

  InitNet();
  TrainModelThread();
  
  fo = fopen(output_file.c_str(), "wb");
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
  fclose(fo);
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1)
    return 0;
  
  output_file = "./vector.bin";
  train_file = argv[1];

  alpha = 0.05; // Initial learning rate
  
  global_loader.AddData(readFile(train_file));
  global_loader.RemoveInfrequent(min_count);
  global_loader.InitUnigramTable();
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
