rm -f vectors.bin word2vec
make && time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 20 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 1 | ts
./distance vectors.bin
