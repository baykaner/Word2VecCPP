rm -f vectors.bin word2vec
make && time ./word2vec text8 | ts
./distance vectors.bin
