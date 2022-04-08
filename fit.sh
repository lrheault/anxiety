#!/bin/bash

VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=100
WINDOW_SIZE=15
CORPUS=lipad8015_lemmas.csv
VFILE=lipad-vocab.txt
CFILE=lipad-cooccurrences.bin
CSFILE=lipad-cooccurrences.shuf.bin
OUTFILE=lipad-vectors-300d

./GloVe-1.2/build/vocab_count -min-count $VOCAB_MIN_COUNT < $CORPUS > $VFILE
./GloVe-1.2/build/cooccur -vocab-file $VFILE -memory 16.0 -window-size $WINDOW_SIZE -overflow-file tempoverflow < $CORPUS > $CFILE
./GloVe-1.2/build/shuffle -memory 16.0 < $CFILE > $CSFILE
./GloVe-1.2/build/glove -input-file $CSFILE -vocab-file $VFILE -save-file $OUTFILE -vector-size $VECTOR_SIZE -threads 6 -iter $MAX_ITER -binary 0
