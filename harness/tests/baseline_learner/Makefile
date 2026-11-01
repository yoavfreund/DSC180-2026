CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall

all: train predict

train: train.c pgm.h
	$(CC) $(CFLAGS) -o $@ train.c -lm

predict: predict.c pgm.h
	$(CC) $(CFLAGS) -o $@ predict.c -lm
