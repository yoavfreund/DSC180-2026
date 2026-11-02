CC=cc
CFLAGS=-O2 -Wall
all: train predict
train: train.c common.h
	$(CC) $(CFLAGS) -o train train.c -lm
predict: predict.c common.h
	$(CC) $(CFLAGS) -o predict predict.c -lm
