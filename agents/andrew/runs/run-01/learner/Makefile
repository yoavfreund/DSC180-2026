CC ?= cc
CFLAGS = -O3 -std=c99 -Wall
all: train predict
train: train.c common.h
	$(CC) $(CFLAGS) -o train train.c -lm -lpthread
predict: predict.c common.h
	$(CC) $(CFLAGS) -o predict predict.c -lm
clean:
	rm -f train predict
