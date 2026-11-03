CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wno-unused-function
all: train predict
train: train.c common.h feat.h
	$(CC) $(CFLAGS) -o train train.c -lm -lpthread
predict: predict.c common.h feat.h
	$(CC) $(CFLAGS) -o predict predict.c -lm
clean:
	rm -f train predict