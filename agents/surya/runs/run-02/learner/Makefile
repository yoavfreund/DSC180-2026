CC=gcc
CFLAGS=-O3 -std=c11 -Wall -Wextra
all: train predict
train: cascade.c
	$(CC) $(CFLAGS) -o $@ $< -lm
predict: train
	cp train predict
clean:
	rm -f train predict
