CC=cc
CFLAGS=-O3 -march=native -std=c11 -Wall -Wextra
all: train predict
train: cascade.c
	$(CC) $(CFLAGS) -DTRAIN cascade.c -lm -o train
predict: cascade.c
	$(CC) $(CFLAGS) cascade.c -lm -o predict
clean:
	rm -f train predict
