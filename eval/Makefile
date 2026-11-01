CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Werror

eval: eval.c
	$(CC) $(CFLAGS) -o $@ eval.c -lm

test: eval
	./tests/run_tests.sh

clean:
	rm -f eval

.PHONY: test clean
