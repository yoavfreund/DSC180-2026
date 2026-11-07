CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic

ifeq ($(OS),Windows_NT)
LIB = weak.dll
else
LIB = libweak.so
CFLAGS += -fPIC
endif

$(LIB): weak.c weak.h evaluate.c image_features.c image_features.h
	$(CC) $(CFLAGS) -shared -o $@ weak.c evaluate.c image_features.c -lm

.PHONY: clean test
test: tests/test_features.c image_features.c evaluate.c weak.h image_features.h
	$(CC) $(CFLAGS) -I. -o test_features tests/test_features.c image_features.c evaluate.c -lm
	./test_features
clean:
	$(RM) $(LIB) test_features test_features.exe
