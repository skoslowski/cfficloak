CC = gcc
CFLAGS = -pedantic -Wall -Wextra -march=native -g -fPIC
LDFLAGS = -shared
LIBS = -lm

TARGET = libtest.so
SOURCES = test.c
HEADERS = test.h
OBJECTS = test.o

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) ${LDFLAGS} -o $@ $^ $(LIBS)

$(SOURCES:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@ $(LIBS)

include $(SOURCES:.c=.d)

.PHONY: clean
clean:
	-rm -rf ${TARGET} ${OBJECTS} $(SOURCES:.c=.d) __pycache__ *.pyc
