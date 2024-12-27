export CC = g++

export CCFLAGS = -I./common -std=c++17 -Wall -pedantic -Wextra -Weffc++
export LDFLAGS = -pthread -lm -lstdc++ -L./common 

.PHONY: all

all:
	$(MAKE) -C common all
	# if compiled to share lib, we need --fPIC
	$(CC) client.cc $(CCFLAGS) $(LDFLAGS) -o client -lnet
	$(CC) server.cc $(CCFLAGS)  $(LDFLAGS) -o server -lnet

clean:
	$(MAKE) -C common clean
	-rm *.o client server
