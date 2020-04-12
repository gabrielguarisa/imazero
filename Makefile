.PHONY: imazero/imazero.so
imazero/imazero.so:
	g++ -fPIC -shared wrapper/imazero.cc -o imazero/imazero.so -Wall -O3 -std=c++2a

all: imazero/imazero.so
