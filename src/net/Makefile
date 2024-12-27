SOURCES = $(wildcard ./*.cc)
OBJECTS = $(SOURCES:.cc=.o)
SOURCES += $(wildcard ./*.hh)

CCFLAGS += -fPIC 
# LDFLAGS += -lstdc++

TARGET = libnet.a
# %.o: %.cc
# 	$(CC) $(CCFLAGS) $<

all:
	$(CC) $(SOURCES) $(CCFLAGS) $(LDFLAGS) -shared -o $(TARGET) 

clean:
	-rm -f *.o
	-rm -f $(TARGET)