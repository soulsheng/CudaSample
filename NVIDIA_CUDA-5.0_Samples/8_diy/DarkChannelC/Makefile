
objects = DarkChannel.o bmpHandler.o main.o
CUDA_PATH       ?= /usr/local/cuda-6.0

CC = g++
LDFLAGS :=
LDFLAGS += $(CC)
CCFLAGS := 
CCFLAGS += $(LDFLAGS)
CCFLAGS += -c
CCFLAGS += -I$(CUDA_PATH)/include
ifeq ($(dbg),1)
CCFLAGS += -g
endif

DarkChannel : $(objects)
	$(LDFLAGS) -o DarkChannel $(objects)

DarkChannel.o : DarkChannel.cpp
	$(CCFLAGS) DarkChannel.cpp
	
bmpHandler.o : bmpHandler.cpp
	$(CCFLAGS) bmpHandler.cpp
	
main.o : main.cpp
	$(CCFLAGS) main.cpp
	
.PHONY : clean
clean :
	rm DarkChannel $(objects)
	
