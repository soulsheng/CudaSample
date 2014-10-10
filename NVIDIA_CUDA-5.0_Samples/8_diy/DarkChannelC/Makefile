
objects = DarkChannel.o bmpHandler.o main.o

DarkChannel : $(objects)
	g++ -o DarkChannel $(objects)

DarkChannel.o : DarkChannel.cpp
	g++ -c DarkChannel.cpp
	
bmpHandler.o : bmpHandler.cpp
	g++ -c bmpHandler.cpp
	
main.o : main.cpp
	g++	-c main.cpp
	
.PHONY : clean
clean :
	rm DarkChannel $(objects)
	