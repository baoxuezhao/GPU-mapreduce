mpic++ -c -I../src InvertedIndex.cpp
mpic++ -o invertedindex InvertedIndex.o ../src/libmrmpi_linux.a -lmpich
