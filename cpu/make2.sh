mpic++ -c -I../src IntCount.cpp
mpic++ -o intcount IntCount.o ../src/libmrmpi_linux.a -lmpich
