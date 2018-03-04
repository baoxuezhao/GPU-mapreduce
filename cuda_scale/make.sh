nvcc -c -I../src InvertedIndex.cu -arch sm_20
mpic++ -o invertedindex InvertedIndex.o ../src/libmrmpi_linux.a -lmpich /usr/local/cuda/lib64/libcudart.so
