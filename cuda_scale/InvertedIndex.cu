// MapReduce Inverted Index example using CUDA
// Syntax: invertedindex path-of-data-dir
// (1) assume each host has four processors, each corresponds
//     to a GPU, and read one parts of the files in the local dir
// (2) parse into words separated by whitespace
// (3) count occurrence of each word in all files
// (4) print top 10 words

#include <mpi.h>
#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstring>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "mapreduce.h"
#include "keyvalue.h"

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>

#define CEIL(n,m) ((n)/(m) + (int)((n)%(m) !=0))
#define THREAD_CONF(grid, block, gridBound, blockBound) do {\
	    block.x = blockBound;\
	    grid.x = gridBound; \
		if (grid.x > 65535) {\
		   grid.x = (int)sqrt((double)grid.x);\
		   grid.y = CEIL(gridBound, grid.x); \
		}\
	}while (0)

using namespace MAPREDUCE_NS;
using namespace std;

void mymap(int , KeyValue *, void *);
void myreduce(char *, int, char *, int, int *, KeyValue *, void *);
void mycombine(char *, int, char *, int, int *, KeyValue *, void *);

char inputdir[100];
int  num_file = 1;
int me, nprocs;

//int ncompare(char *, int, char *, int);
//void output(uint64_t, char *, int, char *, int, KeyValue *, void *);

struct Info
{
	int me;
	int nproc;
};

#define START		0x00
#define IN_TAG		0x01
#define IN_ATAG		0x02
#define FOUND_HREF	0x03
#define START_LINK	0x04

struct is_start
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x==1;
	}
};

__global__ void mark(
		char 	*text,
		int 	*d_segmask,
		int 	length)
{
	const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

	const int tid = tid_y * (blockDim.x*gridDim.x) + tid_x;

	if(tid < length)
		d_segmask[tid] = 0;

	if(tid >= length-9)
		return;

	if(text[tid] == '<' &&
			text[tid+1] == 'a' &&
			text[tid+2] == ' ' &&
			text[tid+3] == 'h' &&
			text[tid+4] == 'r' &&
			text[tid+5] == 'e' &&
			text[tid+6] == 'f' &&
			text[tid+7] == '=' &&
			text[tid+8] == '\"')
	{
		d_segmask[tid+9] = 1;
	}
}

__global__ void compute_url_length(
		char 	*d_text,
		int		*d_urloffset,
		int		*d_urllength,
		int 	textlen,
		int		url_num)
{
	const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

	const int tid = tid_y * (blockDim.x*gridDim.x) + tid_x;

	if(tid >= url_num)
		return;

	int start = d_urloffset[tid];

	for(int i=start; i < textlen; i++)
	{
		if(d_text[i] == '\"' || i == textlen-1)
		{
			d_urllength[tid] = i-start;
			d_text[i] = '\0';
			return;
		}
	}
}

/* ---------------------------------------------------------------------- */

//parameters: 
int main(int argc, char **args)
{
	MPI_Init(&argc,&args);

	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	
	if (argc <= 2)
	{
		if (me == 0) printf("Syntax: invertedindex [input_dir num_file]...\n");
		MPI_Abort(MPI_COMM_WORLD,1);
	}
	
	strcpy(inputdir, args[1]);
	num_file = atoi(args[2]);

	MapReduce *mr = new MapReduce(MPI_COMM_WORLD);
	mr->verbosity = 2;
	mr->timer = 1;

	if(NULL==opendir("/mnt/mrmpi/temp"))
	{
		system("mkdir /mnt/mrmpi");
		system("mkdir /mnt/mrmpi/temp");	
	}
	mr->set_fpath("/mnt/mrmpi/temp");
	mr->memsize = 64;
	//mr->outofcore = 1;

	MPI_Barrier(MPI_COMM_WORLD);
	double tstart = MPI_Wtime();

	//printf("start map %d\n", me);

	int mapitem = mr->map(nprocs, mymap, &me);
	//int nfiles = mr->mapfilecount;

	//mr->compress(mycombine, NULL);

	//printf("start aggregate %d\n", me);

	mr->aggregate(NULL);

	//printf("end aggregate %d\n", me);

	mr->convert();
	//mr->collate(NULL);

	//printf("end convert %d\n", me);


	if(NULL==opendir("/mnt/mrmpi_output"))
	{
		system("mkdir /mnt/mrmpi_output");
	}
	system("rm /mnt/mrmpi_output/InvertedIndex*");

	Info info;
	info.me = me;
	info.nproc = nprocs;

	int reduceitem = mr->reduce(myreduce, &info);

	MPI_Barrier(MPI_COMM_WORLD);
	double tstop = MPI_Wtime();

	/*
	mr->sort_values(&ncompare);

	Count count;
	count.n = 0;
	count.limit = 10;
	count.flag = 0;
	mr->map(mr,output,&count);

	mr->gather(1);
	mr->sort_values(ncompare);

	count.n = 0;
	count.limit = 10;
	count.flag = 1;
	mr->map(mr,output,&count);
	*/

	delete mr;

	//printf("map and reduce item are %d, %d\n", mapitem, reduceitem);

	if (me == 0)
	{
		//printf("%d total words, %d unique words\n",nwords,nunique);
		printf("Time to process on %d procs = %g (secs), %d, %d\n", nprocs, tstop-tstart, mapitem, reduceitem);
	}

	MPI_Finalize();
}


int getfilename(char *fullpath, char *filename)
{
	size_t found;
	std::string path(fullpath);
	found=path.find_last_of("/\\");
	const char *name = path.substr(found+1).c_str();
	memcpy(filename, name, strlen(name)+1);

	return strlen(name);
}

/* ----------------------------------------------------------------------
   read a file
   for each word in file, emit key = word, value = NULL
------------------------------------------------------------------------- */
void mymap(int nmap, KeyValue *kv, void *ptr)
{
	int me = *(int*)ptr;
	cudaSetDevice(0);

	struct timeval	start_map, end_map;
	double time_map = 0.0;

	cudaDeviceSynchronize();
	gettimeofday(&start_map, NULL);

	int resultlen;
	char hostname[20];
	MPI_Get_processor_name(hostname, &resultlen);
	int host_id = -1;
	
	if(strcmp(hostname, "master\0")==0)
		host_id = 0;
	else
	{
		sscanf(hostname, "node%d", &host_id);
		//host_id -= 1;	
	}

	int file_each_proc = num_file/nprocs;

	for(int fid=me*file_each_proc; fid<(me+1)*file_each_proc && fid < num_file; fid++)
	{
		char fullname[100];
		sprintf(fullname, "%s/part-%05d\0", inputdir, fid);

		printf("full file name and gpu id is %s, %d\n", fullname, me%4);

		// filesize = # of bytes in file
		struct stat stbuf;
		int flag = stat(fullname,&stbuf);
		if (flag < 0) {
			printf("ERROR: Could not query file size %d, %s\n", me, fullname);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		int filesize = stbuf.st_size;

		FILE *fp = fopen(fullname,"r");
		char *text = new char[filesize+1];
		int nchar = fread(text,1,filesize,fp);
		text[nchar] = '\0';
		fclose(fp);

		char filename[100];
		int namelen = getfilename(fullname, filename);

		//copy text data into gpu memory
		char *d_text;
		cudaMalloc((void**)&d_text, (filesize+1)*sizeof(char));
		cudaMemcpy(d_text, text, (filesize+1)*sizeof(char), cudaMemcpyHostToDevice);

		//record the start position of each url
		int *d_sequence;
		int *d_segmask;

		cudaMalloc((void**)&d_sequence, (filesize+1)*sizeof(int));
		cudaMalloc((void**)&d_segmask, (filesize+1)*sizeof(int));

		thrust::device_ptr<int> dev_sequence(d_sequence);
		thrust::device_ptr<int> dev_segmask(d_segmask);

		thrust::sequence(dev_sequence, dev_sequence+(filesize+1));

		dim3 h_dimBlock(256,1,1);
		dim3 h_dimGrid(1,1,1);
		int numBlocks = CEIL(filesize+1, h_dimBlock.x);
		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

		//cudaEvent_t start, stop;
		//float time1;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start, 0);

		//record the position array (about 4ms for 64M)
		mark<<<h_dimGrid, h_dimBlock>>>(d_text, d_segmask, (filesize+1));

		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&time1, start, stop);
		//printf("time is %f\n", time1);

		//printf("zhao2 %d\n", me);

		int urlcount =	thrust::count(dev_segmask, dev_segmask+(filesize+1), 1);

		if(urlcount == 0)
			return;

		int *d_urloffset;
		int *d_urllength;

		cudaMalloc((void**)&d_urloffset, urlcount*sizeof(int));
		cudaMalloc((void**)&d_urllength, urlcount*sizeof(int));

		thrust::device_ptr<int> dev_urloffset(d_urloffset);

		//about 14ms
		thrust::copy_if(dev_sequence, dev_sequence+(filesize+1),
				dev_segmask, dev_urloffset, is_start());

		dim3 h_dimGrid2(1,1,1);
		dim3 h_dimBlock2(256,1,1);
		numBlocks = CEIL(urlcount, h_dimBlock2.x);
		THREAD_CONF(h_dimGrid2, h_dimBlock2, numBlocks, h_dimBlock2.x);

		//about 8ms
		compute_url_length<<<h_dimGrid2, h_dimBlock2>>>(
				d_text,
				d_urloffset,
				d_urllength,
				(filesize+1),
				urlcount);

		int *h_urloffset = new int[urlcount];
		int *h_urllength = new int[urlcount];

		cudaMemcpy(text, d_text, (filesize+1)*sizeof(char), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_urloffset, d_urloffset, urlcount*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_urllength, d_urllength, urlcount*sizeof(int), cudaMemcpyDeviceToHost);

		//about 18ms for 64m ii data
		for(int i=0; i<urlcount; i++)
		{
			kv->add(text+h_urloffset[i], h_urllength[i]+1, filename, namelen+1);
		}

		//free device memory
		cudaFree(d_text);
		cudaFree(d_sequence);
		cudaFree(d_segmask);
		cudaFree(d_urloffset);
		cudaFree(d_urllength);

		delete [] text;
		delete [] h_urloffset;
		delete [] h_urllength;
	}
	
	//printf("end of map %d\n", me);

	cudaDeviceSynchronize();
	gettimeofday(&end_map, NULL);
	time_map += (1000*(end_map.tv_sec-start_map.tv_sec)
			+(end_map.tv_usec-start_map.tv_usec + 0.0)/1000);
	printf("time of %d is %f\n", me, time_map);

}


void mycombine(char *key, int keybytes, char *multivalue,
		int nvalues, int *valuebytes, KeyValue *kv, void *ptr)
{

	stringstream ss (stringstream::in | stringstream::out);

	int t = 0;
	if(nvalues)
	{
		char* curval = multivalue;
		for(int i=0; i<nvalues; i++)
		{
			if(t!=0)
				ss << " ";
			ss << curval;
			curval += valuebytes[i];
			t++;
		}
	}
	else
	{
		MapReduce *mr = (MapReduce *) valuebytes;
		int nblocks;
		uint64_t nvalues_total = mr->multivalue_blocks(nblocks);
		for (int iblock = 0; iblock < nblocks; iblock++)
		{
			int nv = mr->multivalue_block(iblock,&multivalue,&valuebytes);

			char* curval = multivalue;
			for (int i = 0; i < nv; i++)
			{
				if(t!=0)
					ss << " ";
				ss << curval;
				curval += valuebytes[i];
				t++;
				//process each value within the block of values
			}
		}
	}

	string s = ss.str();
	kv->add(key, keybytes, (char*)s.c_str(), (int)(s.length()+1));
}

/* ----------------------------------------------------------------------
   count word occurrence
   emit key = word, value = # of multi-values
------------------------------------------------------------------------- */

void myreduce(char *key, int keybytes, char *multivalue,
		int nvalues, int *valuebytes, KeyValue *kv, void *ptr)
{

	Info *info = (Info*) ptr;
	int me = info->me;
	int nproc = info->nproc;

	char filename[50];
	sprintf(filename, "/mnt/mrmpi_output/InvertedIndex-%d-%d\0", nproc , me);

	//printf("filename is %s, %d\n", filename, nvalues);

	std::fstream filestr;
	filestr.open (filename, fstream::out | fstream::app);

	filestr << key << "\t";

	if(nvalues)
	{
		char* curval = multivalue;
		for(int i=0; i<nvalues; i++)
		{
			filestr <<  curval << " ";
			curval += valuebytes[i];
		}
		filestr << endl;
	}
	else
	{
		MapReduce *mr = (MapReduce *) valuebytes;
		int nblocks;
		uint64_t nvalues_total = mr->multivalue_blocks(nblocks);
		for (int iblock = 0; iblock < nblocks; iblock++)
		{
			int nv = mr->multivalue_block(iblock,&multivalue,&valuebytes);

			char* curval = multivalue;
			for (int i = 0; i < nv; i++)
			{
				filestr <<  curval << " ";
				curval += valuebytes[i];
				//process each value within the block of values
			}
		}

		filestr << endl;
	}

	filestr.close();
}

/* ----------------------------------------------------------------------
   compare two counts
   order values by count, largest first
------------------------------------------------------------------------- */

int ncompare(char *p1, int len1, char *p2, int len2)
{
	int i1 = *(int *) p1;
	int i2 = *(int *) p2;
	if (i1 > i2) return -1;
	else if (i1 < i2) return 1;
	else return 0;
}

/* ----------------------------------------------------------------------
   process a word and its count
   depending on flag, emit KV or print it, up to limit
------------------------------------------------------------------------- */

void output(uint64_t itask, char *key, int keybytes, char *value,
		int valuebytes, KeyValue *kv, void *ptr)
{
	/*
	Count *count = (Count *) ptr;
	count->n++;
	if (count->n > count->limit) return;

	int n = *(int *) value;
	if (count->flag)
		printf("%d %s\n",n,key);
	else
		kv->add(key,keybytes,(char *) &n,sizeof(int));
	*/
}

