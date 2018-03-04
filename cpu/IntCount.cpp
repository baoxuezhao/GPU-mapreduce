// MapReduce Inverted Index example using CUDA
// Syntax: invertedindex path-of-data-dir
// (1) assume each host has four processors, each corresponds
//     to a GPU, and read one parts of the files in the local dir
// (2) parse into words separated by whitespace
// (3) count occurrence of each word in all files
// (4) print top 10 words

#include "/usr/local/mpich2-1.5/include/mpi.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "sys/stat.h"
#include "mapreduce.h"
#include "keyvalue.h"

using namespace std;

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

void mymap(int , KeyValue *, void *);
void myreduce(char *, int, char *, int, int *, KeyValue *, void *);
void mycombine(char *, int, char *, int, int *, KeyValue *, void *);
//int ncompare(char *, int, char *, int);
//void output(uint64_t, char *, int, char *, int, KeyValue *, void *);

struct Info
{
	int me;
	int nproc;
};

/* ---------------------------------------------------------------------- */

int main(int argc, char **args)
{
	MPI_Init(&argc,&args);

	int me, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/*
	if (argc <= 1)
	{
		if (me == 0) printf("Syntax: invertedindex dir ...\n");
		MPI_Abort(MPI_COMM_WORLD,1);
	}*/

	MapReduce *mr = new MapReduce(MPI_COMM_WORLD);
	mr->verbosity = 2;
	mr->timer = 1;
	mr->all2all = 0;
	mr->set_fpath("/data/bzhaoad/mrmpi/temp");
	mr->memsize = 128;
	//mr->outofcore = 1;

	MPI_Barrier(MPI_COMM_WORLD);
	double tstart = MPI_Wtime();

	int mapitem = mr->map(nprocs, mymap, &me);
	//int nfiles = mr->mapfilecount;

	//mr->compress(mycombine, NULL);

	mr->aggregate(NULL);
	mr->convert();
	//mr->collate(NULL);

	//system("mkdir /data/bzhaoad/mrmpi_output");
	//system("mkdir /data/bzhaoad/mrmpi_output/InvertedIndex");
	//system("rm /data/bzhaoad/mrmpi_output/InvertedIndex/InvertedIndex*");

	Info info;
	info.me = me;
	info.nproc = nprocs;
	int reduceitem;// = mr->reduce(myreduce, &info);

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

#define START		0x00
#define IN_TAG		0x01
#define IN_ATAG		0x02
#define FOUND_HREF	0x03
#define START_LINK	0x04

void mymap(int nmap, KeyValue *kv, void *ptr)
{

	int me = *(int*)ptr;

	struct timeval	start_map, end_map;
	double time_map = 0.0;
	gettimeofday(&start_map, NULL);

	char fullname[100];
	sprintf(fullname, "/data/bzhaoad/intcount_data");

	struct stat stbuf;
	int flag = stat(fullname,&stbuf);
	if (flag < 0) {
		printf("ERROR: Could not query file size\n");
		MPI_Abort(MPI_COMM_WORLD,1);
	}

	int filesize = /*stbuf.st_size;*/ 128*1024*1024;

	FILE *fp = fopen(fullname,"r");
	char *text = new char[filesize+1];
	int nchar = fread(text,1,filesize,fp);
	text[nchar] = '\0';
	fclose(fp);

	int t = 1;

	for(int i=0; i<nchar/4; i++)
		kv->add(text+i*4, 4, (char*)(&t), 4);

	delete [] text;

	gettimeofday(&end_map, NULL);
	time_map = (1000*(end_map.tv_sec-start_map.tv_sec)
			+(end_map.tv_usec-start_map.tv_usec + 0.0)/1000);

	printf("time of %d is %f, %d\n", me, time_map, nchar/(4*1024*1024));

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
	sprintf(filename, "/data/bzhaoad/mrmpi_output/InvertedIndex/InvertedIndex-%d-%d\0", nproc , me);

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
	else
		return 0;
}

/* ----------------------------------------------------------------------
   process a word and its count
   depending on flag, emit KV or print it, up to limit
------------------------------------------------------------------------- */

/*
void output(uint64_t itask, char *key, int keybytes, char *value,
		int valuebytes, KeyValue *kv, void *ptr)
{
	Count *count = (Count *) ptr;
	count->n++;
	if (count->n > count->limit) return;

	int n = *(int *) value;
	if (count->flag)
		printf("%d %s\n",n,key);
	else
		kv->add(key,keybytes,(char *) &n,sizeof(int));
}
 */

