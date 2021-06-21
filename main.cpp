#include <mpi.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>

#define ROOT 0
#define INPUT_PATH "C:\\Users\\cudauser\\Desktop\\input.txt"
#define OUTPUT_PATH "C:\\Users\\cudauser\\Desktop\\ouput.txt"
#define MASSIVE 10000

#define SIGN(f) (f < 0 ? -1 : 1)
#define MATCH_SIGN(a,b) ((a > 0 == b > 0) && a != 0)
#define MIN(a,b) (a > b ? b : a)

#define TAG_FOUNDSOL 10
#define TAG_NOSOL 9

typedef struct point
{
	int group;
	float* pos;
	float* vel;
} point;

typedef struct info		// the struct will store the information on the first line of the file
{
	int N;
	int K;
	float dT;
	float tmax;
	float a;
	int LIMIT;
	float QC;
} params;

extern cudaError_t advanceTimeCuda(float* arrpos, float* velarr, float dT, int N, int K, int numunits, int updateval);

void printarr(int* arr, int size);
int readPointsFromFile(float** pos, float** vel, int** group, params* info);
float compute(float* pos, float* vel, int* group, float* weights ,params* info);
float f(float* w, float* x, int size);
void fixW(float* w, float* p, float f, float a, int size);
int compute_OMP_CUDA2(point* points, float* weights, params* info, float t0, float* finalQT);
void fixWForOMP(float* w, float* p, float f, float a, int size, float iddT, float* velp);
float fForOMP(float* w, float* x, int size, float iddT, float* velp);
int checkAnswer(int* a, int* answeredByProcId, int numproc, MPI_Status* status);
void copyWeights(float* weights, float* otherweights, int size);
point* createPoints(float* posarr, float* velarr, int* grouparr, params* info);
void commitInfoDataType(params* info, MPI_Datatype* InfoMPIType);
void setrange(float* rangeS, int* preAdvance, int numprocs, int myid, params* info, int ksize);
void writeResultToFile(float q, float t, float* w, int k);

int main(int argc,char *argv[])
{
	params info;				
	
    int  namelen, numprocs, myid, preAdvance;
	float rangeS;

    char processor_name[MPI_MAX_PROCESSOR_NAME];
	cudaError_t cudaStatus;

    MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);	

	MPI_Get_processor_name(processor_name,&namelen);
	
	MPI_Status status;
	
	float* 	posarr;
	float* 	velarr;
	int* 	grouparr;
	point* points;

	float* 	weights;

	float* procWeights;
	
	// --- MPI  prepare to send "info" ---
		
	MPI_Datatype InfoMPIType;
	commitInfoDataType(&info, &InfoMPIType);
		
	// --- read data ---
	
	if (myid == ROOT)
	{
		int status = readPointsFromFile(&posarr, &velarr, &grouparr, &info);

		if((status < 0))
			MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	// --- send info ---
	
	if (numprocs > 1)
	{
		MPI_Bcast(&info, 1, InfoMPIType, 0, MPI_COMM_WORLD);

		if(myid != ROOT)
		{
			posarr = (float *)malloc(sizeof(float)*info.N*info.K);
			velarr = (float *)malloc(sizeof(float)*info.N*info.K);
			grouparr = (int *)malloc(sizeof(int)*info.N);
			weights = (float *)calloc(info.K + 1, sizeof(float));
		}
		
		MPI_Bcast(posarr, info.K*info.N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(velarr, info.K*info.N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(grouparr, info.N, MPI_INT, ROOT, MPI_COMM_WORLD);
		
		// --- find range of calcolation for each process --
		
		int ksize = (int)(info.tmax / info.dT);

		// finalize some processes if there are too many
		if (numprocs > ksize)	
		{
			numprocs = ksize;
			if (ksize <= myid)
			{

				free(posarr);
				free(velarr);
				free(grouparr);

				cudaStatus = cudaDeviceReset();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceReset failed!");
					return 1;
				}

				MPI_Finalize();
			}
		}
		
		procWeights = (float*)calloc(1, sizeof(float)*(info.K + 1 + 2));	// prepare information array for sending/receiving,  procWeights saves the weights and q and the time
		setrange(&rangeS, &preAdvance, numprocs, myid, &info, ksize);		// set range of time for each process
	}
	else
	{
		rangeS = 0;
	}

	//--- calcolate ---
	
	if(myid == ROOT)
	{
		weights = (float*)calloc(info.K + 1, sizeof(float));

		// --- calcolation ---

		int foundSol = 0;
		float finalQT[2] = { 0 };
		points = createPoints( posarr, velarr, grouparr, &info);

		foundSol = compute_OMP_CUDA2(points, weights, &info, 0, finalQT);

		// --- logic for receiving information from other processes ---

		int* answeredByProcId = (int*)calloc(numprocs, sizeof(int));	// this array is used for following which processes sent an answer
		int ansindex = 0;												// index of process with current answer
		int minsol = foundSol ? 0 : numprocs;							// lowest index with an answer

		for (int i = 1; i < numprocs && !foundSol; i++)
		{
			MPI_Recv(procWeights, info.K + 1 + 2, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			
			if (status.MPI_SOURCE < minsol && status.MPI_TAG == TAG_FOUNDSOL) 
			{
				minsol = status.MPI_SOURCE;
				copyWeights(weights, procWeights, info.K + 1);

				finalQT[0] = procWeights[info.K + 1];
				finalQT[1] = procWeights[info.K + 2];
			}

			foundSol = checkAnswer(&ansindex, answeredByProcId, numprocs, &status);
		}

		free(answeredByProcId);

		// --- print information ---
		
		if(foundSol)
		{
			printf("t = %f, q = %f\nw = ", finalQT[1], finalQT[0]); 

			for (int i = 0; i < info.K + 1; i++)
				printf("%f ",weights[i]);
			printf("\n");

			writeResultToFile(finalQT[1], finalQT[0], weights, info.K);
		}
		else
		{
			printf("No solution was found.\n");
		}
		
	}
	else 
	{
		// --- advance points by a certain time (according to the range found for each process) ---

		cudaError_t cudaStatus = advanceTimeCuda(posarr, velarr, info.dT, info.N, info.K, 1, preAdvance); 
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "advanceTimeCuda failed!");
			return -1;
		}

		// --- calcolate ---

		int foundSol = 0;
		float finalQT[2] = {0};

		points = points = createPoints(posarr, velarr, grouparr, &info);

		foundSol = compute_OMP_CUDA2(points, weights, &info, rangeS, finalQT);

		// -- prepare information and send ---

		copyWeights(procWeights, weights, info.K + 1);
		procWeights[info.K + 1] = finalQT[0];
		procWeights[info.K + 2] = finalQT[1];

		int tag = foundSol ? TAG_FOUNDSOL : TAG_NOSOL;
		MPI_Send(procWeights, info.K + 1 + 2, MPI_FLOAT, ROOT, tag, MPI_COMM_WORLD);
	}

	// --- end program ---
	
	free(posarr);
	free(velarr);
	free(grouparr);
	free(points);
	if(numprocs > 1)
		free(procWeights);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	MPI_Finalize();
	return 0;
}


int readPointsFromFile(float** pos, float** vel, int** group, params* info){

    FILE *myFile;
    int j, i;

    myFile = fopen(INPUT_PATH, "r");
	if(myFile == NULL){
		fprintf(stderr, "error when opening file\n");
		return -1;
	}

    fscanf(myFile, "%d %d %f %f %f %d %f", &info->N, &info->K, &info->dT, &info->tmax, &info->a, &info->LIMIT, &info->QC);

	if(info->N == 0){
		printf("file empty, ending..\n");
		return -1;
	}

    *pos = (float*)malloc(sizeof(float)*(info->N*info->K));
    *vel = (float*)malloc(sizeof(float)*(info->N*info->K));
    *group = (int*)malloc(sizeof(int)*(info->N));

    for(i = 0; i < info->N; i++){
		for(j = 0; j < info->K; j++){
			fscanf(myFile, "%f", *pos + i*info->K + j);
		}
		for(j = 0; j < info->K; j++){
			fscanf(myFile, "%f", *vel + i*info->K + j);
		}
		fscanf(myFile, "%d", *group + i);
	}
    
    fclose(myFile);
	return 1;
}

void printarr(int* arr, int size){
	int i;
	for(i = 0; i < size; i++){
        printf("%d\n", arr[i]);
        fflush(stdout);
    }
}

// basic fuction to compute
float compute(float* pos, float* vel, int* group, float* weights, params* info)
{
	int N = info->N;
	int K = info->K;
	int i, j, valid, active;
	float t, check, q;
	t = 0;
	active = 1;
	
	// while loop
	while(active){

		for (i = 0; i < K + 1; i++) 
		{
			weights[i] = 0;
		}
		
		valid = 0;		
		
		for(j = 0; j < info->LIMIT && !valid; j++) // loop until reach max LIMIT or classified is ok
		{
			valid = 1;
			
			for(i = 0; i < N; i++)
			{	
				check = f(weights, pos + i*K, K); // check if correct using function(x)
				
				if(!MATCH_SIGN(check, group[i])) // if not 
				{
					valid = 0;
					fixW(weights, pos + K*i, check, info->a, K); // do w = w + [a*sign(f(P))]P
				}
				// continue to check rest of points with updated value of w
			}
		}
		
		// check q=Nmis/N
		q = 0;					
		for(i = 0; i < N; i++)
		{
			check = f(weights, pos + i*K, K);
			q += (!MATCH_SIGN(check, group[i]));
		}
		
		q = q / (float)N;

		
		t += info->dT; // increment
		
		if (q < info->QC || t > info->tmax) // stop if q is less than QC or if t > tmax
		{
			break;
		}
				
		for(i = 0; i< N; i++){
			for(j = 0; j < K; j++){
				pos[i*K + j] += vel[i*K + j] * info->dT; // advance each point P = P0 +t*V
			}
		}
	}
	return t - info->dT;
}


int compute_OMP_CUDA2(point* points, float* weights, params* info, float t0, float* finalQT)
{
	int N = info->N;
	int K = info->K;
	float a = info->a;
	float dT = info->dT;
	int LIMIT = info->LIMIT;
	float tmax = info->tmax;
	float QC = info->QC;

	int i, j, v, done, id;
	float check, q, t;

	int maxthreads = omp_get_max_threads();

	omp_set_num_threads(maxthreads);

	int* validarr = (int*)calloc(maxthreads, sizeof(int));
	float* tempweights = (float*)calloc(maxthreads*(K + 1), sizeof(float));
	float* qs = (float*)calloc(maxthreads, sizeof(float));
	float* iddt = (float*)calloc(maxthreads, sizeof(float));	// not necessery, but works alot faster with this
	
	done = 0;
	t = t0;

	while(!done)
	{
		for (i = 0; i < maxthreads*(K + 1); i++)
		{
			tempweights[i] = 0;
		}

		#pragma omp parallel private(id, i, j, v, check, q)
		{
			
			id = omp_get_thread_num();

			iddt[id] = id*dT;
			if (iddt[id] + t < info->tmax)
			{
				for (j = 0; j < LIMIT && !(validarr[id]); j++) // loop until reach max LIMIT or classified is ok
				{
					validarr[id] = 1;

					for (i = 0; i < N; i++)
					{

						check = fForOMP(tempweights + id*(K + 1), points[i].pos, K, iddt[id], points[i].vel); // check if correct using function(x)

						if (!MATCH_SIGN(check, points[i].group)) // if not 
						{
							validarr[id] = 0;											 // mark as not valid
							fixWForOMP(tempweights + id*(K + 1), points[i].pos, check, a, K, iddt[id], points[i].vel); // do w = w - [a*sign(f(P))]P
						}
						// continue to check rest of points with updated value of w
					}
				}

				qs[id] = 0;
				for (v = 0; v < N; v++)		// for every point in thread
				{
					check = fForOMP(tempweights + id*(K + 1), points[v].pos, K, iddt[id], points[v].vel); // check if correct using function(x)

					qs[id] += (!MATCH_SIGN(check, points[v].group));
				}

				qs[id] = qs[id] / N;
			}
		}
		
		for(i = 0; i < maxthreads; i++)
		{
			float currentT = t + iddt[i];
			if(qs[i] < QC || currentT >= tmax)
			{
				for (v = 0; v < K + 1; v++)	// save weights
				{
					weights[v] = tempweights[i*(K + 1) + v];
				}
				
				t += iddt[i];

				finalQT[0] = qs[i];
				finalQT[1] = t;

				free(iddt);
				free(qs);
				free(validarr);
				free(tempweights);

				return currentT >= tmax ? 0 : 1; // returns 0(False) if current time larger than tmax, otherwise returns 1(True)
			}
		}

		
		// --- cuda advance all points
		
		cudaError_t cudaStatus = advanceTimeCuda(points->pos, points->vel, dT, N, K, 1, maxthreads);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "advanceTimeCuda failed!");
			return -1;
		}

		t += dT*maxthreads;
		
	}
	return 0;
}


void fixW(float* w, float* p, float f, float a, int size)
{
	int sign = SIGN(f);
	
	for(int i = 0; i < size; i++){
		w[i + 1] -= (a*sign)*p[i];
	}
	w[0] -= a*sign;
}

float f(float* w, float* x, int size)
{
	int i;
	float result = 0;
	
	for(i = 0; i < size; i++){
		result += w[i + 1]*x[i];
	}
	result += w[0];
	
	return result;
}

void fixWForOMP(float* w, float* p, float f, float a, int size, float iddT, float* velp)
{
	int sign = SIGN(f);
	
	for(int i = 0; i < size; i++){
		w[i + 1] -= (a*sign)*(p[i] + iddT*velp[i]);
	}
	w[0] -= a*sign;
}


float fForOMP(float* w, float* x, int size, float iddT, float* velp)
{
	int i;
	float result = 0;
	
	for(i = 0; i < size; i++){
		result += w[i + 1]*(x[i] + iddT*velp[i]);
	}
	result += w[0];
	
	return result;
}


int checkAnswer(int* a, int* answeredByProcId, int numproc, MPI_Status* status)
{
	int ansindex = *a;
	int arrivedindex = status->MPI_SOURCE;

	if (status->MPI_TAG == TAG_FOUNDSOL) { answeredByProcId[arrivedindex] = 1; }
	else { answeredByProcId[arrivedindex] = -1; }

	if (arrivedindex == ansindex + 1)
	{
		while(answeredByProcId[ansindex + 1] != 0 && ansindex < numproc - 1)
		{
			ansindex++;

			if (answeredByProcId[ansindex] == 1)
			{
				return 1;
			}
		}
	}
	return 0;
}

void copyWeights(float* weights, float* otherweights, int size)
{
	for (int j = 0; j < size; j++)
	{
		weights[j] = otherweights[j];
	}
}

point* createPoints(float* posarr, float* velarr, int* grouparr, params* info)
{
	point* points = (point*)malloc(sizeof(point)*info->N);
	int i;
	#pragma omp parallel private(i)
	{
		#pragma omp for
		for (i = 0; i < info->N; i++)
		{
			points[i].group = grouparr[i];
			points[i].pos = posarr + i*info->K;
			points[i].vel = velarr + i*info->K;
		}
	}
	return points;
}

void commitInfoDataType(params* info, MPI_Datatype* InfoMPIType)
{
	MPI_Datatype type[7] = { MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT, MPI_FLOAT };
	int blocklen[7] = { 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[7];

	disp[0] = (char *)&info->N		- (char *)info;
	disp[1] = (char *)&info->K		- (char *)info;
	disp[2] = (char *)&info->dT		- (char *)info;
	disp[3] = (char *)&info->tmax	- (char *)info;
	disp[4] = (char *)&info->a		- (char *)info;
	disp[5] = (char *)&info->LIMIT	- (char *)info;
	disp[6] = (char *)&info->QC		- (char *)info;

	MPI_Type_create_struct(7, blocklen, disp, type, InfoMPIType);
	MPI_Type_commit(InfoMPIType);
}

void setrange(float* rangeS, int* preAdvance, int numprocs, int myid, params* info, int ksize)
{
	int chunkSize = (int)(ksize / numprocs);
	float rangeE;
	float rS;

	rS = myid * chunkSize;
	rangeE = rS + chunkSize;
	*preAdvance = (int)rS;

	rangeE *= info->dT;
	rS *= info->dT;

	if (myid == numprocs - 1)
		rangeE = info->tmax;

	*rangeS = rS;

	info->tmax = rangeE;
}

void writeResultToFile(float q, float t, float* w, int k)
{
	FILE* fp;
	fp = fopen(OUTPUT_PATH, "w");

	fprintf(fp, "t = %f, q = %f\nw = ", q, t);

	for (int i = 0; i < k + 1; i++)
		fprintf(fp, "%f ", w[i]);

	fclose(fp);
}