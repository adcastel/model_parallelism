#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "omp.h"
//#include "cblas.h"
#include "mkl.h"

#define INPUT -1
#define FC 0
#define CONV 1

#define START_TIMER_FP(f,s,l) f[s][l] = MPI_Wtime();
#define END_TIMER_FP(s,l)         fp_comp_timer[s][l] = MPI_Wtime() - fp_comp_timer[s][l];\
        MPI_Reduce(&fp_comp_timer[s][l], &fp_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);\
        m = nneurons[l];\
        fp_comp_gflops[s][l] = (2.0*m*n*k/fp_comp_timer[s][l])/(1.0e+9);\
        fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l]); 


/*Threads for the convolution */
#define CONV_THREADS 8
/* Model features */

#define NUM_STEPS 	30  // Steps of the simulation
#define BATCH_SIZE 	64 // Batch size

#ifdef ALEXNET
#define NUM_LAYERS 	9  // Number of layers

#elif defined VGG16 
#define NUM_LAYERS 	17  // Number of layers

#elif defined RESNET
#define NUM_LAYERS 	55  // Number of layers

#elif defined INCEPTION
#define NUM_LAYERS 	96  // Number of layers

#else //Test

#define NUM_LAYERS 	6  // Number of layers

#endif
/* helper functions */
int problem_size(int elements, int nprocs, int rank);

/* Computation functions */
void FC_gemm_fp( int m, int n, int k, float * A, int lda, float * B, int ldb, float * C , int ldc);
void FC_gemm_cg( int m, int n, int k, float * A, int lda, float * B, int ldb, float * C , int ldc);
void FC_gemm_wu( int m, int n, int k, float * A, int lda, float * B, int ldb, float * C , int ldc);
void CONV_fp(int l, int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time);
void CONV_cg(int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time);
void CONV_wu(int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time);

/* Communication functions */
void allgather(int n, float * C, MPI_Comm comm);
void gather(int n, float * C, MPI_Comm comm);
void allreduce(int n, float * C, MPI_Comm comm);
void reduce(int n, float * C, MPI_Comm comm);
void bcast (int n, float * buff, MPI_Comm comm);
void scatter (size_t n, float * buff, MPI_Comm comm);

int main (int argc, char * argv []){

    int rank, size, i, s, l;    
    double alpha = 1.0, beta = 0.0;
#ifdef TIMER
    double scatter_time;
    double step_timer[NUM_STEPS];
    double initial_bcast_timer[NUM_STEPS];
    double fp_comp_timer[NUM_STEPS][NUM_LAYERS];
    double fp_im2col_timer[NUM_STEPS][NUM_LAYERS];
    double fp_comp_gflops[NUM_STEPS][NUM_LAYERS];
    double fp_comp_gflops_per_thread[NUM_STEPS][NUM_LAYERS];
    double fp_comm_timer_red[NUM_STEPS][NUM_LAYERS];
    double fp_comm_timer_bcast[NUM_STEPS][NUM_LAYERS];
    double cg_comp_timer[NUM_STEPS][NUM_LAYERS];
    double cg_im2col_timer[NUM_STEPS][NUM_LAYERS];
    double cg_comp_gflops[NUM_STEPS][NUM_LAYERS];
    double cg_comp_gflops_per_thread[NUM_STEPS][NUM_LAYERS];
    double cg_comm_timer_red[NUM_STEPS][NUM_LAYERS];
    double cg_comm_timer_bcast[NUM_STEPS][NUM_LAYERS];
    double wu_comp_timer[NUM_STEPS][NUM_LAYERS];
    double wu_im2col_timer[NUM_STEPS][NUM_LAYERS];
    double wu_comp_gflops[NUM_STEPS][NUM_LAYERS];
    double wu_comp_gflops_per_thread[NUM_STEPS][NUM_LAYERS];
#endif
    /* Model */

#ifdef ALEXNET
    int type[NUM_LAYERS] = { INPUT, CONV, CONV, CONV, CONV, CONV, FC, FC, FC}; 
    int nneurons[NUM_LAYERS] = { 224*224*3, 55*55*64, 27*27*192 , 13*13*384, 13*13*384, 13*13*256,4096,4096,1001}; //Neurons per layer
#ifndef STATIC
    int min_size[NUM_LAYERS] = { 0, 16, 16, 16, 16, 16,512,512,512}; //minimum data size per layer
#else    
    int min_size[NUM_LAYERS] = { 0, 1, 1, 1, 1, 1,1,1,1}; //minimum data size per layer
#endif    
    int image_size[NUM_LAYERS] = { 224, 55, 27 , 13, 13, 13, 0,0.0}; //pixels
    int nkernels[NUM_LAYERS] = { 0,  64, 192, 384, 384,256,0,0,0}; //kernels per layer
    int channels[NUM_LAYERS] = { 3,  64, 192,384,384,256,1, 1, 1}; //channels per layer
    int kwidth[NUM_LAYERS] = { 0, 11, 5, 3,3,3,0,0,0}; //Sizes of kernels
    int kheight[NUM_LAYERS] = { 0, 11, 5, 3,3,3,0,0,0}; //Sizes of kernels
    int vstrides[NUM_LAYERS] = { 0, 4, 1, 1,1,1,0,0,0}; //Stride of kernels
    int hstrides[NUM_LAYERS] = { 0, 4, 1, 1,1,1,0,0,0}; //Stride of kernels

#elif defined VGG16
    int type[NUM_LAYERS] = { INPUT, CONV, CONV, CONV, CONV, CONV,CONV,CONV,CONV,CONV,CONV,CONV,CONV,CONV, FC, FC, FC}; 
    int nneurons[NUM_LAYERS] = { 227*227*3, 227*227*64, 227*227*64,112*112*128,112*112*128 , 56*56*256,56*56*256,56*56*256,28*28*512, 28*28*512, 28*28*512,14*14*512,14*14*512,14*14*512,4096,4096,1001}; //Neurons per layer
#ifndef STATIC
    int min_size[NUM_LAYERS] = { 0, 16, 16, 16, 16,16,16, 16,16, 16, 16,16,16,16,512,512,512}; //minimum data size per layer
#else    
    int min_size[NUM_LAYERS] = { 0, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1}; //minimum data size per layer
#endif    
    int image_size[NUM_LAYERS] = { 227, 227,227,112,112,56, 56,56,28,28,28 ,14,14,14,0,0,0}; //pixels
    int nkernels[NUM_LAYERS] = { 0,  64, 64,128, 128,256,256,256,512,512,512,512,512,512,0,0,0}; //kernels per layer
    int channels[NUM_LAYERS] = { 3,  64, 64,128, 128,256,256,256,512,512,512,512,512,512,1, 1, 1}; //channels per layer
    int kwidth[NUM_LAYERS] = { 0, 3, 3, 3,3,3,3,3,3,3,3,3,3,3,0,0,0}; //Sizes of kernels
    int kheight[NUM_LAYERS] = {0, 3, 3, 3,3,3,3,3,3,3,3,3,3,3,0,0,0}; //Sizes of kernels
    int vstrides[NUM_LAYERS] = { 0, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,0,0,0}; //Stride of kernels
    int hstrides[NUM_LAYERS] = { 0, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,0,0,0}; //Stride of kernels

#elif defined RESNET

#elif defined INCEPTION

#else //test
#ifdef FULLY
    int type[NUM_LAYERS] = { INPUT, FC, FC, FC, FC, FC}; 
    int nneurons[NUM_LAYERS] = { 224*224*3, 2048, 1024 , 2048, 4096, 1001}; //Neurons per layer
#ifndef STATIC
    int min_size[NUM_LAYERS] = { 0, 512, 512, 512, 512, 512}; //minimum data size per layer
#else    
    int min_size[NUM_LAYERS] = { 0, 1, 1, 1, 1, 1}; //minimum data size per layer
#endif    
#else
    int type[NUM_LAYERS] = { INPUT, CONV, CONV, FC, FC, FC}; 
    int nneurons[NUM_LAYERS] = { 224*224*3, 112*112*64, 56*56*128 , 2048, 4096, 1001}; //Neurons per layer
#ifndef STATIC
    int min_size[NUM_LAYERS] = { 0,  8, 8, 512, 512, 512}; //minimum data size per layer    
#else    
    int min_size[NUM_LAYERS] = { 0, 1, 1, 1, 1, 1}; //minimum data size per layer
#endif    

#endif    
    /* ONLY USED IN CONV LAYERS */
    int image_size[NUM_LAYERS] = { 224, 112, 56 , 0, 0,0 }; //pixels
    int nkernels[NUM_LAYERS] = { 0,  64, 128, 0, 0,0}; //kernels per layer
    int channels[NUM_LAYERS] = { 3,  64, 128, 1, 1,0}; //channels per layer
    //int min_size[NUM_LAYERS] = { 0,  1, 1, 1, 1,1}; //minimum data size per layer
    int kwidth[NUM_LAYERS] = { 0, 11, 3, 0,0,0}; //Sizes of kernels
    int kheight[NUM_LAYERS] = { 0,11, 3, 0,0,0}; //Sizes of kernels
    int vstrides[NUM_LAYERS] = { 0, 1, 1, 0,0,0}; //Stride of kernels
    int hstrides[NUM_LAYERS] = { 0, 1, 1, 0,0,0}; //Stride of kernels

#endif

    int procs[NUM_LAYERS]; //number of procs that worker per layer
    int max_procs = 0; //This variable is used for the group and communicator creation;

    const char* env = getenv("OMP_NUM_THREADS");
    int OMP_NUM_THREADS = (env!=NULL) ? atoi(env) : 1;

    MPI_Comm communicators[NUM_LAYERS],/* world_group,*/ max_procs_comm;
    MPI_Group groups[NUM_LAYERS], max_procs_group,world_group;

     
    /* Some MPI initializations */
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
/* We first calculate the max size of matrix A, B and C so we only allocate once 
       and reuse them for all the execution */
    
    size_t max_size_fc = 0;
    size_t max_size_conv = 0;
    for (l = 1; l < NUM_LAYERS; l++){
	if(type[l]==FC){    
        	if ((1.0*nneurons[l] * nneurons[l-1]) > max_size_fc){
	    		max_size_fc = (1.0*nneurons[l] * nneurons[l-1]);
		}
	}
	else{
		if((nneurons[l] + nkernels[l]*kwidth[l]*kheight[l]) > max_size_conv){
			max_size_conv = (nneurons[l] + nkernels[l]*kwidth[l]*kheight[l]);
		}
        }
    }
    
    /* This matrices are for FC layers */
    float * matrix_A = malloc( max_size_fc * sizeof( float));
    float * matrix_B = malloc( max_size_fc * sizeof( float));
    float * matrix_C = malloc( max_size_fc * sizeof( float));

    size_t max_i = channels[0]*BATCH_SIZE*image_size[0]*image_size[0];
    size_t max_ip = channels[0]*kwidth[0]*kheight[0]*BATCH_SIZE*image_size[0]*image_size[0];
    size_t max_o = nkernels[0]*BATCH_SIZE*image_size[0]*image_size[0];
    size_t max_f = nkernels[0]*channels[0]*kwidth[0]*kheight[0];
    for (l = 1; l < NUM_LAYERS; l++){
	if(type[l] == CONV){
	    size_t mi = channels[l]*BATCH_SIZE*image_size[l]*image_size[l];
            size_t mip = channels[l]*kwidth[l]*kheight[l]*BATCH_SIZE*image_size[l]*image_size[l];
    	    size_t mo = nkernels[l]*BATCH_SIZE*image_size[l]*image_size[l];
            size_t mf = nkernels[l]*channels[l]*kwidth[l]*kheight[l];
	    if(mi > max_i){
                max_i = mi;
            }
            if(mip > max_ip){
	        max_ip = mip;
            }
	    if(mo > max_o){
                max_o = mo;
            }
            if(mf > max_f){
	        max_f = mf;
            }
    	}
    }
    float * conv_i = malloc(max_i*sizeof(float));
    float * conv_ip = malloc(max_ip*sizeof(float));
    float * conv_o = malloc(max_o*sizeof(float));
    float * conv_f = malloc(max_f*sizeof(float));
    /* The maximum size of the matrix model fits with the one of the matrix */    
    size_t model_size = (max_size_fc > max_size_conv) ? max_size_fc : max_size_conv; //This is the maximum size of the model layers
    float * model = malloc (model_size * sizeof(float));
    /* The data size is equal to the neurons in the first layer per the batch size */
    size_t data_size = nneurons[0]*BATCH_SIZE;
    float * data = malloc (data_size * sizeof(float));
    

    

    /* With the minimum size, we calculate the number of procs that acts in each layer */     
    /* Moreover we calculate the max number of processes working in the model */
    for (l = 1; l < NUM_LAYERS; l++){
        int num_procs;
        if (type[l] == FC){
	    num_procs = (int)ceil(nneurons[l]*1.0f/min_size[l]);
	}
	else{
	    num_procs = (int)ceil(nkernels[l]*1.0f/min_size[l]);
	}	
        procs[l] = (num_procs > size) ? size : num_procs;
        if (procs[l] > max_procs) max_procs = procs[l];
    }
    procs[0] = procs[1];
    /* we now generate a list that will be used in the communication group construction */
    int * ranks = malloc (max_procs * sizeof(int));
    for (i = 0; i < max_procs; i++){
	ranks[i] = i;
    }
   
    /* For each layer, we generate a group with the number of active processes*/
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    for(l = 0; l < NUM_LAYERS; l++){
        MPI_Group_incl(world_group, procs[l], ranks, &groups[l]);
    }
    MPI_Group_incl(world_group, max_procs, ranks, &max_procs_group); //Used for barriers inside simulation
    MPI_Group_free(&world_group);

    /* For each group we create a communicator */
    /* The if clause is needed because each process acts in the creation */
    for(l = 0; l < NUM_LAYERS; l++){
       if(rank < procs[l]){
           MPI_Comm_create_group(MPI_COMM_WORLD,groups[l],0,&communicators[l]); 
       }
    }
    if(rank < max_procs){
        MPI_Comm_create_group(MPI_COMM_WORLD,max_procs_group,0,&max_procs_comm); 
    }

if(rank == 0){
#ifdef ALEXNET
	printf("****AlexNet Model****\n");

#elif defined VGG16
	printf("****VGG16 Model****\n");

#elif defined RESNET
	printf("****ResNet-50 Model****\n");

#elif defined INCEPTION
	printf("****Inception-v3 Model****\n");

#else //Test
	printf("****Test Model****\n");

#endif
}

    MPI_Barrier(MPI_COMM_WORLD);

/* Warm-up zone */
   int omp_warm;
   #pragma omp parallel
   {
       omp_warm = omp_get_thread_num();
   }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				200,200,200, 1,
				matrix_A, 200, matrix_B, 200,0,matrix_C,200);
   
/* Now, we start the simulation */
/* We first divide the model among the number of processes */
/* TODO: Should we use another group instead of general? */ 
    if (rank < max_procs){
//Scatter from P0
#ifdef TIMER
    scatter_time = MPI_Wtime();
#endif
    for (l = 1; l < NUM_LAYERS; l++){
	size_t comm_size;
        if(type[l] == FC){
	    comm_size = (1.0*nneurons[l] /** nneurons[l-1]*/)/max_procs;
        }
        else{
	    comm_size = nneurons[l] + 1.0*nkernels[l]*kwidth[l]*kheight[l]/max_procs;
        }
//	printf("Scatter lyer %d\n",l);
 	scatter(comm_size, model, max_procs_comm);

    }
#ifdef TIMER
    scatter_time = MPI_Wtime() - scatter_time;
#endif

    for (s = 0; s < NUM_STEPS; s++)
    {
#ifdef PROGRESS
	printf("Starting Step %d\n",s);
#endif
	//printf("Step  %d\n",s);
#ifdef TIMER
	step_timer[s] = MPI_Wtime();
#endif
	/* Here, we send the data of the begining to the process in the first layer */
#ifdef TIMER
    initial_bcast_timer[s] = MPI_Wtime();
#endif
        if(rank < procs[0]){ 
	    bcast(data_size , data, communicators[0]);
        }
#ifdef TIMER
	initial_bcast_timer[s] = MPI_Wtime() - initial_bcast_timer[s]; 
#endif
        //Forward pass
    	for (l = 1; l < NUM_LAYERS-1; l++)
        {
	//printf("FP layer %d\n",l);
            /* We need both, the procs for current and next layers */
            if(rank < procs[l] || rank < procs[l+1]){
                /* procs in the current layer computes and gathers to root */
                if(rank < procs[l]){ 
	            if (type[l] == FC){ //FC
	    	        int m = problem_size(nneurons[l],procs[l],rank);//nneurons[l]/procs[l];//antes /size
		        int n = BATCH_SIZE;
		        int k = nneurons[l-1]; //We need to reshape if the previous one was CONV
		        int lda = m; 
		        int ldb = k;
		        int ldc = m;
#ifdef TIMER
	//START_TIMER_FP(fp_comp_timer,s,l)
	fp_comp_timer[s][l] = MPI_Wtime();
#endif
		        FC_gemm_fp( m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc);
#ifdef TIMER
	//END_TIMER_FP(s,l)
	fp_comp_timer[s][l] = MPI_Wtime() - fp_comp_timer[s][l];
        MPI_Reduce(&fp_comp_timer[s][l], &fp_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        m = nneurons[l];
	//printf("FC_gemm_fp FLOPS = 2* %d * %d * %d = %f\n",m,n,k,2.0*m*n*k);
        fp_comp_gflops[s][l] = (2.0*m*n*k/fp_comp_timer[s][l])/(1.0e+9);
        fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l]);
#endif
             
		        int comm_size = (nneurons[l]/procs[l])*BATCH_SIZE; //antes /size
#ifdef TIMER
	fp_comm_timer_red[s][l] = MPI_Wtime();
#endif
                        gather(comm_size, matrix_C, communicators[l]); 
#ifdef TIMER
	fp_comm_timer_red[s][l] = MPI_Wtime() - fp_comm_timer_red[s][l];
#endif
                    }
	            else{ //conv
            	        int num_kernels = problem_size(nkernels[l],procs[l],rank);//nneurons[l]/procs[l];//antes /size
		        //int num_channels = channels[l-1];//printf("Error. Not yet implemented\n"); return -1;
#ifdef TIMER
	//START_TIMER_FP(fp_comp_timer,s,l)
	fp_comp_timer[s][l] = MPI_Wtime();
#endif
			
		        //printf("CONV_fp(%d, %d, %d, %d, %d, %d, %d, conv_i, conv_ip, conv_o, conv_f, &fp_im2col_timer[s][l]);\n",num_kernels, BATCH_SIZE, image_size[l-1], image_size[l-1], kheight[l], kwidth[l], channels[l-1]);
		        //printf("Rank = %d ", rank);
			CONV_fp(l,num_kernels, BATCH_SIZE, image_size[l], image_size[l], kheight[l], kwidth[l], channels[l],  conv_i, conv_ip, conv_o, conv_f, &fp_im2col_timer[s][l]);
		        //printf("END CONV_fp(%d, %d, %d, %d, %d, %d, %d, conv_i, conv_ip, conv_o, conv_f, &fp_im2col_timer[s][l]);\n",num_kernels, BATCH_SIZE, image_size[l-1], image_size[l-1], kheight[l], kwidth[l], channels[l-1]);
                            	
#ifdef TIMER
	fp_comp_timer[s][l] = MPI_Wtime() - fp_comp_timer[s][l];
        MPI_Reduce(&fp_comp_timer[s][l], &fp_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        MPI_Reduce(&fp_im2col_timer[s][l], &fp_im2col_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        int m= nkernels[l];
	int n= BATCH_SIZE*image_size[l]*image_size[l];
	int k= channels[l]*kheight[l]*kwidth[l];
        //fp_comp_timer[s][l] = fp_comp_timer[s][l]/procs[l];
        fp_comp_gflops[s][l] = (2.0*m*n*k/fp_comp_timer[s][l])/(1.0e+9);
        fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l]);
#endif
			int comm_size = (nneurons[l]/procs[l])*BATCH_SIZE;
#ifdef TIMER
	fp_comm_timer_red[s][l] = MPI_Wtime();
#endif
                        gather(comm_size, conv_o, communicators[l]); 
#ifdef TIMER
	fp_comm_timer_red[s][l] = MPI_Wtime() - fp_comm_timer_red[s][l];
#endif
	            }
                }
                /* This if clause is needed because we don't know the procs in next layer */
                if(rank < procs[l+1]){
	            int comm_size;
                    if (type[l] == FC){
			comm_size  = nneurons[l]*BATCH_SIZE;
#ifdef TIMER
	fp_comm_timer_bcast[s][l] = MPI_Wtime();
#endif
                    bcast(comm_size,matrix_C,communicators[l+1]);
#ifdef TIMER
	fp_comm_timer_bcast[s][l] = MPI_Wtime() - fp_comm_timer_bcast[s][l];
#endif
		    }
		    else{
			comm_size  = nneurons[l]*BATCH_SIZE;
#ifdef TIMER
	fp_comm_timer_bcast[s][l] = MPI_Wtime();
#endif
                    bcast(comm_size,conv_o,communicators[l+1]);
#ifdef TIMER
	fp_comm_timer_bcast[s][l] = MPI_Wtime() - fp_comm_timer_bcast[s][l];
#endif
		    }
                    /* Now we bcast to new workers */
                }
            }
            /* We need this barrier so a process that not works in current layer can not 
               continue */
            MPI_Barrier(max_procs_comm);
        }
        /* This layer is isolated because the code is more easy to follow right now */
        //last layer in FP
        int ll = NUM_LAYERS-1;
	//printf("FP layer %d\n",ll);
        if(rank < procs[ll]){
            if (type[ll] == FC){ //FC
	    	int m = problem_size(nneurons[ll],procs[ll],rank);//nneurons[l]/procs[l];//antes /size
                int n = BATCH_SIZE;
	        int k = nneurons[ll-1]; //We need to reshape if the previous one was CONV
                int lda = m;    
                int ldb = k;
                int ldc = nneurons[ll];
#ifdef TIMER
	fp_comp_timer[s][ll] = MPI_Wtime();
#endif
                FC_gemm_fp( m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc);
#ifdef TIMER
	fp_comp_timer[s][ll] = MPI_Wtime() - fp_comp_timer[s][ll];
        MPI_Reduce(&fp_comp_timer[s][ll], &fp_comp_timer[s][ll], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[ll]);
        m = nneurons[ll];
        //fp_comp_timer[s][ll] = fp_comp_timer[s][ll]/procs[ll];
	//printf("FC_gemm_fp FLOPS = 2* %d * %d * %d = %f\n",m,n,k,2.0*m*n*k);
        fp_comp_gflops[s][ll] = (2.0*m*n*k/fp_comp_timer[s][ll])/(1.0e+9);
        fp_comp_gflops_per_thread[s][ll] = fp_comp_gflops[s][ll]/(OMP_NUM_THREADS*procs[ll]);
#endif
                            
                int comm_size = (nneurons[ll]/procs[ll])*BATCH_SIZE;
                /* In this case we use allgather because the procs for BP are the same */
#ifdef TIMER
	fp_comm_timer_red[s][ll] = MPI_Wtime();
#endif
                allgather(comm_size, matrix_C, communicators[ll]); 
#ifdef TIMER
	fp_comm_timer_red[s][ll] = MPI_Wtime() - fp_comm_timer_red[s][ll];
	fp_comm_timer_bcast[s][ll] = 0;
#endif
            }
            else{ //CONV are not usually placed as last layer but we take it into account for fully-convolutional models  
		int num_kernels = problem_size(nkernels[ll],procs[ll],rank);
#ifdef TIMER
	fp_comp_timer[s][ll] = MPI_Wtime();
#endif
		        //printf("Rank = %d ", rank);
		        CONV_fp(ll,num_kernels, BATCH_SIZE, image_size[ll], image_size[ll], kheight[ll], kwidth[ll], channels[ll], conv_i, conv_ip, conv_o, conv_f,&fp_im2col_timer[s][ll]);
                            	
#ifdef TIMER
	fp_comp_timer[s][ll] = MPI_Wtime() - fp_comp_timer[s][ll];
        MPI_Reduce(&fp_comp_timer[s][ll], &fp_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[ll]);
        MPI_Reduce(&fp_im2col_timer[s][ll], &fp_im2col_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[ll]);
        int m= nkernels[ll];
	int n= BATCH_SIZE*image_size[ll]*image_size[ll];
	int k= channels[ll]*kheight[ll]*kwidth[ll];
        //fp_comp_timer[s][ll] = fp_comp_timer[s][ll]/procs[ll];
        fp_comp_gflops[s][ll] = (2.0*m*n*k/fp_comp_timer[s][ll])/(1.0e+9);
        fp_comp_gflops_per_thread[s][ll] = fp_comp_gflops[s][ll]/(1.0*OMP_NUM_THREADS*procs[ll]);
#endif
                    
                 //CONV    
                 
		 int comm_size = (nneurons[ll]/procs[ll])*BATCH_SIZE;
#ifdef TIMER
	fp_comm_timer_red[s][ll] = MPI_Wtime();
#endif
                 allgather(comm_size, conv_o, communicators[ll]); 
#ifdef TIMER
	fp_comm_timer_red[s][ll] = MPI_Wtime() - fp_comm_timer_red[s][ll];
	fp_comm_timer_bcast[s][ll] = 0;
#endif
            }
        }
        MPI_Barrier(max_procs_comm);

        //Backward pass 
        //Last layer only performs an allreduce that now is a reduce(current comm) + bcast(next comm)
        //We do not take into account the CG computation because it is quadratic
        // In this version we assume that the last layer is a FC
        //CG
	//printf("CG layer %d\n",NUM_LAYERS-1);
        if (rank < procs[NUM_LAYERS-1] || rank < procs[NUM_LAYERS-2]){
	    if (rank < procs[NUM_LAYERS-1]){
                int comm_size = (nneurons[NUM_LAYERS-1]/procs[NUM_LAYERS-1])*BATCH_SIZE;
#ifdef TIMER
	cg_comp_timer[s][NUM_LAYERS-1] = 0;
	cg_comp_gflops[s][NUM_LAYERS-1] = 0;
	cg_comp_gflops_per_thread[s][NUM_LAYERS-1] = 0;
	cg_comm_timer_red[s][NUM_LAYERS-1] = MPI_Wtime();
#endif
                reduce(comm_size, matrix_C, communicators[NUM_LAYERS-1]);
#ifdef TIMER
	cg_comm_timer_red[s][NUM_LAYERS-1] = MPI_Wtime() - cg_comm_timer_red[s][NUM_LAYERS-1];
#endif
            }
            if (rank < procs[NUM_LAYERS-2]){
	        int comm_size = (nneurons[NUM_LAYERS-1])*BATCH_SIZE;
#ifdef TIMER
	cg_comm_timer_bcast[s][NUM_LAYERS-1] = MPI_Wtime();
#endif
	        bcast(comm_size,matrix_C,communicators[NUM_LAYERS-2]);
#ifdef TIMER
	cg_comm_timer_bcast[s][NUM_LAYERS-1] = MPI_Wtime() - cg_comm_timer_bcast[s][NUM_LAYERS-1];
#endif
            }
        }
        //WU
	//printf("WU layer %d\n",NUM_LAYERS-1);
        if(rank < procs[NUM_LAYERS-1]){
            if(type[NUM_LAYERS-1] == FC){ //FC
                int m = problem_size(nneurons[NUM_LAYERS-1],procs[NUM_LAYERS-1],rank);
                int n = nneurons[NUM_LAYERS-2];
                int k = BATCH_SIZE;
	        int lda = m; 
		int ldb = n;
		int ldc = nneurons[NUM_LAYERS-1];
#ifdef TIMER
	wu_comp_timer[s][NUM_LAYERS-1] = MPI_Wtime();
#endif
		FC_gemm_wu( m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc);
#ifdef TIMER
        wu_comp_timer[s][NUM_LAYERS-1] = MPI_Wtime() - wu_comp_timer[s][NUM_LAYERS-1];
        MPI_Reduce(&wu_comp_timer[s][NUM_LAYERS-1], &wu_comp_timer[s][NUM_LAYERS-1], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[NUM_LAYERS-1]);
        //wu_comp_timer[s][NUM_LAYERS-1] = wu_comp_timer[s][NUM_LAYERS-1]/procs[NUM_LAYERS-1];
        m = nneurons[NUM_LAYERS-1];
	wu_comp_gflops[s][NUM_LAYERS-1] = (2.0*m*n*k/wu_comp_timer[s][NUM_LAYERS-1])/(1.0e+9);
        wu_comp_gflops_per_thread[s][NUM_LAYERS-1] = wu_comp_gflops[s][NUM_LAYERS-1]/(1.0*OMP_NUM_THREADS*procs[NUM_LAYERS-1]);
#endif

            }
	    else{
	        //printf("Error. Not yet implemented\n"); //return -1; We assume that last layer is a FC
            }
        }
        MPI_Barrier(max_procs_comm);

        //Central layers
    	for (l = NUM_LAYERS-2; l >= 2; l--){
	//printf("CG layer %d\n",l);
	    //CG
	    if(type[l] == FC){ //FC
                if(rank < procs[l] || rank < procs[l+1]){
                    if(rank < procs[l+1]){
	                int m = nneurons[l];
	                int n = BATCH_SIZE;
		        int k = problem_size(nneurons[l+1],procs[l+1],rank);
		        int lda = nneurons[l+1];//m; 
		        int ldb = nneurons[l+1];
		        int ldc = m;
#ifdef TIMER
	cg_comp_timer[s][l] = MPI_Wtime();
#endif
		        FC_gemm_cg( m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc );
#ifdef TIMER
	cg_comp_timer[s][l] = MPI_Wtime() - cg_comp_timer[s][l];
        MPI_Reduce(&cg_comp_timer[s][l], &cg_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l+1]);
        //cg_comp_timer[s][l] = cg_comp_timer[s][l]/procs[l+1];
	k = nneurons[l+1];
        cg_comp_gflops[s][l] = (2.0*m*n*k/cg_comp_timer[s][l])/(1.0e+9);
        cg_comp_gflops_per_thread[s][l] = cg_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l+1]);
#endif
		    
                        //Allreduce = reduce + bcast
                        int comm_size = (nneurons[l]/procs[l+1])*BATCH_SIZE;
#ifdef TIMER
	cg_comm_timer_red[s][l] = MPI_Wtime();
#endif
		        reduce(comm_size, matrix_C, communicators[l+1]);
#ifdef TIMER
	cg_comm_timer_red[s][l] = MPI_Wtime() - cg_comm_timer_red[s][l];
#endif
                   }
                   if (rank < procs[l]){
                       int comm_size;
                       comm_size = (nneurons[l])*BATCH_SIZE;
#ifdef TIMER
	cg_comm_timer_bcast[s][l] = MPI_Wtime();
#endif
	               bcast(comm_size,matrix_C,communicators[l]);
#ifdef TIMER
	cg_comm_timer_bcast[s][l] = MPI_Wtime() - cg_comm_timer_bcast[s][l];
#endif
                   }
                }
           }
	   else{ //CONV //We do not include the reduction of the deconvolution inside a process because is quadratic
                if(rank < procs[l] || rank < procs[l-1]){
                       if(rank < procs[l]){
		           int num_kernels = problem_size(nkernels[l],procs[l],rank);
#ifdef TIMER
	cg_comp_timer[s][l] = MPI_Wtime();
#endif
			    //We swap conv_i and conv_o
		            CONV_cg(num_kernels, BATCH_SIZE, (type[l+1] == CONV) ? image_size[l+1]: image_size[l], (type[l+1] == CONV) ? image_size[l+1] : image_size[l], kheight[l], kwidth[l], (type[l+1] == CONV) ? channels[l+1] : channels[l], conv_i, conv_ip, conv_o, conv_f, &cg_im2col_timer[s][l]);
                            	
#ifdef TIMER
	cg_comp_timer[s][l] = MPI_Wtime() - cg_comp_timer[s][l];
        MPI_Reduce(&cg_comp_timer[s][l], &cg_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        MPI_Reduce(&cg_im2col_timer[s][l], &cg_im2col_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        int m= nkernels[l];
	int aux = (type[l+1] == CONV) ? image_size[l+1]: image_size[l];
	int aux2 = (type[l+1] == CONV) ? image_size[l+1] : image_size[l];
	int n= BATCH_SIZE*aux*aux2;
	int k= channels[l]*kheight[l]*kwidth[l];
        //cg_comp_timer[s][l] = cg_comp_timer[s][l]/procs[l+1];
        cg_comp_gflops[s][l] = (2.0*m*n*k/cg_comp_timer[s][l])/(1.0e+9);
        cg_comp_gflops_per_thread[s][l] = cg_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l]);
#endif
                        
                            int comm_size =image_size[l-1]*image_size[l-1] *BATCH_SIZE;
		//We fusse the output so we don't need to make multiple reduce    for (int c = 0; c < channels[l-1]; c++){ //We need a reduce for each channel of the previous layer
#ifdef TIMER
	cg_comm_timer_red[s][l] = MPI_Wtime();
        //printf("pillo tiempo\n");
#endif
		        	reduce(comm_size, conv_i, communicators[l]);
#ifdef TIMER
	cg_comm_timer_red[s][l] = MPI_Wtime() - cg_comm_timer_red[s][l];
#endif
		//	}
                   }
		   if(rank < procs[l-1]){
		       int comm_size  = nneurons[l-1]*BATCH_SIZE;
#ifdef TIMER
	cg_comm_timer_bcast[s][l] = MPI_Wtime();
#endif
	           bcast(comm_size,conv_i,communicators[l-1]);
#ifdef TIMER
	cg_comm_timer_bcast[s][l] = MPI_Wtime() - cg_comm_timer_bcast[s][l];
#endif

		   }
               }
             }
	    //WU
	    //printf("WU layer %d\n",l);
            if(rank < procs[l]){
                if(type[l] == FC){ //FC
                    int m = problem_size(nneurons[l],procs[l],rank);
                    int n = nneurons[l-1];
                    int k = BATCH_SIZE;
		    int lda = m; 
		    int ldb = n;
		    int ldc = nneurons[l];
#ifdef TIMER
	wu_comp_timer[s][l] = MPI_Wtime();
#endif
		    FC_gemm_wu( m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc);
#ifdef TIMER
	wu_comp_timer[s][l] = MPI_Wtime() - wu_comp_timer[s][l];
        MPI_Reduce(&wu_comp_timer[s][l], &wu_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        //wu_comp_timer[s][l] = wu_comp_timer[s][l]/procs[l];
        m = nneurons[l];
        wu_comp_gflops[s][l] = (2.0*m*n*k/wu_comp_timer[s][l])/(1.0e+9);
        wu_comp_gflops_per_thread[s][l] = wu_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l]);
#endif
                }
	        else{ //CONV
		     int num_kernels = problem_size(nkernels[l],procs[l],rank);
                     //int num_channels = channels[l];//printf("Error. Not yet implemented\n"); return -1;
#ifdef TIMER
			wu_comp_timer[s][l] = MPI_Wtime();
#endif
			//We swap f and o because the kernels are stored in f
		        CONV_wu(num_kernels, BATCH_SIZE, image_size[l], image_size[l],  kheight[l], kwidth[l], channels[l], conv_i, conv_ip, conv_o, conv_f, &wu_im2col_timer[s][l]);
                            	
#ifdef TIMER
	wu_comp_timer[s][l] = MPI_Wtime() - wu_comp_timer[s][l];
        MPI_Reduce(&wu_comp_timer[s][l], &wu_comp_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        MPI_Reduce(&wu_im2col_timer[s][l], &wu_im2col_timer[s][l], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[l]);
        int m= nkernels[l];
	int n= BATCH_SIZE*image_size[l]*image_size[l];
	int k= channels[l]*kheight[l]*kwidth[l];
        //wu_comp_timer[s][l] = wu_comp_timer[s][l]/procs[l];
        wu_comp_gflops[s][l] = (2.0*m*n*k/wu_comp_timer[s][l])/(1.0e+9);
        wu_comp_gflops_per_thread[s][l] = wu_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l]);
#endif
                }
            }
        }
        MPI_Barrier(max_procs_comm);
        /* Now we compute the BP of the first layer */
        //CG
        int fl = 1;
	//printf("CG layer %d\n",fl);
        if(type[fl] == FC){ //FC
            if(rank < procs[fl+1]){
                int m = nneurons[fl];
                int n = BATCH_SIZE;
                int k = problem_size(nneurons[fl+1],procs[fl+1],rank);
                int lda = nneurons[fl+1];//m; 
                int ldb = nneurons[fl+1];
                int ldc = m;
#ifdef TIMER
	cg_comp_timer[s][fl] = MPI_Wtime();
#endif
                FC_gemm_cg( m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc );
#ifdef TIMER
	cg_comp_timer[s][fl] = MPI_Wtime() - cg_comp_timer[s][fl];
        MPI_Reduce(&cg_comp_timer[s][fl], &cg_comp_timer[s][fl], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[fl+1]);
        //cg_comp_timer[s][fl] = cg_comp_timer[s][fl]/procs[fl+1];
        k = nneurons[fl+1];
	cg_comp_gflops[s][fl] = (2.0*m*n*k/cg_comp_timer[s][fl])/(1.0e+9);
        cg_comp_gflops_per_thread[s][fl] = cg_comp_gflops[s][fl]/(1.0*OMP_NUM_THREADS*procs[fl+1]);
#endif

                 int comm_size = (nneurons[fl]/procs[fl])*BATCH_SIZE;
#ifdef TIMER
	cg_comm_timer_red[s][fl] = MPI_Wtime();
#endif
                 allreduce(comm_size, matrix_C, communicators[fl+1]);
#ifdef TIMER
	cg_comm_timer_red[s][fl] = MPI_Wtime() - cg_comm_timer_red[s][fl];
	cg_comm_timer_bcast[s][fl] = 0;
#endif
             }
         }
         else{ //CONV
	     if(rank < procs[fl]){
                 int num_kernels = problem_size(nkernels[fl],procs[fl],rank);
#ifdef TIMER
			cg_comp_timer[s][fl] = MPI_Wtime();
#endif           
                 //We swap input and output
		 CONV_cg(num_kernels, BATCH_SIZE, (type[fl+1] == CONV) ? image_size[fl+1]: nneurons[fl+1], (type[fl+1] == CONV) ? image_size[fl+1] : 1, kheight[fl], kwidth[fl], channels[fl+1], conv_i, conv_ip, conv_o, conv_f, &cg_im2col_timer[s][fl]);
                            	
#ifdef TIMER
	cg_comp_timer[s][fl] = MPI_Wtime() - cg_comp_timer[s][fl];
        MPI_Reduce(&cg_comp_timer[s][fl], &cg_comp_timer[s][fl], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[fl]);
        MPI_Reduce(&cg_im2col_timer[s][fl], &cg_im2col_timer[s][fl], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[fl]);
        int m= nkernels[fl];
	int aux = (type[fl+1] == CONV) ? image_size[fl+1]: nneurons[fl+1];
	int aux2 = (type[fl+1] == CONV) ? image_size[fl+1] : 1;
	int n= BATCH_SIZE*aux*aux2;
	int k= channels[fl]*kheight[fl]*kwidth[fl];
        //cg_comp_timer[s][fl] = cg_comp_timer[s][fl]/procs[fl];
        cg_comp_gflops[s][fl] = (2.0*m*n*k/cg_comp_timer[s][fl])/(1.0e+9);
        cg_comp_gflops_per_thread[s][fl] = cg_comp_gflops[s][fl]/(1.0*OMP_NUM_THREADS*procs[fl]);
#endif
                 int comm_size =image_size[l-1]*image_size[l-1] *BATCH_SIZE;
#ifdef TIMER
	cg_comm_timer_red[s][fl] = MPI_Wtime();
        //printf("pillo tiempo\n");
#endif
		 allreduce(comm_size, conv_i, communicators[fl]);
#ifdef TIMER
	cg_comm_timer_red[s][fl] = MPI_Wtime() - cg_comm_timer_red[s][fl];
        cg_comm_timer_bcast[s][fl] = 0;
#endif
             }
         }
         //WU
	//printf("WU layer %d\n",fl);
         if(rank < procs[fl]){
             if(type[fl] == FC){ //FC
                 int m = problem_size(nneurons[fl],procs[fl],rank);
                 int n = nneurons[fl-1];
                 int k = BATCH_SIZE;
                 int lda = m;
                 int ldb = n;
                 int ldc = nneurons[fl];
#ifdef TIMER
	wu_comp_timer[s][fl] = MPI_Wtime();
#endif
                 FC_gemm_wu( m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc);
#ifdef TIMER
	wu_comp_timer[s][fl] = MPI_Wtime() - wu_comp_timer[s][fl];
        MPI_Reduce(&wu_comp_timer[s][fl], &wu_comp_timer[s][fl], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[fl]);
        //wu_comp_timer[s][fl] = wu_comp_timer[s][fl]/procs[fl];
        m = nneurons[fl];
	wu_comp_gflops[s][fl] = (2.0*m*n*k/wu_comp_timer[s][fl])/(1.0e+9);
        wu_comp_gflops_per_thread[s][fl] = wu_comp_gflops[s][fl]/(1.0*OMP_NUM_THREADS*procs[fl]);
#endif
             }
             else{
		 int num_kernels = problem_size(nkernels[fl],procs[fl],rank);
#ifdef TIMER
			wu_comp_timer[s][fl] = MPI_Wtime();
#endif
		        CONV_wu(num_kernels, BATCH_SIZE, image_size[fl], image_size[fl],  kheight[fl], kwidth[fl], channels[fl], conv_i, conv_ip, conv_o, conv_f, &wu_im2col_timer[s][fl]);
                            	
#ifdef TIMER
	wu_comp_timer[s][fl] = MPI_Wtime() - wu_comp_timer[s][fl];
        MPI_Reduce(&wu_comp_timer[s][fl], &wu_comp_timer[s][fl], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[fl]);
        MPI_Reduce(&wu_im2col_timer[s][fl], &wu_im2col_timer[s][fl], 1, MPI_DOUBLE, MPI_MAX, 0,communicators[fl]);
        int m= nkernels[fl];
	int n= BATCH_SIZE*image_size[fl]*image_size[fl];
	int k= channels[fl]*kheight[fl]*kwidth[fl];
        //wu_comp_timer[s][fl] = wu_comp_timer[s][fl]/procs[fl];
        wu_comp_gflops[s][fl] = (2.0*m*n*k/wu_comp_timer[s][fl])/(1.0e+9);
        wu_comp_gflops_per_thread[s][fl] = wu_comp_gflops[s][fl]/(1.0*OMP_NUM_THREADS*procs[fl]);
#endif
             }
         }
        
        MPI_Barrier(max_procs_comm);

#ifdef TIMER
	step_timer[s] = MPI_Wtime() - step_timer[s] ;
#endif
    } //steps
#ifdef TIMER
#ifndef SUMMARY
    if (rank == 0){
        double total_time = 0.0;
        for ( s = 0; s < NUM_STEPS; s++){
	    printf("STEP %d\nTime %f\n",s, step_timer[s]);
	    printf("\t Initial Bcast %f\n", initial_bcast_timer[s]);
	    printf("\t **** FP ****\n");
	    for(l = 1; l < NUM_LAYERS; l++){
	        printf("\t Layer %d (type %s)\n", l, (type[l]==CONV)?"Conv":"FC");
	        printf("\t\t FP Computation time %f", fp_comp_timer[s][l]);
	        if(type[l] == CONV) printf(" (im2col = %f)",fp_im2col_timer[s][l]); 
	        printf(" | GFlops %f (Procs %d) GFlops/core %f (cores %d) ", fp_comp_gflops[s][l], procs[l], fp_comp_gflops_per_thread[s][l], procs[l]*OMP_NUM_THREADS);
	        printf(" Communication time Gather %f (Procs %d)| Bcast %f (Procs %d)\n", fp_comm_timer_red[s][l], procs[l], fp_comm_timer_bcast[s][l], (l == NUM_LAYERS-1) ? procs[l] : procs[l+1]);
            }
	    printf("\t **** BP ****\n");
	    for(l = NUM_LAYERS-1; l > 0; l--){
	        printf("\t Layer %d (type %s)\n", l, (type[l]==CONV)?"Conv":"FC");
	        printf("\t\t CG ");
	        printf(" Computation time %f", cg_comp_timer[s][l]);
	        if(type[l] == CONV) printf(" (im2col = %f)",cg_im2col_timer[s][l]); 
		printf(" | GFlops %f (Procs %d) GFlops/core %f (cores %d)|", cg_comp_gflops[s][l], ( l == NUM_LAYERS-1) ? procs[l] : procs[l+1], cg_comp_gflops_per_thread[s][l], OMP_NUM_THREADS * (( l == NUM_LAYERS-1) ? procs[l] : procs[l+1]));
	        printf(" Communication time Reduce %f (Procs %d)| Bcast %f (Procs %d)\n", cg_comm_timer_red[s][l], ( l == NUM_LAYERS-1) ? procs[l] : procs[l+1], cg_comm_timer_bcast[s][l], procs[l]);
	        printf("\t\t WU ");
	        printf(" Computation time %f",wu_comp_timer[s][l]);
	        if(type[l] == CONV) printf(" (im2col = %f)",wu_im2col_timer[s][l]); 
	        printf(" | GFlops %f (Procs %d) GFlops/core %f (cores %d)\n", wu_comp_gflops[s][l], procs[l], wu_comp_gflops_per_thread[s][l], procs[l]*OMP_NUM_THREADS);
	
            }
            total_time+=step_timer[s];
        }
	printf("Time of scatter = %f\n",scatter_time);
	printf("Time per step = %f\n",total_time/NUM_STEPS);
    }
#else

    if (rank == 0){
        double total_time_fp[NUM_LAYERS], total_time_comp_fp[NUM_LAYERS], total_time_comm_fp[NUM_LAYERS], total_time_bp[NUM_LAYERS], total_time_bp_cg[NUM_LAYERS],total_time_comp_bp_cg[NUM_LAYERS],total_time_comm_bp_cg[NUM_LAYERS], total_time_bp_wu[NUM_LAYERS];
	for (l=1; l<NUM_LAYERS;l++){
		total_time_fp[l] = 0;
		total_time_comp_fp[l] = 0;
		total_time_comm_fp[l] = 0;
		total_time_bp[l] = 0;
		total_time_bp_cg[l] = 0;
		total_time_comp_bp_cg[l] = 0;
		total_time_comm_bp_cg[l] = 0;
		total_time_bp_wu[l] = 0;
		
        }
        /*printf("************FP*********\n");
        printf("#step");
        for (l=1; l<NUM_LAYERS;l++){
            printf(" layer_%d",l);
        }
        printf("\n");
        for ( s = 1; s < NUM_STEPS; s++){
             printf("%d ",s);
             for (l=1; l<NUM_LAYERS;l++){
                 printf(" %f",fp_comp_timer[s][l] + fp_comm_timer_red[s][l] + fp_comm_timer_bcast[s][l]);
             }
            printf("\n");
        }
        printf("\n");
        printf("*********BP***********\n");
        printf("#step");
        for (l=1; l<NUM_LAYERS;l++){
            printf(" layer_%d",l);
        }
        printf("\n");
        for ( s = 1; s < NUM_STEPS; s++){
             printf("%d ",s);
             for (l=1; l<NUM_LAYERS;l++){
                 printf(" %f", cg_comp_timer[s][l] + cg_comm_timer_red[s][l] + cg_comm_timer_bcast[s][l]+wu_comp_timer[s][l]);
             }
            printf("\n");
        }
            printf("\n");
*/
        for ( s = 1; s < NUM_STEPS; s++){
             for (l=1; l<NUM_LAYERS;l++){
		total_time_fp[l] += fp_comp_timer[s][l] + fp_comm_timer_red[s][l] + fp_comm_timer_bcast[s][l];
		total_time_comp_fp[l] += fp_comp_timer[s][l];
		total_time_comm_fp[l] += fp_comm_timer_red[s][l] + fp_comm_timer_bcast[s][l];
		total_time_bp_cg[l] += cg_comp_timer[s][l] + cg_comm_timer_red[s][l] + cg_comm_timer_bcast[s][l];
		total_time_comp_bp_cg[l] += cg_comp_timer[s][l];
		total_time_comm_bp_cg[l] += cg_comm_timer_red[s][l] + cg_comm_timer_bcast[s][l];
		total_time_bp_wu[l] += wu_comp_timer[s][l];
		total_time_bp[l] += total_time_bp_cg[l] + total_time_bp_wu[l];
             }
        }
        // SUMMARY PARA HUAWEI
       printf("#layer time_fp time_comp_fp time_comm_fp time_bp time_cg time_comp_cg time_comm_cg time_wu\n");
	for (l=1; l<NUM_LAYERS;l++){
		printf("%d %f %f %f %f %f %f %f %f\n",l,total_time_fp[l]/(NUM_STEPS-1),total_time_comp_fp[l]/(NUM_STEPS-1),total_time_comm_fp[l]/(NUM_STEPS-1), (total_time_bp_cg[l]/(NUM_STEPS-1)) + (total_time_bp_wu[l]/(NUM_STEPS-1)),total_time_bp_cg[l]/(NUM_STEPS-1),total_time_comp_bp_cg[l]/(NUM_STEPS-1),total_time_comm_bp_cg[l]/(NUM_STEPS-1), total_time_bp_wu[l]/(NUM_STEPS-1));//*/total_time_bp[l]/(NUM_STEPS-1));
        }

    }
#endif
#endif    
    } //if(rank<max_procs)

    MPI_Barrier(MPI_COMM_WORLD);

    free(matrix_A);
    free(matrix_B);
    free(matrix_C);
//    free(model);
  //  free(data);
    //free(ranks);
    MPI_Finalize();
    return 0;
}


void FC_gemm_fp( int m, int n, int k, float * A, int lda, float * B, int ldb, float * C , int ldc){
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				m,n,k, 1,
				A, lda, B, ldb,0,C,ldc);
}
void FC_gemm_cg( int m, int n, int k, float * A, int lda, float * B, int ldb, float * C , int ldc){
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
				m,n,k, 1,
				A, lda, B, ldb,0,C,ldc);
}

void FC_gemm_wu( int m, int n, int k, float * A, int lda, float * B, int ldb, float * C , int ldc){
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
				m,n,k, 1,
				A, lda, B, ldb,0,C,ldc);
}
void CONV_fp(int l,int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time){

// B batch size
// Input image of size H x W, with C channels
// K Kernels of size KH x KW each, with C channels
/*
float I[C][B][H][W];          // Input: C x (B H W)
float IP[C][KW][KH][B][H][W]; // Patched input: (C K_H K_W) x (B H W)
float O[K][B][H][W];          // Output: K x (B H W)
float F[K][C][KH][KW];        // Filter: K x (C K_H K_W)
*/

    int b,h,w,kh,kw,c;
#ifdef TIMER
    *time = MPI_Wtime();
#endif
    // Im2col: I -> IP

    int kk1 = KH*KW*B*H*W;
     int kk2 = KW*B*H*W;
     int kk3 = B*H*W;
     int kk4 = H*W;
     int kk5 = B*(H+KH)*(W+KW);
     int kk6 = (H+KH)*(W+KW);
     int kk7 = (W+KW);
     int jk1,ik1,ik2,jk2,jk3,jk4,ik3,ik4,ik5;

#pragma omp parallel for num_threads(CONV_THREADS) private(b,h,w,kh,kw,ik1,ik2,ik3,ik4,ik5,jk1,jk2,jk3,jk4)
for ( c = 0; c < C; c++ ) {
  ik1 = c*kk1;
  jk1 = c*kk5;
  for ( b = 0; b < B; b++ ) {
    ik2 = ik1 + b*kk4;
    jk2 = jk1 + b*kk6;
    for ( kh = 0; kh < KH; kh++ ) {
      ik3 = ik2 + kh*kk2;
      jk3 = jk2 + (h+kh)*kk7;
      for ( kw = 0; kw < KW; kw++ ) {
        ik4 = ik3 + kw * kk3;
        jk4 = jk3 + kw;
        for ( h = 0; h < H; h++ ) {
          ik5 = ik4 + h*W;
          for ( w = 0; w < W; w++ )
            IP[ ik5 + w ] = I[ jk4 + w ];
        }
      } 
    } 
  } 
} 
/*


    int kk1 = KH*KW*B*H*W;
     int kk2 = KW*B*H*W;
     int kk3 = B*H*W;
     int kk4 = H*W;
     int kk5 = B*(H+KH)*(W+KW);
     int kk6 = (H+KH)*(W+KW);
     int kk7 = (W+KW);
    #pragma omp parallel for num_threads(CONV_THREADS)  private(b,h,w,kh,kw)
    for ( c = 0; c < C; c++ ) 
        for ( b = 0; b < B; b++ )
             for ( kh = 0; kh < KH; kh++ )
                for ( kw = 0; kw < KW; kw++ )
                    for ( h = 0; h < H; h++ ) 
                        for ( w = 0; w < W; w++ ) 
                            IP[ c*kk1 + kh*kk2 + kw*kk3 + b*kk4 + h*W + w ] = I[ c*kk5 + b*kk6 + (h+kh)*kk7 + w + kw ];

*/
#ifdef TIMER
 *time = MPI_Wtime() - *time;
 /* char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
  double a = *time;    
  printf("%s Layer %d FP space=%f time=%f  GB/s = %f\n",processor_name,l,C*B*H*W*KH*KW*2.0,a,(C*B*H*W*KH*KW*2.0*4.0/1000000000.0)/a);
*/
#endif
    // Gemm
    int m = K;
    int n = B*H*W;
    int k = C*KH*KW;
    int lda = m; 
    int ldb = k;
    int ldc = m;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				m,n,k, 1,
				//miF, lda, miIP, ldb,0,miO,ldc);
				F, lda, IP, ldb,0,O,ldc);

   /*free(miI);
   free(miIP);
   free(miO);
   free(miF);*/
/*GEMM -> O = F * IP

        No transpose, No transpose

        alpha = 1.0

        beta  = 0.0
        m = K, n = (B H W), k = (C K_H K_W) 
*/
}
void CONV_cg(int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time){

// B batch size
// Input image of size H x W, with C channels
// K Kernels of size KH x KW each, with C channels
/*
float I[C][B][H][W];          // Input: C x (B H W)
float IP[C][KW][KH][B][H][W]; // Patched input: (C K_H K_W) x (B H W)
float O[K][B][H][W];          // Output: K x (B H W)
float F[K][C][KH][KW];        // Filter: K x (C K_H K_W)
*/

    int b,h,w,kh,kw,c;
#ifdef TIMER
    *time = MPI_Wtime();
#endif

    int kk1 = KH*KW*B*H*W;
     int kk2 = KW*B*H*W;
     int kk3 = B*H*W;
     int kk4 = H*W;
     int kk5 = B*(H+KH)*(W+KW);
     int kk6 = (H+KH)*(W+KW);
     int kk7 = (W+KW);
     int jk1,ik1,ik2,jk2,jk3,jk4,ik3,ik4,ik5;

#pragma omp parallel for num_threads(CONV_THREADS) private(b,h,w,kh,kw,ik1,ik2,ik3,ik4,ik5,jk1,jk2,jk3,jk4)
for ( c = 0; c < C; c++ ) {
  ik1 = c*kk1;
  jk1 = c*kk5;
  for ( b = 0; b < B; b++ ) {
    ik2 = ik1 + b*kk4;
    jk2 = jk1 + b*kk6;
    for ( kh = /*-KH/2*/0; kh < KH/*/2*/; kh++ ) {
      ik3 = ik2 + kh*kk2;
      jk3 = jk2 + (h+kh)*kk7;
      for ( kw = /*-KW/2*/0; kw < KW/*/2*/; kw++ ) {
        ik4 = ik3 + kw * kk3;
        jk4 = jk3 + kw;
        for ( h = 0; h < H; h++ ) {
          ik5 = ik4 + h*W;
          for ( w = 0; w < W; w++ )
            IP[ ik5 + w ] = I[ jk4 + w ];
        }
      } 
    } 
  } 
} 
#ifdef TIMER
  //printf("CG GB/s = %f\n",C*B*H*W*KH*KW*2.0*4.0/1000000000.0);
 *time = MPI_Wtime() - *time;    
#endif
 // Gemm
    int m = K;
    int n = B*H*W;
    int k = C*KH*KW;
    int lda = m; 
    int ldb = k;
    int ldc = m;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				m,n,k, 1,
				F, lda, IP, ldb,0,O,ldc);
//	printf("Conv_fp not yet implemented\n");
}
void CONV_wu(int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time){
    int b,h,w,kh,kw,c;

#ifdef TIMER
 *time = MPI_Wtime();    
#endif
    // Im2col: I -> IP
    int kk1 = KH*KW*B*H*W;
     int kk2 = KW*B*H*W;
     int kk3 = B*H*W;
     int kk4 = H*W;
     int kk5 = B*(H+KH)*(W+KW);
     int kk6 = (H+KH)*(W+KW);
     int kk7 = (W+KW);
     int jk1,ik1,ik2,jk2,jk3,jk4,ik3,ik4,ik5;

#pragma omp parallel for num_threads(CONV_THREADS) private(b,h,w,kh,kw,ik1,ik2,ik3,ik4,ik5,jk1,jk2,jk3,jk4)
for ( c = 0; c < C; c++ ) {
  ik1 = c*kk1;
  jk1 = c*kk5;
  for ( b = 0; b < B; b++ ) {
    ik2 = ik1 + b*kk4;
    jk2 = jk1 + b*kk6;
    for ( kh = /*-KH/2*/0; kh < KH/*/2*/; kh++ ) {
      ik3 = ik2 + kh*kk2;
      jk3 = jk2 + (h+kh)*kk7;
      for ( kw = /*-KW/2*/0; kw < KW/*/2*/; kw++ ) {
        ik4 = ik3 + kw * kk3;
        jk4 = jk3 + kw;
        for ( h = 0; h < H; h++ ) {
          ik5 = ik4 + h*W;
          for ( w = 0; w < W; w++ )
            IP[ ik5 + w ] = I[ jk4 + w ];
        }
      } 
    } 
  } 
} 
#ifdef TIMER
//  printf("WU GB/s = %f\n",C*B*H*W*KH*KW*2.0*4.0/1000000000.0);
 *time = MPI_Wtime() - *time;    
#endif
    // Gemm
    int m = K;
    int n = B*H*W;
    int k = C*KH*KW;
    int lda = m; 
    int ldb = k;
    int ldc = m;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				m,n,k, 1,
				F, lda, IP, ldb,0,O,ldc);
}

void allgather(int n, float * C, MPI_Comm comm){
#ifndef NOCOM    
	MPI_Allgather(C, n,MPI_FLOAT, C, n, MPI_FLOAT, comm);
#endif
} 

void gather(int n, float * C, MPI_Comm comm){
#ifndef NOCOM    
    MPI_Gather(C, n, MPI_FLOAT, C, n, MPI_FLOAT, 0, comm);
#endif
} 

void allreduce(int n, float * C, MPI_Comm comm){
#ifndef NOCOM    
    MPI_Allreduce(C, C, n, MPI_FLOAT, MPI_SUM, comm);
#endif
} 

void reduce(int n, float * C, MPI_Comm comm){
#ifndef NOCOM    
    MPI_Reduce(C, C, n, MPI_FLOAT, MPI_SUM, 0,comm);
#endif
} 

void bcast (int n, float * data, MPI_Comm comm){
#ifndef NOCOM
    MPI_Bcast(data, n, MPI_FLOAT, 0, comm);
#endif
}

void scatter (size_t n, float * buff, MPI_Comm comm){
#ifndef NOCOM    
    MPI_Scatter(buff, n, MPI_FLOAT, buff, n, MPI_FLOAT, 0, comm);
#endif
}

int problem_size(int elements, int nprocs, int rank){
    int part = elements/nprocs;
    int rest = elements%nprocs;
    if (rest > 0){
	if (rank<rest){
	    part++;
        }
    }
    return part;
}
