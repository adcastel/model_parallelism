#include <stdio.h>

#include "omp.h"
#include "mkl.h"
#include "mpi.h"






int main(int argc, char * argv[]){


float * I, *IP, *F, *O;
int b,h,w,kh,kw,c;
int size, rank;
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


int C=3, B=64, K=8, H= 224, W=224,KH=3,KW=3;


double time_im2col, time_gemm;

    time_im2col = MPI_Wtime();
I = (float *)malloc(64.0*C*B*H*W*sizeof(float));
IP = (float *)malloc(64.0*C*KW*KH*B*H*W*sizeof(float));
O = (float *)malloc(64.0*K*B*H*W*sizeof(float));
F = (float *)malloc(64.0*K*C*KH*KW*sizeof(float));
    // Im2col: I -> IPi
    int kk1 = KH*KW*B*H*W;
    int kk2 = KW*B*H*W;
    int kk3 = B*H*W;
    int kk4 = H*W;
    int kk5 = B*(H+KH)*(W+KW);
    int kk6 = (H+KH)*(W+KW);
    int kk7 = (W+KW);

    for ( c = 0; c < C; c++ )
    #pragma omp parallel for num_threads(8) private(h,w,kh,kw) firstprivate(c)
        for ( b = 0; b < B; b++ ) 
            for ( h = 0; h < H; h++ ) 
                for ( w = 0; w < W; w++ ) 
                    for ( kh = 0; kh < KH; kh++ )
                        for ( kw = 0; kw < KW; kw++ )
                            IP[ c*kk1 + kh*kk2 + kw*kk3 + b*kk4 + h*W + w ] = I[ c*kk5 + b*kk6 + (h+kh)*kk7 + w + kw ];
                            //IP[c]/*[kh][kw][b][h][w]*/ = I[c]/*[b][h+kh][w+kw]*/;
 time_im2col = MPI_Wtime() - time_im2col;
    // Gemm
    int m = K;
    int n = B*H*W;
    int k = C*KH*KW;
    int lda = m;
    int ldb = k;
    int ldc = m;
    time_gemm = MPI_Wtime();

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m,n,k, 1,
                                F, lda, IP, ldb,0,O,ldc);

 time_gemm = MPI_Wtime() - time_gemm;
  //free(I);
  //free(IP);

  printf("Rank %d Total time = %f (im2col = %f + gemm = %f) MEMBW = %f\n",rank,time_gemm+time_im2col,time_im2col, time_gemm, (2.0*C*B*H*W*KH*KW*4.0)/time_im2col/1.0e9);
MPI_Finalize();

return 0;
}



