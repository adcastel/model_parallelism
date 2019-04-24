/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2011 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char *argv[])
{
    int size, rank, i, *incl, *excl, sizes[2];
    MPI_Group world_group, groups[2];
    MPI_Comm comms[2];

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size % 2) {
        printf("this program requires a multiple of 2 number of processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    excl = malloc((size / 4) * sizeof(int));
    incl = malloc((size / 2) * sizeof(int));
 //   assert(excl);
    sizes[0] = size/2;
    sizes[1] = size/4;
    /* exclude the odd ranks */
    for (i = 0; i < sizes[0]; i++)
        incl[i] = i;
    for (i = 0; i < sizes[1]; i++)
        excl[i] = i;

    /* Create some groups */
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, sizes[0], incl, &groups[0]);
    MPI_Group_incl(world_group, sizes[1], incl, &groups[1]);
        
    MPI_Group_free(&world_group);

    if (rank < sizes[0]) {
        /* Even processes create a group for themselves */
        MPI_Comm_create_group(MPI_COMM_WORLD, groups[0], 0, &comms[0]);
        int gsize,grank;
        MPI_Comm_size(comms[0], &gsize);
        MPI_Comm_rank(comms[0], &grank);
        printf("my rank in WORLD is %d/%d and in comm[0] is %d/%d\n",rank,size,grank,gsize);
        MPI_Barrier(comms[0]);
        MPI_Comm_free(&comms[0]);
    }

    MPI_Group_free(&groups[0]);
    if (rank < sizes[1]) {
        /* Even processes create a group for themselves */
        MPI_Comm_create_group(MPI_COMM_WORLD, groups[1], 0, &comms[1]);
        int gsize,grank;
        MPI_Comm_size(comms[1], &gsize);
        MPI_Comm_rank(comms[1], &grank);
        printf("my rank in WORLD is %d/%d and in comm[1] is %d/%d\n",rank,size,grank,gsize);
        MPI_Barrier(comms[1]);
        MPI_Comm_free(&comms[1]);
    }

    MPI_Group_free(&groups[1]);
    MPI_Barrier(MPI_COMM_WORLD);

    free(incl);
    MPI_Finalize();
    return 0;
}
