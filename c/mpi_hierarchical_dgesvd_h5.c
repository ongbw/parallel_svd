#include <stdlib.h>
#include <stdio.h>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "mpi.h"
#include "math.h"
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"

#define INDEX 100

extern void test_fail(char *file, int line, char *call, int retval);

/* DGESVD prototype */
extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );
/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );

/* Main program */


int main(int argc, char *argv[])
{

  /* this code assumes that we have nprocs processors available to
     merge M=nprocs blocks of data, where each block is stored in an
     enumerated hdf5 file.

     Code assumes that M (total number of sketches to merge) is some 
     power of the number of sketches, N, to merge at a time, i.e.,
     M = n^p, where p is some integer
   */


  /******************* MPI Initialization / Read in files on each process **********/

  extern void dummy(void *);
  float real_time, proc_time, mflops;
  long long flpins;
  int retval;

  int nprocs, proc_id;
  int provided;

  MPI_Init (&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

  int M = nprocs; // total number of initial blocks

  int N; // number of sketches to merge at each time

  
  if (argc != 2) {
    if (proc_id == 0) {
      printf("[error] usage: ./executable  nsketches\n\n");
      fflush(stdout);
    }
    MPI_Finalize();
    exit(1);
    
  } else {
    N = atoi(argv[1]);
  }

  /* Setup PAPI library and begin collecting data from the counters */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_flops", retval);


  int P = (int)ceil(log((double) M)/log((double) N));  // number of levels;

  // if num_procs = 1, just take SVD of input matrix
  if (M == 1)
    P = 0;

  // some rounding errors here ...
  if (pow(N,P) != M)
    {
      if (proc_id == 0) {
	printf("[error]: total number of sketches to merge, %d, is not a power of specified number of sketches to merge at each time, %d.\n  ...Exiting\n\n",M,N);
	fflush(stdout);
      }
      MPI_Finalize();
      exit(1);
    }


    
  if (proc_id == 0) {
    printf("P = %d\n",P);
  }



  
  /* Specify name of data file based on proc id*/
  char FILE[16];
  sprintf(FILE,"matrix_%d_%d.h5",nprocs,proc_id+1);

    
  hid_t       file_id;
  double      * a;
  hsize_t     dims[2];
  size_t      nrow, n_values;
  
  /* open file */
  file_id = H5Fopen (FILE, H5F_ACC_RDONLY, H5P_DEFAULT);

  /* get the dimensions of the dataset */
  H5LTget_dataset_info(file_id,"/TestSet",dims,NULL,NULL);

  /* Initialize matrix for local SVD */
  a = (double *)malloc(sizeof(double)*dims[0]*dims[1]);
  for (int i = 0; i < dims[0]*dims[1]; i++) {
    a[i] = 0.0;
  }  
  
  /* read dataset */
  H5LTread_dataset_double(file_id,"/TestSet",a);

  /* close file */
  H5Fclose (file_id);

  /* information about data read in */
  int nrows, ncols;
  nrows = (size_t)dims[1];
  ncols = (size_t)dims[0];
  

  /* Local variables for DGESVD  */
  int m = nrows;
  int n = ncols;
  int lda = m;
  int ldu = m;
  int ldvt = n;
  
  /* singular values */
  double * s;

  /* left singular vectors */
  double * u;

  /* right singular vectors */
  double * vt;



  /* variables for SVD */
  double wkopt;
  double* work;
  int lwork, info;



  /******************* Loop over hierarchical levels, as appropriate  **********/

  
  for (int p = 0; p<P; p++) {
    
    // number of workers computing simultaneously this hierarchy level
    int num_workers = pow(N,P-p);
    
    if (proc_id < num_workers) {
      //printf("proc id %d on level %d \n", proc_id, p);
      
      
      s = (double *)malloc(sizeof(double)*m);
      for (int i = 0; i < m; i++) {
	s[i] = 0.0;
      }
      
      u = (double *)malloc(sizeof(double)*m*m);
      for (int i = 0; i < m*m; i++) {
	u[i] = 0.0;
      }

      /*
      vt = (double *)malloc(sizeof(double)*n*n);
      for (int i = 0; i < n*n; i++) {
	vt[i] = 0.0;
      }
      */
      
      // locally query optimal workspace 
      lwork = -1;
      dgesvd( "All", "N", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
	      &info );
      lwork = (int)wkopt;
      work = (double*)malloc( lwork*sizeof(double) );
      for (int i =0; i<lwork; i++) {
	work[i] = 0.0;
      }

      // locally compute SVD 
      dgesvd( "All", "N", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
	      &info );

      // locally Check for convergence 
      if( info > 0 ) {
	printf( "The algorithm computing SVD failed to converge.\n" );
	exit( 1 );
      }
      
      /* each worker computes U*S.  Assume intrinsic dimension not known, use ambient dim */
      double * us;
      us = (double *)malloc(sizeof(double)*m*m);
      
      // scale the singular vectors 
      for (int i =0; i < m; i++) {
	for (int j =0; j< m; j++) {
	  us[i+j*m] = u[i+j*m] * s[j];
	}
      }
      
      // create variable for proxy data 
      double ** proxy;
      
      
      // create variables for proxy SVD 
      int pm = m;
      int pn = N * m;
      int plda = m;
      int pldu = m;
      int pldvt = pn;
      
      // allocate space on next hierarichal level
      int masters =   pow(N,P-p-1);

      //printf("num masters = %d, level = %d \n",masters, p);
      
      if (proc_id < masters) {
	proxy = (double **) malloc(N *sizeof(double *) );
	for (int i =0; i<N; i++) {
	  proxy[i] = (double *) malloc(m*m*sizeof(double));
	}
      }

      free(a);
      
      if (proc_id < masters) {
	m = pm;
	n = pn;
	lda = plda;
	ldu = pldu;
	ldvt = pldvt;
	
	a = (double *)malloc(sizeof(double)*m*n);
      }


      MPI_Request send_req[N];
      MPI_Request recv_req[N];
      
      if (proc_id < masters) {
	for (int j = 1; j<N; j++) {
	  MPI_Recv(proxy[j], m*m, MPI_DOUBLE, j*masters+proc_id, p, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
      } else {
	MPI_Send(us, m*m, MPI_DOUBLE, proc_id%masters, p, MPI_COMM_WORLD);
      }

      
      if (proc_id < masters) {

	// copy it's own U*S into a
	for (int i =0; i< m*m; i++) {
	  a[i] = us[i];
	}

	for (int j = 1; j< N; j++) {

	  for (int i =0; i < m*m; i++) {
	    a[j*m*m + i] = proxy[j][i];
	  }
	}
      }


      if (proc_id < masters) {
	/* Print left Proxy Matrix */
	//print_matrix( "Proxy Matrix", m, N*m, a, ldu );

	for (int i =0; i<N; i++) {
	  free(proxy[i]);
	}
	free(proxy);
      }
	
      free(us);
      free(s);
      free(u);
      //free(vt);
      
    }  // if worker is needed this loop
  } // for loop for number of hierarichal levels

  //MPI_Barrier(MPI_COMM_WORLD);
  
  /************ compute SVD of final level, A^{p,0} *************/
  
  if (proc_id  == 0) {
    
    s = (double *)malloc(sizeof(double)*m);
    for (int i = 0; i < m; i++) {
      s[i] = 0.0;
    }
    
    u = (double *)malloc(sizeof(double)*m*m);
    for (int i = 0; i < m*m; i++) {
      u[i] = 0.0;
    }
    
    /*
    vt = (double *)malloc(sizeof(double)*n*n);
    for (int i = 0; i < n*n; i++) {
      vt[i] = 0.0;
    }
    */
    
    
    /* locally query optimal workspace */
    lwork = -1;
    dgesvd( "All", "N", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
	    &info );
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    for (int i =0; i<lwork; i++) {
      work[i] = 0.0;
    }
    
    /* locally compute SVD */
    dgesvd( "All", "N", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
	    &info );
    
    /* locally Check for convergence */
    if( info > 0 ) {
      printf( "The algorithm computing SVD failed to converge.\n" );
      exit( 1 );
    }
    
    /* print out singular values */
    //print_matrix( "Singular values", 1, m, s, 1 );
    
    /* Print left singular vectors */
    //print_matrix( "Left singular vectors (stored columnwise)", m, m, u, ldu );
    
    /* free workspace */
    free(s);
    free(u);
    //free(vt);
    free(work);
    
  } // END final svd

    /* Collect the data into the variables passed in */
    if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

    printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
	   real_time, proc_time, flpins, mflops);
    printf("%s\tPASSED\n", __FILE__);
    PAPI_shutdown();

    

  
  MPI_Finalize();
  return 0;
  
  
} // END main


    
/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
  int i, j;
  printf( "\n %s\n", desc );
  for( i = 0; i < m; i++ ) {
    for( j = 0; j < n; j++ ) {
      printf( " %6.2f", a[i+j*lda] );
    }
    printf( "\n" );
  }
}

static void test_fail(char *file, int line, char *call, int retval){
  printf("%s\tFAILED\nLine # %d\n", file, line);
  if ( retval == PAPI_ESYS ) {
    char buf[128];
    memset( buf, '\0', sizeof(buf) );
    sprintf(buf, "System error in %s:", call );
    perror(buf);
  }
  else if ( retval > 0 ) {
    printf("Error calculating: %s\n", call );
  }
  else {
    printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
  }
  printf("\n");
  exit(1);
}
