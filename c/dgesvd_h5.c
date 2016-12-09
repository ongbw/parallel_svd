#include <stdlib.h>
#include <stdio.h>
#include "hdf5.h"
#include "hdf5_hl.h"
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"

#define INDEX 100



/* DGESVD prototype */
extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );
/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );

/* specify HDF5 file */
#define FILE "matrix_1_1.h5"
#define DATASET "TestSet"

/* papi function */
extern void test_fail(char *file, int line, char *call, int retval);

/* Main program */
int main() {


  extern void dummy(void *);
  float real_time, proc_time, mflops;
  long long flpins;
  int retval;

  /* Setup PAPI library and begin collecting data from the counters */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

  hid_t       file_id;
  double      * a;
  hsize_t     dims[2];
  size_t      nrows, ncols, n_values;
  
  /* open file from ex_lite1.c */
  file_id = H5Fopen (FILE, H5F_ACC_RDONLY, H5P_DEFAULT);

  /* get the dimensions of the dataset */
  H5LTget_dataset_info(file_id,"/TestSet",dims,NULL,NULL);
  
  a = (double *)malloc(sizeof(double)*dims[0]*dims[1]);
  for (int i = 0; i < dims[0]*dims[1]; i++) {
    a[i] = 0.0;
  }  
  /* read dataset */
  H5LTread_dataset_double(file_id,"/TestSet",a);


  /* print vector that stores a */
  n_values = (size_t)(dims[0] * dims[1]);
  /*
  printf("values read in from hdf5 file: \n");
  for (int i =0; i<n_values; i++) {
    printf (" %f", a[i]);
  }
  printf("\n");
  */

  nrows = (size_t)dims[1];
  ncols = (size_t)dims[0];

  /*  
  printf("Finding the SVD of the following matrix:\n");

  for (int i=0; i<nrows; i++) {
    for (int j =0; j<ncols; j++) {
      printf(" %8.4f",a[i + j*nrows]);
    }
    printf("\n");
  }
  */

  /* close file */
  H5Fclose (file_id);


  /* Locals */
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

  s = (double *)malloc(sizeof(double)*m);
  for (int i = 0; i < m; i++) {
    s[i] = 0.0;
  }
  
  u = (double *)malloc(sizeof(double)*m*m);
  for (int i = 0; i < m*m; i++) {
    u[i] = 0.0;
  }
  
  //try not allocating vt if you don't want to generate right singular vectors?
  
/*
  vt = (double *)malloc(sizeof(double)*n*n);
	
  for (int i = 0; i < n*n; i++) {
    vt[i] = 0.0;
  }
  */
  
  
  /* temporary variables*/
  double wkopt;
  double* work;
  int lwork, info;

  /* query optimal workspace */
  lwork = -1;
  dgesvd( "All", "N", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
	  &info );
  lwork = (int)wkopt;
  work = (double*)malloc( lwork*sizeof(double) );
  
  /* Compute SVD */
  dgesvd( "All", "N", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
	  &info );

  /* Check for convergence */
  if( info > 0 ) {
    printf( "The algorithm computing SVD failed to converge.\n" );
    exit( 1 );
  }
	
  /* print singular values */
  //print_matrix( "Singular values", 1, m, s, 1 );

  /* Print left singular vectors */
  //print_matrix( "Left singular vectors (stored columnwise)", m, m, u, ldu );

  /* Print right singular vectors */
  //print_matrix( "Right singular vectors (stored rowwise)", m, n, vt, ldvt );

  /* Free workspace */
  free( (void*)work );
  free( (void*)s);
  free( (void*)u);
  //free( (void*)vt);    
  free( (void*)a);    
  
  //free(a);

  /* Collect the data into the variables passed in */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

  printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
	 real_time, proc_time, flpins, mflops);
  printf("%s\tPASSED\n", __FILE__);
  PAPI_shutdown();


  return 0;

  
  
  
}


    
/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
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
