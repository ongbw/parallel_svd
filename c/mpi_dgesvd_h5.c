#include <stdlib.h>
#include <stdio.h>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "mpi.h"


/* DGESVD prototype */
extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );
/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );




/* Main program */


int main(int argc, char *argv[])
{

  int nprocs, proc_id;
  int provided;



  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

  char FILE[64];
  sprintf(FILE,"matrix_%d.h5",proc_id+1);

    
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


  s = (double *)malloc(sizeof(double)*m);
  for (int i = 0; i < m; i++) {
    s[i] = 0.0;
  }
  
  u = (double *)malloc(sizeof(double)*m*m);
  for (int i = 0; i < m*m; i++) {
    u[i] = 0.0;
  }
  
  vt = (double *)malloc(sizeof(double)*n*n);
  for (int i = 0; i < n*n; i++) {
    vt[i] = 0.0;
  }
  
  
  
  /* temporary variables*/
  double wkopt;
  double* work;
  int lwork, info;

  /* locally query optimal workspace */
  lwork = -1;
  dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
	  &info );
  lwork = (int)wkopt;
  work = (double*)malloc( lwork*sizeof(double) );
  for (int i =0; i<lwork; i++) {
    work[i] = 0.0;
  }
  
  /* locally compute SVD */
  dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
	  &info );

  /* locally Check for convergence */
  if( info > 0 ) {
    printf( "The algorithm computing SVD failed to converge.\n" );
    exit( 1 );
  }
	
  /* free workspace */
  free( (void*)work );


  /* each worker computes U*S.  Assume intrinsic dimension not known, use ambient dim */
  double * us;
  us = (double *)malloc(sizeof(double)*m*m);

  for (int i =0; i < m; i++) {
    for (int j =0; j< m; j++) {
      us[i+j*m] = u[i+j*m] * s[j];
    }
  }
 

  /* print out singular values */
  print_matrix( "Singular values", 1, m, s, 1 );

  /* Print left singular vectors */
  print_matrix( "Left singular vectors (stored columnwise)", m, m, u, ldu );
  print_matrix( "scaled left singular vectors (stored columnwise)", m, m, us, ldu );

  /* Print right singular vectors */
  print_matrix( "Right singular vectors (stored rowwise)", m, n, vt, ldvt );

  /* create variable for proxy data */
  double * proxy;


  /* create variables for proxy SVD */
  int pm = m;
  int pn = nprocs * m;
  int plda = m;
  int pldu = m;
  int pldvt = pn;
  
  /* allocate space on master process and initialize */
  
  if (proc_id ==0) {
    proxy = (double *)malloc(sizeof(double)*m*pn);
    for (int i =0; i<m*pn; i++) {
      proxy[i] = 0.0;
    }
  }
  
  
  /* gather proxy matrix on master node */

  MPI_Gather( us, m*m, MPI_DOUBLE, proxy, m*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(us);
  free(s);
  free(u);
  free(vt);
  free(a);

  // print proxy svd
  if (proc_id ==0)
    {
      printf("proxy matrix:\n");
      for (int i=0; i<pm; i++) {
	for (int j =0; j<pn; j++) {
	  printf(" %8.4f",proxy[i + j*nrows]);
	}
	printf("\n");
      }
  
      /* create variables for proxy svd */

      
      double * ps;
      ps = (double *)malloc(sizeof(double)*pm);
      for (int i = 0; i < pm; i++) {
	ps[i] = 0.0;
      }
  
      double * pu;
      pu = (double *)malloc(sizeof(double)*pm*pm);
      for (int i = 0; i < pm*pm; i++) {
	pu[i] = 0.0;
      }
  
      double * pvt;
      pvt = (double *)malloc(sizeof(double)*pn*pn);
      for (int i = 0; i < pn*pn; i++) {
	pvt[i] = 0.0;
      }
      
  
      /* find SVD of proxy  matrix */
      
      /* locally query optimal workspace */

      lwork = -1;
      dgesvd( "All", "All", &pm, &pn, proxy, &plda, ps, pu, &pldu, pvt, &pldvt, &wkopt, &lwork,
	      &info );
      lwork = (int)wkopt;
      work = (double*)malloc( lwork*sizeof(double) );

      /* locally compute SVD */

      dgesvd( "All", "All", &pm, &pn, proxy, &plda, ps, pu, &pldu, pvt, &pldvt, work, &lwork,
	      &info );

      /* print out singular values */
      print_matrix( "Proxy Singular values", 1, pm, ps, 1 );
      
    
      /* Print left singular vectors */
      print_matrix( "Proxy Left singular vectors (stored columnwise)", pm, pm, pu, pldu );
      
      /* Print right singular vectors */
      print_matrix( "Proxy Right singular vectors (stored rowwise)", pm, pn, pvt, pldvt );

      /* free workspace */

      free( (void*)proxy);
      free( (void*)ps);
      free( (void*)pu);
      free( (void*)pvt);
      free(work);
  
    }
  
  MPI_Finalize();
  return 0;

  
  
  
}


    
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
