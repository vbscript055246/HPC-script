#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NRA 62                /* number of rows in matrix A */
#define NCA 15                /* number of columns in matrix A */
#define NCB 7                 /* number of columns in matrix B */

int main (int argc, char *argv[]) 
{
	double st, sp;
	int	tid, nthreads, i, j, k, chunk;
	double	a[NRA][NCA],           /* matrix A to be multiplied */
	b[NCA][NCB],           /* matrix B to be multiplied */
	c[NRA][NCB];           /* result matrix C */

chunk = 10;                    /* set loop iteration chunk size */

/*** Spawn a parallel region explicitly scoping all variables ***/
st = omp_get_wtime(); // timer start
#pragma omp parallel shared(a,b,c,nthreads,chunk) private(tid,i,j,k)
  {

  /*** Initialize matrices ***/
  #pragma omp for schedule (static, chunk) 
  for (i=0; i<NRA; i++)
    for (j=0; j<NCA; j++)
      a[i][j]= i+j;
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++)
      b[i][j]= i*j;
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NRA; i++)
    for (j=0; j<NCB; j++)
      c[i][j]= 0;
  
  #pragma omp for schedule (static, chunk)
  for (i=0; i<NRA; i++)    
    {
    for(j=0; j<NCB; j++)       
      for (k=0; k<NCA; k++)
        c[i][j] += a[i][k] * b[k][j];
    }
  }   /*** End of parallel region ***/
  sp = omp_get_wtime(); // timer stop

	printf ("Done.\n");
	printf("%lf s\n", sp - st);

}