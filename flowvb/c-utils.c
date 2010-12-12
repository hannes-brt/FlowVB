#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NROW 1000000
#define NCOL 100
#define NROW_SMALL 5
#define NCOL_SMALL 5
#define PRECISION 1e-3

void normalize_logspace_matrix(size_t nrow, size_t ncol, 
			       size_t rowstride, size_t colstride,
			       double mat[]);
void normalize_logspace(double vec[], size_t stride, size_t ct);
double logsumexp(const double nums[], size_t stride, size_t ct);


int main(void)
{
     
     size_t i, j, rowstride = NCOL, colstride = 1, 
	  rowstride_small = NCOL_SMALL, colstride_small = 1;
     int all_ok = 1;
     double total, elapsed;
     static double mat1d[NROW * NCOL];
     double mat_small[NROW_SMALL * NCOL_SMALL];
     clock_t start, end;
     
     srand48(time(NULL));
     
     // First test with a very large matrix
     for (i = 0 ; i < NROW * NCOL ; i++)
     {
	  mat1d[i] = drand48();
     }
     
     start = clock();
     normalize_logspace_matrix(NROW, NCOL, rowstride, colstride, mat1d);
     end = clock();
     elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
     
     for (i = 0 ; i < NROW ; i++)
	{
	     total = 0;
	     
	     for (j = 0 ; j < NCOL ; j++)
	     {
		  total += exp(mat1d[i * NCOL + j]);
		  if (abs(total - 1.0) > PRECISION)
		    all_ok = 0;
	     }
	}
	
     if (all_ok) printf("All rows sum to 1.0 (with a precision of %3.2e)\n", 
			PRECISION);
     printf("Time to normalize a %dx%d matrix: %5.4fs\n", 
	    NROW, NCOL, elapsed);
	

     // Now test with a small matrix
     for (i = 0 ; i < NROW_SMALL * NCOL_SMALL ; i++)
	  mat_small[i] = drand48();

     normalize_logspace_matrix(NROW_SMALL, NCOL_SMALL, 
			       rowstride_small, colstride_small,
			       mat_small);

     for (j = 0 ; j < NCOL_SMALL ; j++)
	  printf("%5d ", (int) j);

     printf("%5s\n", "Sum");

     for (i = 0 ; i < NROW_SMALL ; i++)
     {
	  total = 0;
	  
	  for (j = 0 ; j < NCOL_SMALL ; j++)
	  {
	       total += mat_small[i * rowstride_small + 
				  j * colstride_small];
	       
	       printf("%.3f ", mat_small[i *rowstride_small +
					 j * colstride_small]);	       
	  }
	  
	  printf("%.3f\n", total); 
     }

     return 0;

}

void normalize_logspace_matrix(size_t nrow, size_t ncol, 
			       size_t rowstride, size_t colstride,
			       double mat[])
{
     size_t i;
	
     for (i = 0 ; i < nrow ; i++)
	  normalize_logspace(&mat[i * rowstride], colstride, ncol);
	
}

void normalize_logspace(double vec[], size_t stride, size_t len)
{
     size_t i;
     double L;
     
     L = logsumexp(vec, stride, len);
	
     for (i = 0 ; i < len ; i++)
	  vec[i * stride] = exp(vec[i * stride] - L);
     
}


double logsumexp(const double nums[], size_t stride, size_t len) 
{
     double max_exp = nums[0], sum = 0.0;
     size_t i;
     
     for (i = 1 ; i < len ; i++)
	  if (nums[i * stride] > max_exp)
	       max_exp = nums[i * stride];
     
     for (i = 0; i < len ; i++)
	  sum += exp(nums[i * stride] - max_exp);
     
     return log(sum) + max_exp;
}
