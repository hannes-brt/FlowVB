#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NROW 1000000
#define NCOL 100
#define PRECISION 1e-8

void normalize_logspace_matrix(size_t nrow, size_t ncol, double mat[]);
void normalize_logspace(double vec[], size_t ct);
double logsumexp(const double nums[], size_t ct);


int main(void)
{

	size_t i, j;
	int all_ok = 1;
	double total, elapsed;
	static double mat1d[NROW * NCOL];
	clock_t start, end;

	srand48(time(NULL));
	
	for (i = 0 ; i < NROW * NCOL ; i++)
	  {
	    mat1d[i] = drand48();
	  }
	 
	start = clock();
	normalize_logspace_matrix(NROW, NCOL, mat1d);
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
	
	return 0;
}

void normalize_logspace_matrix(size_t nrow, size_t ncol, double mat[])
{
	size_t i;
	
	for (i = 0 ; i < nrow * ncol ; i += ncol)
		normalize_logspace(&mat[i], ncol);
	
}

void normalize_logspace_matrix_c(size_t nrow, size_t ncol, char * mat)
{
  normalize_logspace_matrix(nrow, ncol, (double*) mat);
}



void normalize_logspace(double vec[], size_t ct)
{
	size_t i;
	double L;
		
	L = logsumexp(vec, ct);
	
	for (i = 0 ; i < ct ; i++)
	  vec[i] = exp(vec[i] - L);
	
}


double logsumexp(const double nums[], size_t ct) 
{
	double max_exp = nums[0], sum = 0.0;
	size_t i;
	
	for (i = 1 ; i < ct ; i++)
		if (nums[i] > max_exp)
			max_exp = nums[i];

	for (i = 0; i < ct ; i++)
		sum += exp(nums[i] - max_exp);

	return log(sum) + max_exp;
}
