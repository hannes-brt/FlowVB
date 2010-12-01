#import <math.h>
#import <stdio.h>

#define NROW 5
#define NCOL 5

void normalize_logspace_matrix(size_t nrow, size_t ncol, double mat[nrow][ncol]);
void normalize_logspace(double vec[], size_t ct);
double logsumexp(const double nums[], size_t ct);


int main(void)
{
	double mat[NROW][NCOL] = 
		{
			{3., 5., 1., 2., 3.},
			{2., 4., 3., 8., 1.},
			{23., 43., 29., 23., 123.},
			{43., 20., 129., 49., 12.},
			{90., 12., 92., 89., 12.}
		};

	size_t i, j;
	double total;
	
	
	normalize_logspace_matrix(NROW, NCOL, mat);

	for (j = 0 ; j < NCOL ; j++)
		printf("%5d ", (int) j);

	printf("%5s\n", "Sum");
	
	for (i = 0 ; i < NROW ; i++)
	{
		total = 0;

		for (j = 0 ; j < NCOL ; j++)
		{
			total += exp(mat[i][j]);

			printf("%.3f ", exp(mat[i][j]));
		}

		printf("%.3f\n", total); 
	}
	
	return 0;
}


void normalize_logspace_matrix(size_t nrow, size_t ncol, double mat[nrow][ncol])
{
	size_t i;
	
	for (i = 0 ; i < nrow ; i++)
		normalize_logspace(mat[i], ncol);
	
}



void normalize_logspace(double vec[], size_t ct)
{
	size_t i;
	double L;
		
	L = logsumexp(vec, ct);
	
	for (i = 0 ; i < ct ; i++)
		vec[i] = vec[i] - L;
	
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
