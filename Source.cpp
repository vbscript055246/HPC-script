#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define M 62
#define O 7
#define N 15
#define chunk 5
class node{

public:
	node() {
		array[0][0] = 0;
		array[0][1] = 0;
		array[1][0] = 0;
		array[1][1] = 0;
	}

	node(double A, double B, double C, double D){
		array[0][0] = A;
		array[0][1] = B;
		array[1][0] = C;
		array[1][1] = D;		
	}

	node &operator *(node &other) {
		double C[2][2][2] = { 0 };
		int k, i, j;
		#pragma omp parallel shared(C, other) private(i, j, k)
		{
			#pragma omp for schedule (guided) 
			for (int k = 0; k < 2; ++k) {
				for (int i = 0; i < 2; ++i) {
					for (int j = 0; j < 2; ++j) {
							C[k][i][j] = array[i][j] * other.array[k][j];
					}
				}
			}

		}
		return *new node(C[0][0][0] + C[1][0][0], C[0][0][1] + C[1][0][1], C[0][1][0] + C[1][1][0], C[0][1][1] + C[1][1][1]);
						
	}

	void operator =(node &other) {
		array[0][0] = other.array[0][0];
		array[0][1] = other.array[0][1];
		array[1][0] = other.array[1][0];
		array[1][1] = other.array[1][1];
	}

	node &operator +=(node &other) {
		*this = *this + other;
		return *this;
	}

	node &operator +(node &other) {
		double C[2][2] = { 0 };
		int i, j;
		#pragma omp parallel shared(C, other) private(i, j)
		{
			#pragma omp for schedule (guided) 
				for (int i = 0; i < 2; ++i) {
					for (int j = 0; j < 2; ++j) {
						C[i][j] = array[i][j] + other.array[i][j];
					}
				}

		}
		return *new node(C[0][0],C[0][1],
						 C[1][0],C[1][1]);
	}

	void write(double *matrix, int i, int j,int col, int row) {
		*(matrix + i * N + j) = array[0][0];
		if (j+1 != row)
			*(matrix + i * N + j+1) = array[0][1];
		if (i + 1 != col)
			*(matrix + (i + 1) * N + j) = array[1][0];
		if (j + 1 != row && i + 1 != col)
			*(matrix + (i + 1) * N + j + 1) = array[1][1];
	}
	
	double array[2][2];
};

int main(int argc, char *argv[]) {

	double st, sp;
	
	int i, j, k, I, J;
	int m = (M & 1) ? M + 1 : M, 
		n = (N & 1) ? N + 1 : N, 
		o = (O & 1) ? O + 1 : O;

	int page_num = o / 2 + 1;
	int page_row = n / 2;
	int page_col = m / 2;

	double c[M][N];
	double** a,** b;

//=============================================

	/*** Initialize matrices ***/
	a = new double*[m];
	b = new double*[o];

#pragma omp parallel shared(i, j, m, o, n)
{
	for (i = 0; i < m; ++i)
		a[i] = new double[o];

	for (j = 0; j < o; ++j)
		b[j] = new double[n];

	#pragma omp barrier
}

#pragma omp parallel shared(a, b) private(i, j)
{
	#pragma omp for schedule(guided) 
		for (i = 0; i < m; ++i)
			for (j = 0; j < o; ++j)
				if (i == M || j == O)	a[i][j] = 0;
				else                    a[i][j] = i + j + 1;

	#pragma omp for schedule (guided)
		for (i = 0; i < o; ++i)
			for (j = 0; j < n; ++j)
				if (i == O || j == N)	b[i][j] = 0;
				else     				b[i][j] = i * j + 1;
	#pragma omp barrier
}


	st = omp_get_wtime(); // timer start

//           ready to start

	printf("ready to start :%lf\n", omp_get_wtime() - st);

// mul
#pragma omp parallel shared(page, o ,m, page_ptr, a, b) private(I, J, i, j)
{
	#pragma omp for schedule (guided) 
		
	#pragma omp barrier

}



	//sum
#pragma omp parallel shared(page, page_ptr, page_col, page_row) private(k, i, j)
{
	

	#pragma omp barrier
}
	printf("sum done :%lf\n", omp_get_wtime() - st);

	//combine
#pragma omp parallel shared(page, page_col, page_row, c) private(i, j)
{
	
	#pragma omp barrier
}
	printf("combine done :%lf\n", omp_get_wtime() - st);
	sp = omp_get_wtime();

//=============================================

	printf("%lf s\n", sp - st);
	//system("pause");
}