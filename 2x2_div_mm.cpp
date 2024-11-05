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

	node ***page = new node**[page_num];

#pragma omp parallel shared(page)  private(i, j)
{
	#pragma omp for schedule(guided)
		for (i = 0; i < page_num; ++i)
			page[i] = new node*[page_col];

	#pragma omp for schedule (guided)
		for (i = 0; i < page_num; ++i)
			for (j = 0; j < page_col; ++j)
				page[i][j] = new node[page_row];
	#pragma omp barrier
}
//           ready to start

	printf("ready to start :%lf\n", omp_get_wtime() - st);
	int page_ptr = 0;

#pragma omp parallel shared(page, o ,m, page_ptr, a, b) private(I, J, i, j)
{
	#pragma omp for schedule (guided) 
		for (I = 0; I < o; I += 2) {
			for (J = 0; J < o; J += 2) {
				for (i = 0; i < m; i += 2) {

					node AM(a[i][I], a[i][I + 1], a[i + 1][I], a[i + 1][I + 1]);
					for (j = 0; j < m; j += 2) {
						node BM(b[J][j], b[J][j + 1], b[J + 1][j], b[J + 1][j + 1]);
						sp = omp_get_wtime();
						page[page_ptr][i / 2][j / 2] = AM * BM;
						printf("cal done :%lf\n", omp_get_wtime() - sp);
					}

				}
			}
			page_ptr++;
		}
	#pragma omp barrier

}

	printf("page done :%lf\n", omp_get_wtime() - st);


	//sum
#pragma omp parallel shared(page, page_ptr, page_col, page_row) private(k, i, j)
{
	#pragma omp for schedule(guided) 
		for (k = 0; k < page_ptr; k++) {
			for (i = 0; i < page_col; i++) {
				for (j = 0; j < page_row; j++) {
					page[page_num - 1][i][j] += page[k][i][j];
				}
			}
		}

	#pragma omp barrier
}
	printf("sum done :%lf\n", omp_get_wtime() - st);

	//combine
#pragma omp parallel shared(page, page_col, page_row, c) private(i, j)
{
	#pragma omp for schedule(guided) 
		for (i = 0; i < page_col; ++i) {
			for (j = 0; j < page_row; ++j) {
				page[page_num - 1][i][j].write((double *)c, i * 2, j * 2, M, N);
			}
		}
	#pragma omp barrier
}
	printf("combine done :%lf\n", omp_get_wtime() - st);
	sp = omp_get_wtime();

//=============================================

	printf("%lf s\n", sp - st);
	//system("pause");
}