

#define T_ELEM	float
//#define T_ELEM	double

int inverseMatrixBLAS( T_ELEM *A , T_ELEM *C, int matrixRows, int bDebug = false);
int inverseMatrixBLAS( T_ELEM **A , T_ELEM **C , int matrixRows , int sizeBatch ,int bDebug = false );
