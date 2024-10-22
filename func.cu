#define N 10

__global__ void find_max(int* input, int* max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int old_max;

    if (i < n) {
        int val = input[i];

        // compare the maxValue with val to see if it is the current max
        old_max = atomicCAS(max, *max, val);

        // chance that some other thread updates maxValue above
        // after this thread, causing the oldMax value to change
        // so we check if this old max is smaller than current val, we update again
        while (old_max < val) {
            old_max = atomicCAS(max, old_max, val);
        }
    }
}

__global__ void find_min(int *input, int *min, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int old_max;

    if (i < n) {
        int val = input[i];

        // compare the maxValue with val to see if it is the current max
        old_min = atomicCAS(min, *min, val);

        // chance that some other thread updates maxValue above
        // after this thread, causing the oldMax value to change
        // so we check if this old max is smaller than current val, we update again
        while (old_min > val) {
            old_min = atomicCAS(min, old_min, val);
        }
    }
}

__global__ void vector_addition(int *a, int *b, int *c) {
    // blockIdx.x ranges from 0 to 255
    // threadIdx.x ranges from 0 to 3

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];  // each thread doing an addition
}

__device__ void fold(int *arr, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int stride;
    stride = 2;

    if(i<n) {
        int s;
        while(stride<=blockDim.x) {
            s = blockDim.x/stride;
            if(i<s) {
                arr[i] += arr[i + s];
            }
            __syncthreads();
            stride*=2;
        }

        if(threadIdx.x == 0) {  // thread 0 of each block acts as summary thread for that block
            *c = arr[0];
        }
    }
}

__global__ void dot_product(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int product[N];  // shared object between threads to store products

    if(i<n) {
        product[i] = a[i]*b[i];
    }
    else {
        product[i] = 0;
    }

    __syncthreads();  // barrier for all threads in a single thread to update their values

    // parallel folding
    fold(product, c, n);
}

__global__ void matrix_multiplication(int *a, int *b, int *c, int n) {
    int i = blockIdx.x;  // blockIndex
    int j = threadIdx.x;  // threadIndex in the ith block

    for(int k=0;k<n;k++) {
        c[i * n + j] += a[i*n + k]*b[k*n + j];
    }
}