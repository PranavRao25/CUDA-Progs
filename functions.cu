#define N 10

__global__ void find_max(int* input, int* max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int old_max;

    if (i < n) {
        int val = input[i];

        // compare the max with val to see if it is the current max
        old_max = atomicCAS(max, *max, val);

        // chance that some other thread updates max above
        // after this thread, causing the old_max value to change
        // so we check if this old_max is smaller than current val, we update again
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

        // compare the min with val to see if it is the current min
        old_min = atomicCAS(min, *min, val);

        // chance that some other thread updates min above
        // after this thread, causing the old_min value to change
        // so we check if this old_min is larger than current val, we update again
        while (old_min > val) {
            old_min = atomicCAS(min, old_min, val);
        }
    }
}

__global__ void vector_addition(int *a, int *b, int *c) {

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

__device__ unsigned int atomic_inc_CAS(unsigned int* address, unsigned int max) {
    unsigned int old_value, new_value;

    do {
        old_value = *address;
        new_value = (old_value >= max) ? 0 : old_value + 1;  // timer logic

        // If another thread has done atomicCAS or updated address,
        // then this update will fail, and the loop will run again
        // then the new_value will be different for the new value of address
    } while (atomicCAS(address, old_value, new_value) != old_value);

    return old_value;
}

__device unsigned int atomic_add_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value;

    do {
        old_value = *address;
    } while(atomicCAS(address, old_value, old_value + val) != old_value);

    return old_value;
}

__device__ unsigned int atomic_sub_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value;

    do {
        old_value = *address;
    } while(atomicCAS(address, old_value, old_value - val) != old_value);

    return old_value;
}

__device__ unsigned int atomic_max_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value, new_value;

    do {
        old_value = *address;
        new_value = (old_value > val) ? old_value : val;
    } while(atomicCAS(address, old_value, new_value) != old_value);

    return old_value;
}

__device__ unsigned int atomic_min_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value, new_value;

    do {
        old_value = *address;
        new_value = (old_value < val) ? old_value : val;
    } while(atomicCAS(address, old_value, new_value) != old_value);

    return old_value;
}

__device__ unsigned int atomic_exch_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value;

    do {
        old_value = *address;
    } while(atomicCAS(address, old_value, val) != old_value);

    return old_value;
}

__device__ unsigned int atomic_and_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value, new_value;

    do {
        old_value = *address;
        new_value = old_value & val;
    } while(atomicCAS(address, old_value, new_value) != old_value);

    return old_value;
}

__device__ unsigned int atomic_or_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value, new_value;

    do {
        old_value = *address;
        new_value = old_value | val;
    } while(atomicCAS(address, old_value, new_value) != old_value);

    return old_value;
}

__device__ unsigned int atomic_xor_CAS(unsigned int *address, unsigned int val) {
    unsigned int old_value, new_value;

    do {
        old_value = *address;
        new_value = old_value ^ val;
    } while(atomicCAS(address, old_value, new_value) != old_value);

    return old_value;
}
