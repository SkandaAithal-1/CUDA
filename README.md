# CUDA

This is to serve as the documentation for CUDA learning journey. Studied from the book **PMPP (Programming Massively Parallel Processors)**.

---

## Task 01: Vector Addition

**Summary** :  
Implemented a simple kernel to add two vectors. Explored launching kernels, device memory management, data transfer between host and device and basic CUDA programming concepts.

---

## Task 02: Matrix Addition

**Summary** :  
Implemented a kernel to add two matrices. Learnt about barrier synchronisation, thread scheduling, hardware resource management (warps and stuff) and latency tolerance.

---

## Task 03: Matrix Vector Multiplication

**Summary** :
Implemented a kernel to multiply a matrix with a vector. Explored shared memory, thread cooperation and performance optimisations.

---

## Task 04: Vector sum

**Summary** :  
Implemented a kernel to compute the sum of all elements of a vector. Explored reduction techniques, and warp divergence and performance optimisations.

---

## Task 05: Layer Normalization

**Summary** :  
Implemented a kernel to perform layer normalization on a matrix.

---

## Task 06: Dot Product

**Summary** :  
Implemented a kernel to compute the dot product of two vectors. Used vector sum kernel previously implemented to compute the sum of products.

---

## Task 07: L2 Norm

**Summary** :  
Implemented a kernel to compute the L2 norm of a vector. Used vector sum kernel again.

---

## Task 08: Transpose

**Summary** :  
Implemented a kernel to transpose a matrix.

---

## Task 09: 1d Convolution

**Summary** :  
Implemented a simple kernel for 1d convolution with no optimisation consideration. Will optimise it further.

**Update** :  
Optimized the kernel to use shared memory to reduce global memory access. Tiling is done so that the thread block covers the input vector.

**Update** :  
Another optimized kernel where the threads map to the output vector. This makes loading the input slightly complex.

---

## Task 10: 2d Convolution

**Summary** :  
Implemented an optimised kernel for 2d convolution. The threads map to the input vector and some threads of the block are turned off during output vector calculation.

## Task 11: Stencil

**Summary** :  
Implemented a 3D 7 point stencil operation. The tiling is similar to the 2D convolution.

**Details** :

- This uses thread coarsening to reduce the number of threads launched.
- The data reuse in this case is less compared to 2D convolution, so the upper limit on floating-point to global memory access ratio is not as high.
- This is the motivation for thread coarsening.
- Register tiling is also done here to reduce the amount of shared memory used since data in the third dimension is not necessarily shared with the entire block.
