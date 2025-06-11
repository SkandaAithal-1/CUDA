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

## Task 09: 1d Convolution

**Summary** :  
Implemented a simple kernel for 1d convolution with no optimisation consideration. Will optimise it further.