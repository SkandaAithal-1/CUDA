# 🚀 CUDA Learning Journey

This repository documents my exploration of CUDA programming, inspired by the book **Programming Massively Parallel Processors (PMPP)**.

---

## 📁 Repository Structure

```
CUDA/
├── Task01/  # Vector Addition
├── Task02/  # Matrix Addition
├── Task03/  # Matrix-Vector Multiplication
├── Task04/  # Vector Sum (Reduction)
├── Task05/  # Layer Normalization
├── Task06/  # Dot Product
├── Task07/  # L2 Norm
├── Task08/  # Matrix Transpose
├── Task09/  # 1D Convolution
├── Task10/  # 2D Convolution
├── Task11/  # 3D Stencil
├── Task12/  # Histogram
├── utils/   # Utility Functions
├── kernels.cuh
├── utils.h
├── run.sh
└── README.md
```

---

## 📝 Task-wise Documentation

### Task 01: Vector Addition
**Summary:**  
Implemented a simple kernel to add two vectors. Explored kernel launches, device memory management, data transfer between host and device, and basic CUDA programming concepts.

---

### Task 02: Matrix Addition
**Summary:**  
Implemented a kernel to add two matrices. Learned about barrier synchronization, thread scheduling, hardware resource management (warps), and latency tolerance.

---

### Task 03: Matrix-Vector Multiplication
**Summary:**  
Implemented a kernel to multiply a matrix with a vector. Explored shared memory, thread cooperation, and performance optimizations.

---

### Task 04: Vector Sum (Reduction)
**Summary:**  
Implemented a kernel to compute the sum of all elements in a vector. Explored reduction techniques, warp divergence, and performance optimizations.

---

### Task 05: Layer Normalization
**Summary:**  
Implemented a kernel to perform layer normalization on a matrix.

---

### Task 06: Dot Product
**Summary:**  
Implemented a kernel to compute the dot product of two vectors, utilizing the previously implemented vector sum kernel.

---

### Task 07: L2 Norm
**Summary:**  
Implemented a kernel to compute the L2 norm of a vector, again leveraging the vector sum kernel.

---

### Task 08: Matrix Transpose
**Summary:**  
Implemented a kernel to transpose a matrix.

---

### Task 09: 1D Convolution
**Summary:**  
Implemented a simple kernel for 1D convolution without optimizations.  
**Update:** Optimized the kernel using shared memory and tiling to reduce global memory access.  
**Update:** Further optimized with threads mapping to the output vector, making input loading more complex.

---

### Task 10: 2D Convolution
**Summary:**  
Implemented an optimized kernel for 2D convolution. Threads map to the input vector, with some threads in the block turned off during output calculation.

---

### Task 11: 3D Stencil
**Summary:**  
Implemented a 3D 7-point stencil operation, with tiling similar to 2D convolution.

**Details:**
- Utilizes thread coarsening to reduce the number of threads launched.
- Data reuse is less compared to 2D convolution, motivating thread coarsening.
- Register tiling reduces shared memory usage, as data in the third dimension is not always shared across the block.

---

### Task 12: Histogram
**Summary:**  
Implemented a kernel to compute the histogram of a vector, using atomic operations with privatization and thread coarsening (interleaved partitioning).

**Learned:**
- Atomic operations in CUDA for safe concurrent updates.
- Privatization to reduce contention on atomic operations.
- Thread coarsening to further reduce contention during histogram reduction.
- Explored contiguous and interleaved partitioning schemes.
- Aggregation strategies for localized identical values in the input vector.

---

## 🛠️ Build & Run

To compile and run a task, use the provided script:

```sh
./run.sh Task01/vector_addition.cu vector_add
```
Replace the file path and output name as needed.

---

## 📚 References

- _Programming Massively Parallel Processors (PMPP)_
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)