# Kernels
PTX and OpenCL source files used in the formal verification.

- `sum_even.cl` — OpenCL source of the even-number reduction kernel
- `sum_even.ptx` — compiled PTX assembly (NVVM output)

The Promela model in the repository root implements the semantics of these files.
