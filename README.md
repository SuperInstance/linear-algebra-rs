# linear-algebra-rs

Linear algebra operations: vector/matrix ops, Gaussian elimination, power method eigenvalues, and 2×2 SVD.

## Features

- **Vector**: Arithmetic, dot/cross products, norms, projection, angle
- **Matrix**: Multiply, transpose, determinant, inverse, rank, Frobenius norm
- **Elimination**: Gaussian elimination, RREF, LU decomposition, linear system solving
- **Eigen**: Power method, inverse power method, 2×2 analytical eigenvalues
- **SVD**: 2×2 SVD, condition number, pseudo-inverse, nuclear norm, low-rank approximation

Pure Rust, no external dependencies.

## Usage

```rust
use linear_algebra_rs::{Matrix, Vector};

let a = Matrix::from_2d(&[vec![2.0, 1.0], vec![1.0, 3.0]]);
let inv = a.inverse().unwrap();
let product = a.mul(&inv); // ≈ identity
```

License: MIT OR Apache-2.0
