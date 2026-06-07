//! Matrix operations: creation, arithmetic, transpose, determinant, and inverse.

use crate::vector::Vector;

/// A matrix of f64 values stored in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Data stored row-major: data[i * cols + j] = element at (i, j).
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a matrix from a 2D slice.
    pub fn from_2d(data: &[Vec<f64>]) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        Self { rows, cols, data: flat }
    }

    /// Create a matrix from rows of Vector.
    pub fn from_rows(rows: &[Vector]) -> Self {
        let n = rows.len();
        let m = if n > 0 { rows[0].len() } else { 0 };
        Self {
            rows: n,
            cols: m,
            data: rows.iter().flat_map(|r| r.data.iter().copied()).collect(),
        }
    }

    /// Create an identity matrix of size n.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self { rows: n, cols: n, data }
    }

    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.cols + j] = val;
    }

    /// Get row i as a Vector.
    pub fn row(&self, i: usize) -> Vector {
        let start = i * self.cols;
        Vector::new(self.data[start..start + self.cols].to_vec())
    }

    /// Get column j as a Vector.
    pub fn col(&self, j: usize) -> Vector {
        Vector::new((0..self.rows).map(|i| self.get(i, j)).collect())
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.get(i, j);
            }
        }
        Matrix { rows: self.cols, cols: self.rows, data }
    }

    /// Check if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Matrix multiplication: self * other.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "incompatible dimensions for multiplication");
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Multiply matrix by a vector.
    pub fn mul_vec(&self, v: &Vector) -> Vector {
        assert_eq!(self.cols, v.len(), "incompatible dimensions");
        Vector::new(
            (0..self.rows)
                .map(|i| {
                    (0..self.cols)
                        .map(|j| self.get(i, j) * v.get(j))
                        .sum()
                })
                .collect(),
        )
    }

    /// Scale the matrix by a scalar.
    pub fn scale(&self, s: f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    /// Add two matrices element-wise.
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(),
        }
    }

    /// Compute the trace (sum of diagonal elements).
    pub fn trace(&self) -> f64 {
        (0..self.rows.min(self.cols)).map(|i| self.get(i, i)).sum()
    }

    /// Compute the determinant (for square matrices).
    pub fn det(&self) -> f64 {
        assert!(self.is_square(), "determinant requires square matrix");
        let n = self.rows;
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return self.get(0, 0);
        }
        if n == 2 {
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        }

        // LU decomposition approach
        let mut mat = self.data.clone();
        let mut d = 1.0;
        for col in 0..n {
            // Find pivot
            let mut max_val = mat[col * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let v = mat[row * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                return 0.0;
            }
            if max_row != col {
                for j in 0..n {
                    mat.swap(col * n + j, max_row * n + j);
                }
                d *= -1.0;
            }
            d *= mat[col * n + col];
            for row in (col + 1)..n {
                let factor = mat[row * n + col] / mat[col * n + col];
                for j in (col + 1)..n {
                    mat[row * n + j] -= factor * mat[col * n + j];
                }
                mat[row * n + col] = 0.0;
            }
        }
        d
    }

    /// Compute the inverse of a square matrix.
    ///
    /// Returns `None` if the matrix is singular.
    pub fn inverse(&self) -> Option<Matrix> {
        assert!(self.is_square(), "inverse requires square matrix");
        let n = self.rows;
        let mut aug = vec![0.0; n * 2 * n];

        // Build augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = self.get(i, j);
            }
            aug[i * 2 * n + n + i] = 1.0;
        }

        // Gauss-Jordan elimination
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col * 2 * n + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * 2 * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                return None;
            }
            if max_row != col {
                for j in 0..(2 * n) {
                    aug.swap(col * 2 * n + j, max_row * 2 * n + j);
                }
            }
            let pivot = aug[col * 2 * n + col];
            for j in 0..(2 * n) {
                aug[col * 2 * n + j] /= pivot;
            }
            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row * 2 * n + col];
                for j in 0..(2 * n) {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }

        // Extract inverse
        let mut inv_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                inv_data[i * n + j] = aug[i * 2 * n + n + j];
            }
        }
        Some(Matrix { rows: n, cols: n, data: inv_data })
    }

    /// Compute the Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Check if the matrix is symmetric.
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i)).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the rank of the matrix via row echelon form.
    pub fn rank(&self) -> usize {
        let mut mat = self.data.clone();
        let m = self.rows;
        let n = self.cols;
        let mut rank = 0;

        for col in 0..n {
            if rank >= m { break; }
            // Find pivot
            let mut found = false;
            for row in rank..m {
                if mat[row * n + col].abs() > 1e-10 {
                    // Swap rows
                    if row != rank {
                        for j in 0..n {
                            mat.swap(rank * n + j, row * n + j);
                        }
                    }
                    found = true;
                    break;
                }
            }
            if !found { continue; }

            // Eliminate below
            for row in (rank + 1)..m {
                let factor = mat[row * n + col] / mat[rank * n + col];
                for j in col..n {
                    mat[row * n + j] -= factor * mat[rank * n + j];
                }
            }
            rank += 1;
        }
        rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 1), 1.0);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 3);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 2), 5.0);
    }

    #[test]
    fn test_mul() {
        let a = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::identity(2);
        let c = a.mul(&b);
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn test_mul_vec() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let v = Vector::new(vec![1.0, 0.0]);
        let r = m.mul_vec(&v);
        assert!((r.data[0] - 1.0).abs() < 1e-10);
        assert!((r.data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_2x2() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!((m.det() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_det_3x3() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]);
        assert!(m.det().abs() < 1e-10); // Singular
    }

    #[test]
    fn test_det_identity() {
        assert!((Matrix::identity(3).det() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_2x2() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv);
        // Should be identity
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_inverse_singular() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(m.inverse().is_none());
    }

    #[test]
    fn test_trace() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!((m.trace() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm() {
        let m = Matrix::identity(2);
        assert!((m.frobenius_norm() - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_rank() {
        assert_eq!(Matrix::identity(3).rank(), 3);
        let singular = Matrix::from_2d(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert_eq!(singular.rank(), 1);
    }

    #[test]
    fn test_is_symmetric() {
        assert!(Matrix::identity(3).is_symmetric());
        let asym = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(!asym.is_symmetric());
    }

    #[test]
    fn test_scale() {
        let m = Matrix::identity(2).scale(3.0);
        assert_eq!(m.get(0, 0), 3.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::identity(2);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 2.0);
        assert_eq!(c.get(1, 1), 5.0);
    }
}
