//! Power method for eigenvalue computation and related operations.

use crate::matrix::Matrix;
use crate::vector::Vector;

/// Result of the power method.
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// The dominant eigenvalue.
    pub eigenvalue: f64,
    /// The corresponding eigenvector.
    pub eigenvector: Vector,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the method converged.
    pub converged: bool,
}

/// Compute the dominant eigenvalue and eigenvector using the power method.
///
/// # Arguments
/// * `matrix` - A square matrix
/// * `initial` - Initial vector guess
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
pub fn power_method(
    matrix: &Matrix,
    initial: &Vector,
    max_iter: usize,
    tol: f64,
) -> EigenResult {
    assert!(matrix.is_square(), "power method requires square matrix");
    assert_eq!(matrix.rows, initial.len(), "matrix and vector dimensions must match");

    let mut v = initial.clone();
    let mut eigenvalue = 0.0;
    let mut converged = false;

    for iter in 0..max_iter {
        let w = matrix.mul_vec(&v);
        let new_eigenvalue = v.dot(&w) / v.dot(&v);

        // Normalize
        let norm = w.norm();
        if norm < 1e-15 {
            return EigenResult {
                eigenvalue: 0.0,
                eigenvector: v,
                iterations: iter,
                converged: false,
            };
        }
        v = w.scale(1.0 / norm);

        if iter > 0 && (new_eigenvalue - eigenvalue).abs() < tol {
            eigenvalue = new_eigenvalue;
            converged = true;
            return EigenResult {
                eigenvalue,
                eigenvector: v,
                iterations: iter + 1,
                converged,
            };
        }
        eigenvalue = new_eigenvalue;
    }

    EigenResult {
        eigenvalue,
        eigenvector: v,
        iterations: max_iter,
        converged,
    }
}

/// Compute eigenvalues of a 2×2 matrix analytically.
///
/// For a 2×2 matrix [[a, b], [c, d]], eigenvalues are:
/// λ = (a+d)/2 ± sqrt(((a-d)/2)² + bc)
pub fn eigenvalues_2x2(matrix: &Matrix) -> (f64, f64) {
    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.cols, 2);
    let a = matrix.get(0, 0);
    let b = matrix.get(0, 1);
    let c = matrix.get(1, 0);
    let d = matrix.get(1, 1);

    let trace = a + d;
    let det = a * d - b * c;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();

    ((trace + disc) / 2.0, (trace - disc) / 2.0)
}

/// Check if a matrix is positive definite using Sylvester's criterion
/// (all leading principal minors > 0).
pub fn is_positive_definite(matrix: &Matrix) -> bool {
    if !matrix.is_square() {
        return false;
    }
    let n = matrix.rows;
    for k in 1..=n {
        let sub = matrix.submatrix(0, 0, k, k);
        if sub.det() <= 0.0 {
            return false;
        }
    }
    true
}

/// Compute the Rayleigh quotient: (v^T A v) / (v^T v).
pub fn rayleigh_quotient(matrix: &Matrix, v: &Vector) -> f64 {
    let av = matrix.mul_vec(v);
    v.dot(&av) / v.dot(v)
}

/// Compute the spectral radius (largest absolute eigenvalue) using the power method.
pub fn spectral_radius(matrix: &Matrix, max_iter: usize, tol: f64) -> f64 {
    let initial = Vector::ones(matrix.rows);
    let result = power_method(matrix, &initial, max_iter, tol);
    result.eigenvalue.abs()
}

/// Inverse power method to find the eigenvalue closest to a given shift.
pub fn inverse_power_method(
    matrix: &Matrix,
    shift: f64,
    initial: &Vector,
    max_iter: usize,
    tol: f64,
) -> EigenResult {
    // Compute (A - σI)
    let n = matrix.rows;
    let mut shifted = matrix.clone();
    for i in 0..n {
        shifted.set(i, i, shifted.get(i, i) - shift);
    }

    let mut v = initial.clone();
    let mut eigenvalue = 0.0;
    let mut converged = false;

    for iter in 0..max_iter {
        // Solve (A - σI)w = v
        let w = match crate::elimination::solve(&shifted, &v) {
            Some(w) => w,
            None => {
                return EigenResult {
                    eigenvalue: shift,
                    eigenvector: v,
                    iterations: iter,
                    converged: false,
                }
            }
        };

        let norm = w.norm();
        if norm < 1e-15 {
            return EigenResult {
                eigenvalue: shift,
                eigenvector: v,
                iterations: iter,
                converged: false,
            };
        }
        v = w.scale(1.0 / norm);
        let new_eigenvalue = rayleigh_quotient(matrix, &v);

        if iter > 0 && (new_eigenvalue - eigenvalue).abs() < tol {
            eigenvalue = new_eigenvalue;
            converged = true;
            return EigenResult {
                eigenvalue,
                eigenvector: v,
                iterations: iter + 1,
                converged,
            };
        }
        eigenvalue = new_eigenvalue;
    }

    EigenResult {
        eigenvalue,
        eigenvector: v,
        iterations: max_iter,
        converged,
    }
}

impl Matrix {
    /// Extract a submatrix.
    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Matrix {
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = self.get(row_start + i, col_start + j);
            }
        }
        Matrix { rows, cols, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_method_dominant() {
        // [[2, 1], [1, 3]] has eigenvalues ~3.618 and ~1.382
        let m = Matrix::from_2d(&[vec![2.0, 1.0], vec![1.0, 3.0]]);
        let v0 = Vector::new(vec![1.0, 0.0]);
        let result = power_method(&m, &v0, 100, 1e-10);
        assert!(result.converged);
        assert!((result.eigenvalue - 3.618).abs() < 0.01);
    }

    #[test]
    fn test_power_method_identity() {
        let m = Matrix::identity(3);
        let v0 = Vector::new(vec![1.0, 0.0, 0.0]);
        let result = power_method(&m, &v0, 100, 1e-10);
        assert!((result.eigenvalue - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_eigenvalues_2x2() {
        let m = Matrix::from_2d(&[vec![2.0, 1.0], vec![1.0, 3.0]]);
        let (l1, l2) = eigenvalues_2x2(&m);
        // Eigenvalues: (5 ± sqrt(5))/2 ≈ 3.618 and 1.382
        let expected1 = (5.0 + 5.0f64.sqrt()) / 2.0;
        let expected2 = (5.0 - 5.0f64.sqrt()) / 2.0;
        assert!((l1 - expected1).abs() < 1e-10);
        assert!((l2 - expected2).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_2x2_diagonal() {
        let m = Matrix::from_2d(&[vec![3.0, 0.0], vec![0.0, 5.0]]);
        let (l1, l2) = eigenvalues_2x2(&m);
        assert!((l1 - 5.0).abs() < 1e-10);
        assert!((l2 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_positive_definite() {
        let m = Matrix::from_2d(&[vec![2.0, 1.0], vec![1.0, 3.0]]);
        assert!(is_positive_definite(&m));
    }

    #[test]
    fn test_not_positive_definite() {
        let m = Matrix::from_2d(&[vec![-1.0, 0.0], vec![0.0, -1.0]]);
        assert!(!is_positive_definite(&m));
    }

    #[test]
    fn test_rayleigh_quotient() {
        let m = Matrix::from_2d(&[vec![4.0, 0.0], vec![0.0, 2.0]]);
        let v = Vector::new(vec![1.0, 0.0]);
        let rq = rayleigh_quotient(&m, &v);
        assert!((rq - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_radius() {
        let m = Matrix::from_2d(&[vec![3.0, 0.0], vec![0.0, 2.0]]);
        let sr = spectral_radius(&m, 100, 1e-10);
        assert!((sr - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_inverse_power_method() {
        let m = Matrix::from_2d(&[vec![2.0, 1.0], vec![1.0, 3.0]]);
        let v0 = Vector::new(vec![1.0, 0.0]);
        let result = inverse_power_method(&m, 1.4, &v0, 100, 1e-6);
        // Should find eigenvalue close to ~1.382
        assert!((result.eigenvalue - 1.382).abs() < 0.1);
    }

    #[test]
    fn test_submatrix() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.get(0, 0), 1.0);
        assert_eq!(sub.get(1, 1), 5.0);
    }
}
