//! Gaussian elimination, LU decomposition, and linear system solving.

use crate::matrix::Matrix;
use crate::vector::Vector;

/// Result of Gaussian elimination: the row echelon form and pivot columns.
#[derive(Debug, Clone)]
pub struct EliminationResult {
    /// Row echelon form of the matrix.
    pub row_echelon: Matrix,
    /// Pivot column indices.
    pub pivots: Vec<usize>,
    /// Number of row swaps performed.
    pub swaps: usize,
}

/// Perform Gaussian elimination with partial pivoting.
pub fn gaussian_elimination(matrix: &Matrix) -> EliminationResult {
    let m = matrix.rows;
    let n = matrix.cols;
    let mut data = matrix.data.clone();
    let mut pivots = vec![];
    let mut swaps = 0;
    let mut pivot_row = 0;

    for col in 0..n {
        if pivot_row >= m { break; }

        // Find pivot
        let mut max_val = data[pivot_row * n + col].abs();
        let mut max_row = pivot_row;
        for row in (pivot_row + 1)..m {
            let v = data[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-12 { continue; }

        // Swap rows
        if max_row != pivot_row {
            for j in 0..n {
                data.swap(pivot_row * n + j, max_row * n + j);
            }
            swaps += 1;
        }

        pivots.push(col);

        // Eliminate below
        for row in (pivot_row + 1)..m {
            let factor = data[row * n + col] / data[pivot_row * n + col];
            for j in col..n {
                data[row * n + j] -= factor * data[pivot_row * n + j];
            }
        }

        pivot_row += 1;
    }

    EliminationResult {
        row_echelon: Matrix { rows: m, cols: n, data },
        pivots,
        swaps,
    }
}

/// Reduce to reduced row echelon form (RREF).
pub fn reduced_row_echelon(matrix: &Matrix) -> Matrix {
    let m = matrix.rows;
    let n = matrix.cols;
    let mut data = matrix.data.clone();
    let mut pivot_row = 0;

    for col in 0..n {
        if pivot_row >= m { break; }

        // Find pivot
        let mut max_row = pivot_row;
        let mut max_val = data[pivot_row * n + col].abs();
        for row in (pivot_row + 1)..m {
            let v = data[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-12 { continue; }

        // Swap
        if max_row != pivot_row {
            for j in 0..n {
                data.swap(pivot_row * n + j, max_row * n + j);
            }
        }

        // Scale pivot row
        let pivot = data[pivot_row * n + col];
        for j in 0..n {
            data[pivot_row * n + j] /= pivot;
        }

        // Eliminate all other rows
        for row in 0..m {
            if row == pivot_row { continue; }
            let factor = data[row * n + col];
            if factor.abs() < 1e-12 { continue; }
            for j in 0..n {
                data[row * n + j] -= factor * data[pivot_row * n + j];
            }
        }

        pivot_row += 1;
    }

    Matrix { rows: m, cols: n, data }
}

/// LU decomposition (with partial pivoting): PA = LU.
///
/// Returns (P, L, U) where P is a permutation matrix, L is lower triangular,
/// and U is upper triangular.
pub fn lu_decompose(matrix: &Matrix) -> Option<(Matrix, Matrix, Matrix)> {
    assert!(matrix.is_square(), "LU decomposition requires square matrix");
    let n = matrix.rows;
    let mut u = matrix.data.clone();
    let mut l = vec![0.0; n * n];
    let mut perm: Vec<usize> = (0..n).collect();

    // Initialize L as identity
    for i in 0..n {
        l[i * n + i] = 1.0;
    }

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = u[col * n + col].abs();
        for row in (col + 1)..n {
            let v = u[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        // Swap rows in U and perm
        if max_row != col {
            perm.swap(col, max_row);
            for j in 0..n {
                u.swap(col * n + j, max_row * n + j);
            }
            // Swap L entries below diagonal
            for j in 0..col {
                l.swap(col * n + j, max_row * n + j);
            }
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = u[row * n + col] / u[col * n + col];
            l[row * n + col] = factor;
            for j in col..n {
                u[row * n + j] -= factor * u[col * n + j];
            }
        }
    }

    // Build permutation matrix
    let mut p_data = vec![0.0; n * n];
    for (i, &j) in perm.iter().enumerate() {
        p_data[i * n + j] = 1.0;
    }

    Some((
        Matrix { rows: n, cols: n, data: p_data },
        Matrix { rows: n, cols: n, data: l },
        Matrix { rows: n, cols: n, data: u },
    ))
}

/// Solve a linear system Ax = b using Gaussian elimination.
///
/// Returns `None` if the system has no unique solution.
pub fn solve(a: &Matrix, b: &Vector) -> Option<Vector> {
    assert_eq!(a.rows, b.len(), "incompatible dimensions");
    let n = a.rows;
    let m = a.cols;

    // Build augmented matrix [A | b]
    let mut aug = vec![0.0; n * (m + 1)];
    for i in 0..n {
        for j in 0..m {
            aug[i * (m + 1) + j] = a.get(i, j);
        }
        aug[i * (m + 1) + m] = b.get(i);
    }

    // Forward elimination
    let mut pivot_row = 0;
    for col in 0..m {
        if pivot_row >= n { break; }
        let mut max_row = pivot_row;
        let mut max_val = aug[pivot_row * (m + 1) + col].abs();
        for row in (pivot_row + 1)..n {
            let v = aug[row * (m + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 { continue; }
        if max_row != pivot_row {
            for j in 0..=m {
                aug.swap(pivot_row * (m + 1) + j, max_row * (m + 1) + j);
            }
        }
        for row in (pivot_row + 1)..n {
            let factor = aug[row * (m + 1) + col] / aug[pivot_row * (m + 1) + col];
            for j in col..=m {
                aug[row * (m + 1) + j] -= factor * aug[pivot_row * (m + 1) + j];
            }
        }
        pivot_row += 1;
    }

    // Check for inconsistency
    for row in pivot_row..n {
        if aug[row * (m + 1) + m].abs() > 1e-10 {
            return None;
        }
    }

    if pivot_row < m {
        return None; // Underdetermined
    }

    // Back substitution
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = aug[i * (m + 1) + m];
        for j in (i + 1)..m {
            sum -= aug[i * (m + 1) + j] * x[j];
        }
        if aug[i * (m + 1) + i].abs() < 1e-12 {
            return None;
        }
        x[i] = sum / aug[i * (m + 1) + i];
    }

    Some(Vector::new(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_identity() {
        let m = Matrix::identity(3);
        let result = gaussian_elimination(&m);
        assert_eq!(result.pivots, vec![0, 1, 2]);
    }

    #[test]
    fn test_gaussian_singular() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        let result = gaussian_elimination(&m);
        assert_eq!(result.pivots.len(), 1);
    }

    #[test]
    fn test_rref() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let rref = reduced_row_echelon(&m);
        assert!((rref.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((rref.get(0, 1)).abs() < 1e-10);
        assert!((rref.get(1, 0)).abs() < 1e-10);
        assert!((rref.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_decompose() {
        let m = Matrix::from_2d(&[vec![2.0, 1.0], vec![6.0, 4.0]]);
        let (p, l, u) = lu_decompose(&m).unwrap();
        // PA should equal LU
        let lu = l.mul(&u);
        let pa = p.mul(&m);
        for i in 0..2 {
            for j in 0..2 {
                assert!((lu.get(i, j) - pa.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lu_singular() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(lu_decompose(&m).is_none());
    }

    #[test]
    fn test_solve_simple() {
        let a = Matrix::from_2d(&[vec![2.0, 1.0], vec![5.0, 3.0]]);
        let b = Vector::new(vec![4.0, 7.0]);
        let x = solve(&a, &b).unwrap();
        // Verify Ax = b
        let ax = a.mul_vec(&x);
        assert!((ax.data[0] - 4.0).abs() < 1e-10);
        assert!((ax.data[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_identity() {
        let a = Matrix::identity(3);
        let b = Vector::new(vec![1.0, 2.0, 3.0]);
        let x = solve(&a, &b).unwrap();
        assert!((x.data[0] - 1.0).abs() < 1e-10);
        assert!((x.data[1] - 2.0).abs() < 1e-10);
        assert!((x.data[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_singular() {
        let a = Matrix::from_2d(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        let b = Vector::new(vec![1.0, 2.0]);
        assert!(solve(&a, &b).is_none());
    }

    #[test]
    fn test_elimination_pivots() {
        let m = Matrix::from_2d(&[vec![0.0, 1.0], vec![1.0, 0.0]]);
        let result = gaussian_elimination(&m);
        assert_eq!(result.pivots.len(), 2);
    }
}
