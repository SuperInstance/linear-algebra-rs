//! 2×2 Singular Value Decomposition (SVD).
//!
//! Computes A = U Σ V^T where U and V are orthogonal and Σ is diagonal.

use crate::matrix::Matrix;
use crate::vector::Vector;

/// Result of 2×2 SVD decomposition.
#[derive(Debug, Clone)]
pub struct SVDResult {
    /// Left singular vectors (orthogonal matrix U).
    pub u: Matrix,
    /// Singular values (diagonal matrix Σ).
    pub sigma: Matrix,
    /// Right singular vectors (orthogonal matrix V).
    pub v: Matrix,
}

/// Compute the SVD of a 2×2 matrix analytically.
///
/// For a 2×2 matrix, we can compute the SVD in closed form using
/// the eigenvalues of A^T A.
pub fn svd_2x2(matrix: &Matrix) -> SVDResult {
    assert_eq!(matrix.rows, 2, "svd_2x2 requires a 2×2 matrix");
    assert_eq!(matrix.cols, 2, "svd_2x2 requires a 2×2 matrix");

    let a = matrix.get(0, 0);
    let b = matrix.get(0, 1);
    let c = matrix.get(1, 0);
    let d = matrix.get(1, 1);

    // Compute A^T A
    let ata_00 = a * a + c * c;
    let ata_01 = a * b + c * d;
    let ata_11 = b * b + d * d;

    // Eigenvalues of A^T A (singular values squared)
    let trace = ata_00 + ata_11;
    let det_ata = ata_00 * ata_11 - ata_01 * ata_01;
    let disc = ((trace * trace - 4.0 * det_ata).max(0.0)).sqrt();

    let sigma1_sq = (trace + disc) / 2.0;
    let sigma2_sq = (trace - disc) / 2.0;
    let sigma1 = sigma1_sq.max(0.0).sqrt();
    let sigma2 = sigma2_sq.max(0.0).sqrt();

    // Compute V from eigenvectors of A^T A
    let v = if ata_01.abs() > 1e-15 {
        let v11 = (sigma1_sq - ata_11) / ata_01;
        let v_norm = (1.0 + v11 * v11).sqrt();
        Matrix::from_2d(&[
            vec![v11 / v_norm, -1.0 / v_norm],
            vec![1.0 / v_norm, v11 / v_norm],
        ])
    } else if ata_00 >= ata_11 {
        Matrix::identity(2)
    } else {
        Matrix::from_2d(&[vec![0.0, 1.0], vec![1.0, 0.0]])
    };

    // Compute U = A V Σ^{-1}
    let sigma_inv = Matrix::from_2d(&[
        vec![if sigma1 > 1e-15 { 1.0 / sigma1 } else { 0.0 }, 0.0],
        vec![0.0, if sigma2 > 1e-15 { 1.0 / sigma2 } else { 0.0 }],
    ]);

    let sigma_mat = Matrix::from_2d(&[vec![sigma1, 0.0], vec![0.0, sigma2]]);
    let av = matrix.mul(&v);
    let u = if sigma1 > 1e-15 && sigma2 > 1e-15 {
        av.mul(&sigma_inv)
    } else if sigma1 > 1e-15 {
        // Only one non-zero singular value
        let u1 = Vector::new(vec![av.get(0, 0), av.get(1, 0)]).normalize().unwrap();
        // Orthogonal complement
        Matrix::from_2d(&[vec![u1.data[0], -u1.data[1]], vec![u1.data[1], u1.data[0]]])
    } else {
        Matrix::identity(2)
    };

    SVDResult {
        u,
        sigma: sigma_mat,
        v,
    }
}

/// Compute the condition number of a 2×2 matrix from its SVD.
pub fn condition_number_2x2(matrix: &Matrix) -> f64 {
    let svd = svd_2x2(matrix);
    let s1 = svd.sigma.get(0, 0);
    let s2 = svd.sigma.get(1, 1);
    if s2.abs() < 1e-15 {
        f64::INFINITY
    } else {
        s1 / s2
    }
}

/// Compute the pseudo-inverse using SVD for a 2×2 matrix.
pub fn pseudo_inverse_2x2(matrix: &Matrix) -> Matrix {
    let svd = svd_2x2(matrix);
    let tol = svd.sigma.get(0, 0) * 1e-10 * 2.0; // tolerance
    let sigma_inv = Matrix::from_2d(&[
        vec![if svd.sigma.get(0, 0) > tol { 1.0 / svd.sigma.get(0, 0) } else { 0.0 }, 0.0],
        vec![0.0, if svd.sigma.get(1, 1) > tol { 1.0 / svd.sigma.get(1, 1) } else { 0.0 }],
    ]);
    svd.v.mul(&sigma_inv).mul(&svd.u.transpose())
}

/// Compute the Frobenius norm from singular values.
pub fn svd_frobenius_norm(matrix: &Matrix) -> f64 {
    let svd = svd_2x2(matrix);
    (svd.sigma.get(0, 0).powi(2) + svd.sigma.get(1, 1).powi(2)).sqrt()
}

/// Compute the nuclear norm (sum of singular values).
pub fn nuclear_norm_2x2(matrix: &Matrix) -> f64 {
    let svd = svd_2x2(matrix);
    svd.sigma.get(0, 0) + svd.sigma.get(1, 1)
}

/// Low-rank approximation: reconstruct from top-k singular values.
pub fn low_rank_approx_2x2(matrix: &Matrix, k: usize) -> Matrix {
    let svd = svd_2x2(matrix);
    let k = k.min(2);
    let mut sigma = svd.sigma.clone();
    for i in k..2 {
        sigma.set(i, i, 0.0);
    }
    svd.u.mul(&sigma).mul(&svd.v.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_orthogonal(m: &Matrix) -> bool {
        if m.rows != m.cols { return false; }
        let mtm = m.transpose().mul(m);
        for i in 0..m.rows {
            for j in 0..m.cols {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (mtm.get(i, j) - expected).abs() > 1e-8 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_svd_identity() {
        let m = Matrix::identity(2);
        let svd = svd_2x2(&m);
        assert!(check_orthogonal(&svd.u));
        assert!(check_orthogonal(&svd.v));
        assert!((svd.sigma.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((svd.sigma.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_diagonal() {
        let m = Matrix::from_2d(&[vec![3.0, 0.0], vec![0.0, 2.0]]);
        let svd = svd_2x2(&m);
        assert!((svd.sigma.get(0, 0) - 3.0).abs() < 1e-8);
        assert!((svd.sigma.get(1, 1) - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_svd_reconstruction() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let svd = svd_2x2(&m);
        let reconstructed = svd.u.mul(&svd.sigma).mul(&svd.v.transpose());
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed.get(i, j) - m.get(i, j)).abs() < 1e-8,
                    "Reconstruction mismatch at ({}, {}): got {} expected {}",
                    i, j, reconstructed.get(i, j), m.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_svd_orthogonal_u() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let svd = svd_2x2(&m);
        assert!(check_orthogonal(&svd.u));
    }

    #[test]
    fn test_svd_orthogonal_v() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let svd = svd_2x2(&m);
        assert!(check_orthogonal(&svd.v));
    }

    #[test]
    fn test_svd_singular_values_ordered() {
        let m = Matrix::from_2d(&[vec![3.0, 1.0], vec![1.0, 3.0]]);
        let svd = svd_2x2(&m);
        assert!(svd.sigma.get(0, 0) >= svd.sigma.get(1, 1));
    }

    #[test]
    fn test_condition_number() {
        let m = Matrix::identity(2);
        let cn = condition_number_2x2(&m);
        assert!((cn - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_pseudo_inverse() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let pinv = pseudo_inverse_2x2(&m);
        // A * A+ should be close to identity
        let product = m.mul(&pinv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product.get(i, j) - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_nuclear_norm() {
        let m = Matrix::from_2d(&[vec![3.0, 0.0], vec![0.0, 2.0]]);
        let nn = nuclear_norm_2x2(&m);
        assert!((nn - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_low_rank_approx() {
        let m = Matrix::from_2d(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let approx1 = low_rank_approx_2x2(&m, 1);
        // Rank-1 approximation
        assert_eq!(approx1.rows, 2);
        assert_eq!(approx1.cols, 2);
        // Should be close but not exact
        let error = approx1.add(&m.scale(-1.0)).frobenius_norm();
        assert!(error > 0.0); // Not exact for rank-1
    }

    #[test]
    fn test_svd_rotation() {
        let theta = std::f64::consts::FRAC_PI_4;
        let m = Matrix::from_2d(&[
            vec![theta.cos(), -theta.sin()],
            vec![theta.sin(), theta.cos()],
        ]);
        let svd = svd_2x2(&m);
        assert!((svd.sigma.get(0, 0) - 1.0).abs() < 1e-8);
        assert!((svd.sigma.get(1, 1) - 1.0).abs() < 1e-8);
    }
}
