//! Iterative methods for solving linear systems.
//!
//! - Conjugate Gradient (CG) method for symmetric positive definite systems
//! - GMRES (Generalized Minimal Residual) for non-symmetric systems
//! - Jacobi and Gauss-Seidel preconditioners

use crate::matrix::Matrix;
use crate::vector::Vector;

// ========== Conjugate Gradient ==========

/// Result of the conjugate gradient method.
#[derive(Debug, Clone)]
pub struct CGResult {
    /// Solution vector.
    pub x: Vector,
    /// Residual norm history.
    pub residuals: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the method converged.
    pub converged: bool,
}

/// Solve Ax = b using the Conjugate Gradient method.
///
/// A must be symmetric positive definite. This is the most efficient
/// iterative solver for SPD systems.
pub fn conjugate_gradient(
    a: &Matrix,
    b: &Vector,
    max_iter: usize,
    tol: f64,
) -> CGResult {
    let n = b.len();
    let mut x = Vector::zeros(n);
    let mut r = b.sub(&a.mul_vec(&x)); // r = b - Ax
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);
    let mut residuals = vec![rs_old.sqrt()];
    let mut converged = false;

    for k in 0..max_iter {
        let ap = a.mul_vec(&p);
        let p_ap = p.dot(&ap);

        if p_ap.abs() < 1e-30 {
            converged = true;
            return CGResult { x, residuals, iterations: k, converged };
        }

        let alpha = rs_old / p_ap;
        x = x.add(&p.scale(alpha));
        r = r.sub(&ap.scale(alpha));

        let rs_new = r.dot(&r);
        residuals.push(rs_new.sqrt());

        if rs_new.sqrt() < tol {
            converged = true;
            return CGResult { x, residuals, iterations: k + 1, converged };
        }

        let beta = rs_new / rs_old;
        p = r.add(&p.scale(beta));
        rs_old = rs_new;
    }

    CGResult { x, residuals, iterations: max_iter, converged }
}

/// Solve Ax = b using Conjugate Gradient with preconditioning.
///
/// The preconditioner M should approximate A^{-1} such that M*A is better
/// conditioned than A.
pub fn preconditioned_cg(
    a: &Matrix,
    b: &Vector,
    preconditioner: &dyn Fn(&Vector) -> Vector,
    max_iter: usize,
    tol: f64,
) -> CGResult {
    let n = b.len();
    let mut x = Vector::zeros(n);
    let mut r = b.sub(&a.mul_vec(&x));
    let mut z = preconditioner(&r);
    let mut p = z.clone();
    let mut rz_old = r.dot(&z);
    let mut residuals = vec![r.norm()];
    let mut converged = false;

    for k in 0..max_iter {
        let ap = a.mul_vec(&p);
        let p_ap = p.dot(&ap);

        if p_ap.abs() < 1e-30 {
            converged = true;
            return CGResult { x, residuals, iterations: k, converged };
        }

        let alpha = rz_old / p_ap;
        x = x.add(&p.scale(alpha));
        r = r.sub(&ap.scale(alpha));

        let r_norm = r.norm();
        residuals.push(r_norm);

        if r_norm < tol {
            converged = true;
            return CGResult { x, residuals, iterations: k + 1, converged };
        }

        z = preconditioner(&r);
        let rz_new = r.dot(&z);
        let beta = rz_new / rz_old;
        p = z.add(&p.scale(beta));
        rz_old = rz_new;
    }

    CGResult { x, residuals, iterations: max_iter, converged }
}

// ========== GMRES ==========

/// Result of the GMRES method.
#[derive(Debug, Clone)]
pub struct GMRESResult {
    /// Solution vector.
    pub x: Vector,
    /// Residual norm history.
    pub residuals: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the method converged.
    pub converged: bool,
}

/// Solve Ax = b using GMRES (restarted).
///
/// Works for general (non-symmetric, non-positive definite) systems.
/// `restart` is the number of iterations before restarting (0 = no restart).
pub fn gmres(
    a: &Matrix,
    b: &Vector,
    max_iter: usize,
    tol: f64,
    restart: usize,
) -> GMRESResult {
    let n = b.len();
    let restart = if restart == 0 { n } else { restart.min(n) };
    let mut x = Vector::zeros(n);
    let mut residuals = vec![];
    let mut total_iter = 0;
    let mut converged = false;

    loop {
        let r0 = b.sub(&a.mul_vec(&x));
        let beta = r0.norm();

        residuals.push(beta);

        if beta < tol {
            converged = true;
            return GMRESResult { x, residuals, iterations: total_iter, converged };
        }

        // Arnoldi process
        let mut v = vec![r0.scale(1.0 / beta)];
        let mut h = vec![vec![0.0; restart + 1]; restart + 1];
        let mut g = vec![0.0; restart + 1];
        g[0] = beta;

        let mut cs = vec![0.0; restart];
        let mut sn = vec![0.0; restart];
        let mut k = 0;

        for j in 0..restart {
            k = j;
            let w = a.mul_vec(&v[j]);

            // Modified Gram-Schmidt
            let mut h_col = vec![0.0; j + 2];
            for i in 0..=j {
                h_col[i] = w.dot(&v[i]);
            }
            let mut w_new = w.clone();
            for i in 0..=j {
                w_new = w_new.sub(&v[i].scale(h_col[i]));
            }
            h_col[j + 1] = w_new.norm();

            for i in 0..=j {
                h[i][j] = h_col[i];
            }
            h[j + 1][j] = h_col[j + 1];

            if h_col[j + 1].abs() > 1e-15 {
                v.push(w_new.scale(1.0 / h_col[j + 1]));
            }

            // Apply previous Givens rotations
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = temp;
            }

            // New Givens rotation
            let rr = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if rr.abs() < 1e-30 {
                continue;
            }
            cs[j] = h[j][j] / rr;
            sn[j] = h[j + 1][j] / rr;
            h[j][j] = rr;
            h[j + 1][j] = 0.0;

            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];

            let residual = g[j + 1].abs();
            residuals.push(residual);
            total_iter += 1;

            if residual < tol {
                converged = true;
                // Solve upper triangular system for y
                let y = back_sub_h(&h, &g, j + 1);
                // Update x: x = x + V(:,0:j) * y
                for i in 0..=j {
                    x = x.add(&v[i].scale(y[i]));
                }
                return GMRESResult { x, residuals, iterations: total_iter, converged };
            }

            if total_iter >= max_iter {
                break;
            }
        }

        // Restart: solve least squares and update x
        let y = back_sub_h(&h, &g, k + 1);
        for i in 0..=k {
            x = x.add(&v[i].scale(y[i]));
        }

        if total_iter >= max_iter {
            break;
        }
    }

    GMRESResult { x, residuals, iterations: total_iter, converged }
}

/// Back substitution for upper Hessenberg matrix.
fn back_sub_h(h: &[Vec<f64>], g: &[f64], k: usize) -> Vec<f64> {
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = g[i];
        for j in (i + 1)..k {
            sum -= h[i][j] * y[j];
        }
        if h[i][i].abs() > 1e-30 {
            y[i] = sum / h[i][i];
        }
    }
    y
}

// ========== Preconditioners ==========

/// Jacobi (diagonal) preconditioner: M^{-1} = diag(1/a_ii).
pub fn jacobi_preconditioner(a: &Matrix) -> Box<dyn Fn(&Vector) -> Vector> {
    let n = a.rows;
    let mut diag_inv = vec![0.0; n];
    for i in 0..n {
        let d = a.get(i, i);
        diag_inv[i] = if d.abs() > 1e-15 { 1.0 / d } else { 0.0 };
    }
    Box::new(move |v: &Vector| {
        Vector::new(v.data.iter().zip(diag_inv.iter()).map(|(&vi, &di)| vi * di).collect())
    })
}

/// Gauss-Seidel preconditioner: applies one sweep of Gauss-Seidel.
pub fn gauss_seidel_preconditioner(a: &Matrix) -> Box<dyn Fn(&Vector) -> Vector> {
    let n = a.rows;
    let a_data = a.data.clone();
    Box::new(move |r: &Vector| {
        let mut z = vec![0.0; n];
        for i in 0..n {
            let mut sum = r.data[i];
            for j in 0..n {
                if j != i {
                    sum -= a_data[i * n + j] * z[j];
                }
            }
            let diag = a_data[i * n + i];
            z[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
        }
        Vector::new(z)
    })
}

/// Identity preconditioner (no preconditioning).
pub fn identity_preconditioner() -> Box<dyn Fn(&Vector) -> Vector> {
    Box::new(|v: &Vector| v.clone())
}

// ========== Additional Iterative Methods ==========

/// Solve Ax = b using the Gauss-Seidel method.
pub fn gauss_seidel(
    a: &Matrix,
    b: &Vector,
    max_iter: usize,
    tol: f64,
) -> (Vector, bool, usize) {
    let n = b.len();
    let mut x = Vector::zeros(n);

    for k in 0..max_iter {
        let mut max_diff = 0.0;
        for i in 0..n {
            let mut sum = b.get(i);
            for j in 0..n {
                if j != i {
                    sum -= a.get(i, j) * x.get(j);
                }
            }
            let diag = a.get(i, i);
            if diag.abs() > 1e-15 {
                let new_val = sum / diag;
                let diff = (new_val - x.get(i)).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                x.set(i, new_val);
            }
        }
        if max_diff < tol {
            return (x, true, k + 1);
        }
    }

    (x, false, max_iter)
}

/// Solve Ax = b using the Jacobi method.
pub fn jacobi(
    a: &Matrix,
    b: &Vector,
    max_iter: usize,
    tol: f64,
) -> (Vector, bool, usize) {
    let n = b.len();
    let mut x = Vector::zeros(n);

    for k in 0..max_iter {
        let mut x_new = vec![0.0; n];
        let mut max_diff = 0.0;

        for i in 0..n {
            let mut sum = b.get(i);
            for j in 0..n {
                if j != i {
                    sum -= a.get(i, j) * x.get(j);
                }
            }
            let diag = a.get(i, i);
            x_new[i] = if diag.abs() > 1e-15 { sum / diag } else { x.get(i) };
            let diff = (x_new[i] - x.get(i)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        x = Vector::new(x_new);

        if max_diff < tol {
            return (x, true, k + 1);
        }
    }

    (x, false, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd_matrix() -> Matrix {
        // [[4, 1], [1, 3]] — symmetric positive definite
        Matrix::from_2d(&[vec![4.0, 1.0], vec![1.0, 3.0]])
    }

    fn nonsymmetric_matrix() -> Matrix {
        Matrix::from_2d(&[vec![2.0, 1.0], vec![0.0, 3.0]])
    }

    #[test]
    fn test_cg_simple() {
        let a = spd_matrix();
        let b = Vector::new(vec![5.0, 4.0]);
        let result = conjugate_gradient(&a, &b, 100, 1e-10);
        assert!(result.converged);
        let ax = a.mul_vec(&result.x);
        assert!((ax.get(0) - 5.0).abs() < 1e-8);
        assert!((ax.get(1) - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_cg_identity() {
        let a = Matrix::identity(3);
        let b = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = conjugate_gradient(&a, &b, 100, 1e-10);
        assert!(result.converged);
        assert!((result.x.get(0) - 1.0).abs() < 1e-10);
        assert!((result.x.get(1) - 2.0).abs() < 1e-10);
        assert!((result.x.get(2) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_residuals_decrease() {
        let a = spd_matrix();
        let b = Vector::new(vec![5.0, 4.0]);
        let result = conjugate_gradient(&a, &b, 100, 1e-10);
        for i in 1..result.residuals.len() {
            assert!(result.residuals[i] <= result.residuals[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_cg_3x3() {
        // SPD 3x3
        let a = Matrix::from_2d(&[vec![4.0, 1.0, 0.0], vec![1.0, 3.0, 1.0], vec![0.0, 1.0, 4.0]]);
        let b = Vector::new(vec![5.0, 5.0, 5.0]);
        let result = conjugate_gradient(&a, &b, 100, 1e-10);
        assert!(result.converged);
        let ax = a.mul_vec(&result.x);
        for i in 0..3 {
            assert!((ax.get(i) - b.get(i)).abs() < 1e-8);
        }
    }

    #[test]
    fn test_preconditioned_cg() {
        let a = spd_matrix();
        let b = Vector::new(vec![5.0, 4.0]);
        let prec = jacobi_preconditioner(&a);
        let result = preconditioned_cg(&a, &b, &prec, 100, 1e-10);
        assert!(result.converged);
        let ax = a.mul_vec(&result.x);
        assert!((ax.get(0) - 5.0).abs() < 1e-6);
        assert!((ax.get(1) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_gmres_simple() {
        let a = nonsymmetric_matrix();
        let b = Vector::new(vec![5.0, 6.0]);
        let result = gmres(&a, &b, 100, 1e-8, 0);
        assert!(result.converged);
        let ax = a.mul_vec(&result.x);
        assert!((ax.get(0) - 5.0).abs() < 1e-6);
        assert!((ax.get(1) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_gmres_spd() {
        let a = spd_matrix();
        let b = Vector::new(vec![5.0, 4.0]);
        let result = gmres(&a, &b, 100, 1e-8, 10);
        assert!(result.converged);
        let ax = a.mul_vec(&result.x);
        assert!((ax.get(0) - 5.0).abs() < 1e-6);
        assert!((ax.get(1) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let a = spd_matrix();
        let prec = jacobi_preconditioner(&a);
        let v = Vector::new(vec![2.0, 3.0]);
        let result = prec(&v);
        assert!((result.data[0] - 0.5).abs() < 1e-10); // 2 / 4
        assert!((result.data[1] - 1.0).abs() < 1e-10); // 3 / 3
    }

    #[test]
    fn test_gauss_seidel_preconditioner() {
        let a = spd_matrix();
        let prec = gauss_seidel_preconditioner(&a);
        let r = Vector::new(vec![4.0, 3.0]);
        let z = prec(&r);
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn test_gauss_seidel_method() {
        let a = spd_matrix();
        let b = Vector::new(vec![5.0, 4.0]);
        let (x, converged, _) = gauss_seidel(&a, &b, 100, 1e-10);
        assert!(converged);
        let ax = a.mul_vec(&x);
        assert!((ax.get(0) - 5.0).abs() < 1e-6);
        assert!((ax.get(1) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_jacobi_method() {
        let a = spd_matrix();
        let b = Vector::new(vec![5.0, 4.0]);
        let (x, converged, _) = jacobi(&a, &b, 200, 1e-8);
        assert!(converged);
        let ax = a.mul_vec(&x);
        assert!((ax.get(0) - 5.0).abs() < 1e-4);
        assert!((ax.get(1) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_identity_preconditioner() {
        let prec = identity_preconditioner();
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = prec(&v);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cg_converges_fast_spd() {
        let a = Matrix::identity(2);
        let b = Vector::new(vec![1.0, 1.0]);
        let result = conjugate_gradient(&a, &b, 100, 1e-10);
        assert!(result.iterations <= 2);
    }
}
