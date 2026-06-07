//! Vector operations: arithmetic, dot/cross products, norms, and utilities.

/// A vector of f64 values.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// The vector components.
    pub data: Vec<f64>,
}

impl Vector {
    /// Create a new vector from a slice.
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Create a zero vector of the given dimension.
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    /// Create a vector of all ones.
    pub fn ones(n: usize) -> Self {
        Self { data: vec![1.0; n] }
    }

    /// Create a basis vector e_i (1 in position i, 0 elsewhere).
    pub fn basis(n: usize, i: usize) -> Self {
        let mut data = vec![0.0; n];
        data[i] = 1.0;
        Self { data }
    }

    /// Dimension of the vector.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index.
    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    /// Set element at index.
    pub fn set(&mut self, i: usize, val: f64) {
        self.data[i] = val;
    }

    /// Dot product of two vectors.
    pub fn dot(&self, other: &Vector) -> f64 {
        self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
    }

    /// Cross product (only defined for 3D vectors).
    ///
    /// # Panics
    /// Panics if either vector is not 3-dimensional.
    pub fn cross(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), 3, "cross product requires 3D vectors");
        assert_eq!(other.len(), 3, "cross product requires 3D vectors");
        Vector::new(vec![
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        ])
    }

    /// Euclidean (L2) norm.
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Squared norm (avoids sqrt).
    pub fn norm_sq(&self) -> f64 {
        self.dot(self)
    }

    /// L1 norm (sum of absolute values).
    pub fn norm_l1(&self) -> f64 {
        self.data.iter().map(|x| x.abs()).sum()
    }

    /// L∞ norm (maximum absolute value).
    pub fn norm_inf(&self) -> f64 {
        self.data.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
    }

    /// Normalize to unit length.
    ///
    /// Returns `None` if the vector is zero.
    pub fn normalize(&self) -> Option<Vector> {
        let n = self.norm();
        if n < 1e-15 {
            None
        } else {
            Some(self.scale(1.0 / n))
        }
    }

    /// Scale the vector by a scalar.
    pub fn scale(&self, s: f64) -> Vector {
        Vector::new(self.data.iter().map(|x| x * s).collect())
    }

    /// Add two vectors.
    pub fn add(&self, other: &Vector) -> Vector {
        Vector::new(self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect())
    }

    /// Subtract two vectors.
    pub fn sub(&self, other: &Vector) -> Vector {
        Vector::new(self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect())
    }

    /// Element-wise multiplication.
    pub fn hadamard(&self, other: &Vector) -> Vector {
        Vector::new(self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect())
    }

    /// Angle between two vectors in radians.
    pub fn angle(&self, other: &Vector) -> f64 {
        let d = self.dot(other);
        let n = self.norm() * other.norm();
        if n < 1e-15 {
            0.0
        } else {
            (d / n).clamp(-1.0, 1.0).acos()
        }
    }

    /// Project self onto other.
    pub fn project_onto(&self, other: &Vector) -> Vector {
        let d = self.dot(other);
        let n = other.dot(other);
        if n.abs() < 1e-15 {
            Vector::zeros(self.len())
        } else {
            other.scale(d / n)
        }
    }
}

/// Compute the distance between two vectors.
pub fn distance(a: &Vector, b: &Vector) -> f64 {
    a.sub(b).norm()
}

/// Compute the sum of all elements.
pub fn sum(v: &Vector) -> f64 {
    v.data.iter().sum()
}

/// Compute the mean of all elements.
pub fn mean(v: &Vector) -> f64 {
    if v.is_empty() { 0.0 } else { sum(v) / v.len() as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let v = Vector::zeros(3);
        assert_eq!(v.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let v = Vector::ones(2);
        assert_eq!(v.data, vec![1.0, 1.0]);
    }

    #[test]
    fn test_basis() {
        let e = Vector::basis(3, 1);
        assert_eq!(e.data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_dot() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross() {
        let a = Vector::new(vec![1.0, 0.0, 0.0]);
        let b = Vector::new(vec![0.0, 1.0, 0.0]);
        let c = a.cross(&b);
        assert!((c.data[0] - 0.0).abs() < 1e-10);
        assert!((c.data[1] - 0.0).abs() < 1e-10);
        assert!((c.data[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm() {
        let v = Vector::new(vec![3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = Vector::new(vec![3.0, 4.0]);
        let n = v.normalize().unwrap();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v = Vector::zeros(3);
        assert!(v.normalize().is_none());
    }

    #[test]
    fn test_add_sub() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![3.0, 4.0]);
        let sum = a.add(&b);
        assert_eq!(sum.data, vec![4.0, 6.0]);
        let diff = a.sub(&b);
        assert_eq!(diff.data, vec![-2.0, -2.0]);
    }

    #[test]
    fn test_scale() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let s = v.scale(2.0);
        assert_eq!(s.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_angle() {
        let a = Vector::new(vec![1.0, 0.0]);
        let b = Vector::new(vec![0.0, 1.0]);
        assert!((a.angle(&b) - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_project() {
        let a = Vector::new(vec![3.0, 4.0]);
        let b = Vector::new(vec![1.0, 0.0]);
        let p = a.project_onto(&b);
        assert!((p.data[0] - 3.0).abs() < 1e-10);
        assert!((p.data[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_l1_linf() {
        let v = Vector::new(vec![-3.0, 4.0, -5.0]);
        assert!((v.norm_l1() - 12.0).abs() < 1e-10);
        assert!((v.norm_inf() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let a = Vector::new(vec![0.0, 0.0]);
        let b = Vector::new(vec![3.0, 4.0]);
        assert!((distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean() {
        let v = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert!((mean(&v) - 2.5).abs() < 1e-10);
    }
}
