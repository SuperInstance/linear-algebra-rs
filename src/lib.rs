//! # linear-algebra-rs
//!
//! A pure-Rust linear algebra library with no external dependencies.
//!
//! ## Modules
//!
//! - [`vector`] — Vector operations (add, subtract, dot, cross, norm)
//! - [`matrix`] — Matrix operations (multiply, transpose, determinant, inverse)
//! - [`elimination`] — Gaussian elimination and LU decomposition
//! - [`eigen`] — Power method eigenvalue computation
//! - [`svd`] — 2×2 Singular Value Decomposition

pub mod eigen;
pub mod elimination;
pub mod matrix;
pub mod svd;
pub mod vector;

pub use matrix::Matrix;
pub use vector::Vector;
