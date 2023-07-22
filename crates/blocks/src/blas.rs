use cblas;
use ndarray::{ArrayView2, ArrayViewMut2};

#[inline(always)]
pub(crate) fn sgemm(transa: cblas::Transpose, transb: cblas::Transpose,
    alpha:f32, a:&[f32], b:&[f32], beta:f32, c: &mut [f32],
    m: usize, n: usize, k:usize, lda: usize, ldb: usize, ldc: usize) {
    unsafe {
        cblas::sgemm(cblas::Layout::RowMajor, transa, transb, m as i32, n as i32, 
            k as i32, alpha, a, lda as i32, b, ldb as i32, beta, c, ldc as i32)
    }
}

#[inline(always)]
pub(crate) fn gemm(transa: cblas::Transpose, transb: cblas::Transpose,
    alpha:f32, a: ArrayView2<f32>, b: ArrayView2<f32>, beta:f32,
    mut c: ArrayViewMut2<f32>) {
    let c_shape = c.shape();
    let (m,n) = (c_shape[0],c_shape[1]);
    let ldc = c_shape[1];
    let a_shape = a.shape();
    let lda = a_shape[1];
    let b_shape = b.shape();
    let ldb = b_shape[1];
    let (m_check, k) = match transa {
        cblas::Transpose::None => (a_shape[0],a_shape[1]),
        cblas::Transpose::Ordinary => (a_shape[1],a_shape[0]),
        cblas::Transpose::Conjugate => (a_shape[1],a_shape[0]),
    };
    let (k_check,n_check) = match transb {
        cblas::Transpose::None => (b_shape[0],b_shape[1]),
        cblas::Transpose::Ordinary => (b_shape[1],b_shape[0]),
        cblas::Transpose::Conjugate => (b_shape[1],b_shape[0]),
    };
    assert!(m==m_check && n==n_check && k==k_check,
    "shape mismatch, provided shapes are a - {:#?}, b - {:#?}, c - {:#?}",a.shape(),b.shape(),c.shape());
    sgemm(transa, transb, alpha, a.as_slice().unwrap(), b.as_slice().unwrap(), beta, 
        c.as_slice_mut().unwrap(), m, n, k, lda, ldb, ldc);
}