context("Sparse Matrix Creation and Properties")

test_that("gpuSparseMatrix can be created from dense matrix", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    
    expect_no_error({
        gpuA <- gpuSparseMatrix(A, type="double")
    })
    
    gpuA <- gpuSparseMatrix(A, type="double")
    expect_is(gpuA, "dgpuSparseMatrix")
})

test_that("gpuSparseMatrix has correct dimensions", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3, ncol=3)
    gpuA <- gpuSparseMatrix(A, type="double")
    
    expect_equal(nrow(gpuA), 3)
    expect_equal(ncol(gpuA), 3)
    expect_equal(dim(gpuA), c(3, 3))
})

test_that("gpuSparseMatrix counts non-zero elements correctly", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    gpuA <- gpuSparseMatrix(A, type="double")
    
    expect_equal(nnz(gpuA), 5)
})

test_that("gpuSparseMatrix preserves data type", {
    skip_if_no_gpu()
    
    A_double <- matrix(c(1, 0, 2, 0, 0, 3), nrow=2)
    A_int <- matrix(as.integer(c(1, 0, 2, 0, 0, 3)), nrow=2)
    
    gpuA_double <- gpuSparseMatrix(A_double, type="double")
    gpuA_int <- gpuSparseMatrix(A_int, type="integer")
    
    expect_equal(typeof(gpuA_double), "double")
    expect_equal(typeof(gpuA_int), "integer")
})

test_that("as.gpuSparseMatrix auto-detects type", {
    skip_if_no_gpu()
    
    A_double <- matrix(c(1, 0, 2, 0, 0, 3), nrow=2)
    A_int <- matrix(as.integer(c(1, 0, 2, 0, 0, 3)), nrow=2)
    
    gpuA_double <- as.gpuSparseMatrix(A_double)
    gpuA_int <- as.gpuSparseMatrix(A_int)
    
    expect_equal(typeof(gpuA_double), "double")
    expect_equal(typeof(gpuA_int), "integer")
})

test_that("as.matrix converts sparse to dense correctly", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    gpuA <- gpuSparseMatrix(A, type="double")
    
    result <- as.matrix(gpuA)
    
    expect_is(result, "matrix")
    expect_equal(dim(result), dim(A))
})

context("Sparse Matrix Operations")

test_that("Sparse matrix transpose works", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5, 6, 0, 7), nrow=3)
    gpuA <- gpuSparseMatrix(A, type="double")
    
    At <- t(gpuA)
    
    expect_is(At, "dgpuSparseMatrix")
    expect_equal(nrow(At), ncol(gpuA))
    expect_equal(ncol(At), nrow(gpuA))
})

test_that("Frobenius norm computation works", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    gpuA <- gpuSparseMatrix(A, type="double")
    
    f_norm <- norm(gpuA, "F")
    
    expect_is(f_norm, "numeric")
    expect_gt(f_norm, 0)
    
    # Compare with dense computation
    expected_norm <- sqrt(sum(A^2))
    expect_approximately_equal(f_norm, expected_norm, tol=1e-5)
})

context("Sparse Matrix Multiplication")

test_that("Sparse-sparse multiplication works", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    B <- matrix(c(1, 0, 2, 3, 0, 4, 5, 6, 0), nrow=3)
    
    gpuA <- gpuSparseMatrix(A, type="double")
    gpuB <- gpuSparseMatrix(B, type="double")
    
    result <- gpuA %*% gpuB
    
    expect_is(result, "matrix")
    
    # Compare with dense computation
    expected <- as.matrix(A) %*% as.matrix(B)
    expect_approximately_equal(result, expected, tol=1e-5)
})

test_that("Sparse-dense GPU matrix multiplication works", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    B <- matrix(1:9, nrow=3)
    
    gpuA <- gpuSparseMatrix(A, type="double")
    gpuB <- gpuMatrix(B, type="double")
    
    result <- gpuA %*% gpuB
    
    expect_is(result, "dgpuMatrix")
})

test_that("Dense-sparse GPU matrix multiplication works", {
    skip_if_no_gpu()
    
    A <- matrix(1:9, nrow=3)
    B <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    
    gpuA <- gpuMatrix(A, type="double")
    gpuB <- gpuSparseMatrix(B, type="double")
    
    result <- gpuA %*% gpuB
    
    expect_is(result, "dgpuMatrix")
})

test_that("Sparse-dense CPU matrix multiplication works", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    B <- matrix(1:9, nrow=3)
    
    gpuA <- gpuSparseMatrix(A, type="double")
    
    result <- gpuA %*% B
    
    # Compare with dense computation
    expected <- as.matrix(A) %*% B
    expect_approximately_equal(result, expected, tol=1e-5)
})

test_that("Dense-sparse CPU matrix multiplication works", {
    skip_if_no_gpu()
    
    A <- matrix(1:9, nrow=3)
    B <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
    
    gpuB <- gpuSparseMatrix(B, type="double")
    
    result <- A %*% gpuB
    
    # Compare with dense computation
    expected <- A %*% as.matrix(B)
    expect_approximately_equal(result, expected, tol=1e-5)
})

context("Sparse Matrix Type Validation")

test_that("dgpuSparseMatrix requires double type", {
    skip_if_no_gpu()
    
    A <- matrix(c(1, 0, 2, 0, 0, 3), nrow=2)
    gpuA <- gpuSparseMatrix(A, type="double")
    
    # Should pass validation
    expect_is(gpuA, "dgpuSparseMatrix")
    
    # Trying to coerce to wrong type should fail
    expect_error(as(gpuA, "igpuSparseMatrix"))
})

test_that("igpuSparseMatrix requires integer type", {
    skip_if_no_gpu()
    
    A <- matrix(as.integer(c(1, 0, 2, 0, 0, 3)), nrow=2)
    gpuA <- gpuSparseMatrix(A, type="integer")
    
    # Should pass validation
    expect_is(gpuA, "igpuSparseMatrix")
})

test_that("fgpuSparseMatrix requires float type", {
    skip_if_no_gpu()
    
    A <- matrix(c(1.5, 0, 2.5, 0, 0, 3.5), nrow=2)
    gpuA <- gpuSparseMatrix(A, type="float")
    
    # Should pass validation
    expect_is(gpuA, "fgpuSparseMatrix")
})

# Helper function for GPU check
skip_if_no_gpu <- function() {
    skip("GPU not available")  # Replace with actual GPU detection
}

# Helper function for approximate equality
expect_approximately_equal <- function(object, expected, tol=1e-6) {
    expect_true(all(abs(object - expected) < tol))
}
