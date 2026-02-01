
#include "gpuR/windows_check.hpp"

// eigen headers for handling the R input data
#include <RcppEigen.h>

#include "gpuR/dynSparseMat.hpp"
#include "gpuR/dynEigenMat.hpp"
#include "gpuR/dynVCLMat.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Rcpp;

/*** Sparse Matrix Conversion and Creation ***/

template <typename T>
Rcpp::XPtr<gpuR::dynSparseMat<T>>
cpp_gpuSparseMatrix_template(SEXP x_,
                             int ctx_id,
                             int platform_id,
                             int device_id) {
    
    // Get ViennaCL context from OpenCL backend
    viennacl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // Try to treat input as sparse matrix from Matrix package
    Rcpp::S4 A(x_);
    std::string cl = A.slot("class");
    
    // Initialize sparse matrix container
    auto spmat = std::make_shared<gpuR::dynSparseMat<T>>(ctx);
    auto mat = std::make_shared<viennacl::compressed_matrix<T>>(ctx);
    
    // Convert from various sparse formats
    if (cl == "dgCMatrix" || cl == "dsCMatrix" || cl == "dtCMatrix" ||
        cl == "igCMatrix" || cl == "isCMatrix" || cl == "itCMatrix" ||
        cl == "ngCMatrix" || cl == "nsCMatrix" || cl == "ntCMatrix") {
        
        // CSR format expected (i, p, x slots)
        Rcpp::IntegerVector i = A.slot("i");
        Rcpp::IntegerVector p = A.slot("p");
        Rcpp::NumericVector x = A.slot("x");
        Rcpp::IntegerVector dims = A.slot("Dim");
        
        int m = dims[0];
        int n = dims[1];
        int nnz = x.size();
        
        // Convert column indices from 0-based to 0-based (already correct)
        // p contains row pointers for CSC, need to convert to CSR if necessary
        
        // Create ViennaCL compressed matrix
        std::vector<T> values(nnz);
        std::vector<unsigned int> column_indices(nnz);
        std::vector<unsigned int> row_pointers(m + 1);
        
        // Copy values and column indices
        for (int j = 0; j < nnz; j++) {
            values[j] = static_cast<T>(x[j]);
            column_indices[j] = static_cast<unsigned int>(i[j]);
        }
        
        // Copy row pointers
        for (int j = 0; j <= m; j++) {
            row_pointers[j] = static_cast<unsigned int>(p[j]);
        }
        
        // Create compressed matrix on GPU
        viennacl::backend::typesafe_host_array<unsigned int> row_buffer(
            mat->handle1(), row_pointers.size());
        viennacl::backend::typesafe_host_array<unsigned int> col_buffer(
            mat->handle2(), column_indices.size());
        
        for (size_t i = 0; i < row_pointers.size(); i++) {
            row_buffer.set(i, row_pointers[i]);
        }
        for (size_t i = 0; i < column_indices.size(); i++) {
            col_buffer.set(i, column_indices[i]);
        }
        
        mat->set(row_buffer.get(), col_buffer.get(), values.data(), m, n, nnz);
        
        auto result = std::make_shared<gpuR::dynSparseMat<T>>(m, n, nnz, ctx);
        *result->getMatrixPtr() = *mat;
        
        return Rcpp::XPtr<gpuR::dynSparseMat<T>>(
            new gpuR::dynSparseMat<T>(m, n, nnz, ctx), true);
    }
    
    // If not recognized as sparse, try as dense matrix
    try {
        Rcpp::NumericMatrix dense(x_);
        int m = dense.nrow();
        int n = dense.ncol();
        int nnz = 0;
        
        // Count non-zeros
        std::vector<T> values;
        std::vector<unsigned int> column_indices;
        std::vector<unsigned int> row_pointers(m + 1, 0);
        
        for (int i = 0; i < m; i++) {
            row_pointers[i] = nnz;
            for (int j = 0; j < n; j++) {
                if (dense(i, j) != 0.0) {
                    values.push_back(static_cast<T>(dense(i, j)));
                    column_indices.push_back(j);
                    nnz++;
                }
            }
        }
        row_pointers[m] = nnz;
        
        // Create compressed matrix
        viennacl::backend::typesafe_host_array<unsigned int> row_buffer(
            mat->handle1(), row_pointers.size());
        viennacl::backend::typesafe_host_array<unsigned int> col_buffer(
            mat->handle2(), column_indices.size());
        
        for (size_t i = 0; i < row_pointers.size(); i++) {
            row_buffer.set(i, row_pointers[i]);
        }
        for (size_t i = 0; i < column_indices.size(); i++) {
            col_buffer.set(i, column_indices[i]);
        }
        
        mat->set(row_buffer.get(), col_buffer.get(), values.data(), m, n, nnz);
        
        auto result = std::make_shared<gpuR::dynSparseMat<T>>(m, n, nnz, ctx);
        *result->getMatrixPtr() = *mat;
        
        return Rcpp::XPtr<gpuR::dynSparseMat<T>>(
            new gpuR::dynSparseMat<T>(m, n, nnz, ctx), true);
    } catch (...) {
        stop("Input must be a sparse matrix or numeric matrix");
    }
}


/*** Sparse Matrix Export Functions ***/

// [[Rcpp::export]]
SEXP cpp_gpuSparseMatrix_double(SEXP x,
                                 int ctx_id,
                                 int platform_id,
                                 int device_id) {
    return wrap(cpp_gpuSparseMatrix_template<double>(x, ctx_id, platform_id, device_id));
}

// [[Rcpp::export]]
SEXP cpp_gpuSparseMatrix_float(SEXP x,
                                int ctx_id,
                                int platform_id,
                                int device_id) {
    return wrap(cpp_gpuSparseMatrix_template<float>(x, ctx_id, platform_id, device_id));
}

// [[Rcpp::export]]
SEXP cpp_gpuSparseMatrix_int(SEXP x,
                              int ctx_id,
                              int platform_id,
                              int device_id) {
    // Integer types not supported in ViennaCL sparse matrices
    Rcpp::stop("Integer sparse matrices not yet supported");
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP cpp_gpuSparseMatrix_fcomplex(SEXP x,
                                   int ctx_id,
                                   int platform_id,
                                   int device_id) {
    // Complex types not supported in ViennaCL sparse matrices yet
    Rcpp::stop("Complex sparse matrices not yet supported");
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP cpp_gpuSparseMatrix_dcomplex(SEXP x,
                                   int ctx_id,
                                   int platform_id,
                                   int device_id) {
    // Complex types not supported in ViennaCL sparse matrices yet
    Rcpp::stop("Complex sparse matrices not yet supported");
    return R_NilValue;
}


/*** Sparse Matrix Queries ***/

template <typename T>
Rcpp::IntegerVector cpp_gpuSparseMatrix_dims_template(SEXP x_) {
    XPtr<gpuR::dynSparseMat<T>> x(x_);
    return IntegerVector::create(x->rows(), x->cols());
}

// [[Rcpp::export]]
Rcpp::IntegerVector cpp_gpuSparseMatrix_dims(SEXP x) {
    // Determine type and dispatch
    Rcpp::XPtr<gpuR::dynSparseMat<double>> temp(x);
    return cpp_gpuSparseMatrix_dims_template<double>(x);
}

template <typename T>
int cpp_gpuSparseMatrix_nrow_template(SEXP x_) {
    XPtr<gpuR::dynSparseMat<T>> x(x_);
    return x->rows();
}

// [[Rcpp::export]]
int cpp_gpuSparseMatrix_nrow(SEXP x) {
    return cpp_gpuSparseMatrix_nrow_template<double>(x);
}

template <typename T>
int cpp_gpuSparseMatrix_ncol_template(SEXP x_) {
    XPtr<gpuR::dynSparseMat<T>> x(x_);
    return x->cols();
}

// [[Rcpp::export]]
int cpp_gpuSparseMatrix_ncol(SEXP x) {
    return cpp_gpuSparseMatrix_ncol_template<double>(x);
}

template <typename T>
int cpp_gpuSparseMatrix_nnz_template(SEXP x_) {
    XPtr<gpuR::dynSparseMat<T>> x(x_);
    return x->nnz();
}

// [[Rcpp::export]]
int cpp_gpuSparseMatrix_nnz(SEXP x) {
    return cpp_gpuSparseMatrix_nnz_template<double>(x);
}


/*** Sparse Matrix to Dense Conversion ***/

template <typename T>
Rcpp::NumericMatrix cpp_gpuSparseMatrix_to_matrix_template(SEXP x_) {
    XPtr<gpuR::dynSparseMat<T>> x(x_);
    
    viennacl::compressed_matrix<T>* mat = x->getMatrix();
    
    // Create dense matrix
    Rcpp::NumericMatrix result(x->rows(), x->cols());
    
    // Initialize to zero
    std::fill(result.begin(), result.end(), 0.0);
    
    // Copy data from GPU (simplified - in real implementation would use viennacl's copy functions)
    // For now, return empty matrix as placeholder
    return result;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_gpuSparseMatrix_to_matrix(SEXP x) {
    return cpp_gpuSparseMatrix_to_matrix_template<double>(x);
}


/*** Sparse Matrix Operations ***/

template <typename T>
SEXP cpp_gpuSparseMat_mult_template(SEXP ptrA_, SEXP ptrB_) {
    XPtr<gpuR::dynSparseMat<T>> ptrA(ptrA_);
    XPtr<gpuR::dynSparseMat<T>> ptrB(ptrB_);
    
    // ViennaCL sparse-sparse multiplication not directly available
    // TODO: Implement custom sparse matrix multiplication kernel
    Rcpp::warning("Sparse-sparse multiplication not yet implemented");
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP cpp_gpuSparseMat_mult(SEXP ptrA, SEXP ptrB) {
    return cpp_gpuSparseMat_mult_template<double>(ptrA, ptrB);
}


template <typename T>
SEXP cpp_gpuSparseGpuDenseMat_mult_template(SEXP ptrA_, SEXP ptrB_) {
    XPtr<gpuR::dynSparseMat<T>> ptrA(ptrA_);
    XPtr<dynEigenMat<T>> ptrB(ptrB_);
    
    // ViennaCL doesn't provide sparse-dense prod directly
    // TODO: Implement custom kernel or fallback
    Rcpp::warning("Sparse-dense multiplication not yet implemented");
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP cpp_gpuSparseGpuDenseMat_mult(SEXP ptrA, SEXP ptrB) {
    return cpp_gpuSparseGpuDenseMat_mult_template<double>(ptrA, ptrB);
}


template <typename T>
SEXP cpp_gpuDenseGpuSparseMat_mult_template(SEXP ptrA_, SEXP ptrB_) {
    XPtr<dynEigenMat<T>> ptrA(ptrA_);
    XPtr<gpuR::dynSparseMat<T>> ptrB(ptrB_);
    
    // ViennaCL doesn't provide dense-sparse prod directly
    // TODO: Implement custom kernel or fallback
    Rcpp::warning("Dense-sparse multiplication not yet implemented");
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP cpp_gpuDenseGpuSparseMat_mult(SEXP ptrA, SEXP ptrB) {
    return cpp_gpuDenseGpuSparseMat_mult_template<double>(ptrA, ptrB);
}


/*** Sparse Matrix Transpose ***/

template <typename T>
SEXP cpp_gpuSparseMat_transpose_template(SEXP ptrA_) {
    XPtr<gpuR::dynSparseMat<T>> ptrA(ptrA_);
    
    viennacl::compressed_matrix<T>* mat_A = ptrA->getMatrix();
    
    // Create transposed sparse matrix with swapped dimensions
    size_t new_rows = mat_A->size2();
    size_t new_cols = mat_A->size1();
    size_t nnz = mat_A->nnz();
    
    // Create result sparse matrix with transposed dimensions
    // Use default context for now
    viennacl::context ctx(viennacl::ocl::get_context(0));
    auto result = new gpuR::dynSparseMat<T>(new_rows, new_cols, nnz, ctx);
    
    // TODO: Implement actual transpose operation on GPU
    
    return wrap(Rcpp::XPtr<gpuR::dynSparseMat<T>>(result, true));
}

// [[Rcpp::export]]
SEXP cpp_gpuSparseMat_transpose(SEXP ptrA) {
    return cpp_gpuSparseMat_transpose_template<double>(ptrA);
}


/*** Sparse Matrix Norms ***/

template <typename T>
double cpp_gpuSparseMat_norm_frobenius_template(SEXP ptrA_) {
    XPtr<gpuR::dynSparseMat<T>> ptrA(ptrA_);
    
    // Compute Frobenius norm: sqrt(sum of squares of all elements)
    // norm_F = sqrt(sum(mat_A[i,j]^2)) for all non-zero elements
    
    double norm_squared = 0.0;
    
    // TODO: Access sparse matrix data and compute norm
    Rcpp::warning("Frobenius norm computation for sparse matrices not yet implemented");
    
    return std::sqrt(norm_squared);
}

// [[Rcpp::export]]
double cpp_gpuSparseMat_norm_frobenius(SEXP ptrA) {
    return cpp_gpuSparseMat_norm_frobenius_template<double>(ptrA);
}
