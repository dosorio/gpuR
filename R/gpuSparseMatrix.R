#' Create GPU Sparse Matrix
#'
#' @description Create a sparse matrix object for GPU computation
#' using Compressed Sparse Row (CSR) format.
#'
#' @param x A sparse matrix (e.g., from \code{Matrix::sparseMatrix}, 
#' \code{Matrix::Matrix(sparse=TRUE)}, etc.) or a dense matrix to be sparsified,
#' or vectors for CSR construction: \code{values}, \code{row_indices}, \code{col_indices}
#' @param type A character string specifying the data type: "integer", "float", or "double"
#' @param ctx_id Integer index of the context on which to place the object
#' @param platform_index Integer index of the platform
#' @param device_index Integer index of the device
#' @param row_indices Vector of row indices for CSR construction (optional)
#' @param col_indices Vector of column indices for CSR construction (optional)
#' @param nrows Number of rows (required if constructing from vectors)
#' @param ncols Number of columns (required if constructing from vectors)
#'
#' @details
#' The function supports creation of sparse matrices in several ways:
#' \enumerate{
#'   \item From a sparse matrix object (Matrix package)
#'   \item From a dense matrix (will be automatically sparsified)
#'   \item From CSR components: values, row_indices, col_indices
#' }
#'
#' Sparse matrices are stored in Compressed Sparse Row (CSR) format for
#' optimal GPU performance.
#'
#' @return An object of class \code{dgpuSparseMatrix}, \code{fgpuSparseMatrix},
#' or \code{igpuSparseMatrix} depending on the specified type.
#'
#' @note R does not contain a native float type. As such, float sparse matrices
#' will be represented as double but downcast when any gpuSparseMatrix methods are used.
#'
#' @seealso \code{\link{as.gpuSparseMatrix}}, \code{\link{as.matrix.gpuSparseMatrix}}
#'
#' @examples
#' \dontrun{
#' # Create from dense matrix
#' A <- matrix(c(1, 0, 2, 0, 0, 3, 4, 0, 5), nrow=3)
#' gpuA <- gpuSparseMatrix(A, type="double")
#'
#' # Create from Matrix package sparse matrix
#' library(Matrix)
#' B <- Matrix::Matrix(A, sparse=TRUE)
#' gpuB <- gpuSparseMatrix(B, type="double")
#' }
#'
#' @export
gpuSparseMatrix <- function(x, type = "double", ctx_id = 0L,
                            platform_index = 0L, device_index = 0L,
                            row_indices = NULL, col_indices = NULL,
                            nrows = NULL, ncols = NULL) {
    
    # Determine type
    if (!(type %in% c("integer", "float", "double", "fcomplex", "dcomplex"))) {
        stop("type must be one of 'integer', 'float', 'double', 'fcomplex', or 'dcomplex'")
    }
    
    # Convert to sparse if needed
    if (is.matrix(x) && !requireNamespace("Matrix", quietly = TRUE)) {
        stop("Matrix package required for sparse matrix conversion")
    }
    
    if (is.matrix(x)) {
        x <- Matrix::Matrix(x, sparse = TRUE)
    }
    
    # Call appropriate C++ function based on type
    ptr <- switch(type,
           integer = cpp_gpuSparseMatrix_int(x, as.integer(ctx_id), 
                                            as.integer(platform_index),
                                            as.integer(device_index)),
           float = cpp_gpuSparseMatrix_float(x, as.integer(ctx_id),
                                            as.integer(platform_index),
                                            as.integer(device_index)),
           double = cpp_gpuSparseMatrix_double(x, as.integer(ctx_id),
                                              as.integer(platform_index),
                                              as.integer(device_index)),
           fcomplex = cpp_gpuSparseMatrix_fcomplex(x, as.integer(ctx_id),
                                                  as.integer(platform_index),
                                                  as.integer(device_index)),
           dcomplex = cpp_gpuSparseMatrix_dcomplex(x, as.integer(ctx_id),
                                                  as.integer(platform_index),
                                                  as.integer(device_index)))
    
    # Wrap in appropriate S4 class
    class_name <- switch(type,
                        integer = "igpuSparseMatrix",
                        float = "fgpuSparseMatrix",
                        double = "dgpuSparseMatrix",
                        fcomplex = "cgpuSparseMatrix",
                        dcomplex = "zgpuSparseMatrix")
    
    obj <- new(class_name,
               address = ptr,
               .context_index = as.integer(ctx_id),
               .platform_index = as.integer(platform_index),
               .platform = "default",
               .device_index = as.integer(device_index),
               .device = "default")
    
    return(obj)
}


#' Convert Matrix to GPU Sparse Matrix
#'
#' @description Convenience function to convert a matrix to a GPU sparse matrix
#' with automatic type detection.
#'
#' @param x A sparse or dense matrix
#' @param ctx_id Integer index of the context
#' @param platform_index Integer index of the platform
#' @param device_index Integer index of the device
#'
#' @details
#' Automatically detects the data type of the input matrix and creates
#' the appropriate GPU sparse matrix type.
#'
#' @return An appropriate gpuSparseMatrix object
#'
#' @export
as.gpuSparseMatrix <- function(x, ctx_id = 0L, platform_index = 0L, device_index = 0L) {
    
    # Determine type from x
    if (is.integer(x)) {
        type <- "integer"
    } else if (is.complex(x)) {
        type <- "dcomplex"
    } else {
        type <- "double"
    }
    
    gpuSparseMatrix(x, type = type, ctx_id = ctx_id,
                   platform_index = platform_index,
                   device_index = device_index)
}


#' Convert GPU Sparse Matrix to Dense Matrix
#'
#' @description Convert a GPU sparse matrix to a dense R matrix
#'
#' @param x A gpuSparseMatrix object
#' @param ... Additional arguments (for compatibility)
#'
#' @return A dense R matrix
#'
#' @export
as.matrix.gpuSparseMatrix <- function(x, ...) {
    cpp_gpuSparseMatrix_to_matrix(x@address)
}


#' Get Sparse Matrix Dimensions
#'
#' @description Get the dimensions of a GPU sparse matrix
#'
#' @param x A gpuSparseMatrix object
#'
#' @return A vector of length 2 with rows and columns
#'
#' @export
setMethod("dim", signature(x="gpuSparseMatrix"),
          function(x) {
              cpp_gpuSparseMatrix_dims(x@address)
          })


#' Get Number of Non-zero Elements
#'
#' @description Get the number of non-zero elements in a GPU sparse matrix
#'
#' @param x A gpuSparseMatrix object
#'
#' @return The count of non-zero elements
#'
#' @export
if (!isGeneric("nnz")) {
    setGeneric("nnz", function(x) standardGeneric("nnz"))
}

setMethod("nnz", signature(x="gpuSparseMatrix"),
          function(x) {
              cpp_gpuSparseMatrix_nnz(x@address)
          })


#' Get Matrix Type
#'
#' @description Get the data type of a GPU sparse matrix
#'
#' @param x A gpuSparseMatrix object
#'
#' @return A character string indicating the type
#'
#' @export
setMethod("typeof", signature(x="gpuSparseMatrix"),
          function(x) {
              if (is(x, "igpuSparseMatrix")) {
                  return("integer")
              } else if (is(x, "fgpuSparseMatrix")) {
                  return("float")
              } else if (is(x, "dgpuSparseMatrix")) {
                  return("double")
              } else if (is(x, "cgpuSparseMatrix")) {
                  return("fcomplex")
              } else if (is(x, "zgpuSparseMatrix")) {
                  return("dcomplex")
              }
              return("unknown")
          })


#' Get Number of Rows
#'
#' @description Get the number of rows in a GPU sparse matrix
#'
#' @param x A gpuSparseMatrix object
#'
#' @return Number of rows
#'
#' @export
setMethod("nrow", signature(x="gpuSparseMatrix"),
          function(x) {
              cpp_gpuSparseMatrix_nrow(x@address)
          })


#' Get Number of Columns
#'
#' @description Get the number of columns in a GPU sparse matrix
#'
#' @param x A gpuSparseMatrix object
#'
#' @return Number of columns
#'
#' @export
setMethod("ncol", signature(x="gpuSparseMatrix"),
          function(x) {
              cpp_gpuSparseMatrix_ncol(x@address)
          })


#' Print GPU Sparse Matrix
#'
#' @description Print method for GPU sparse matrices
#'
#' @param x A gpuSparseMatrix object
#'
#' @export
setMethod("print", signature(x="gpuSparseMatrix"),
          function(x) {
              cat(class(x), "object\n")
              cat("Dimensions:", nrow(x), "x", ncol(x), "\n")
              cat("Non-zeros:", nnz(x), "\n")
              cat("Type:", typeof(x), "\n")
              cat("Sparsity:", 100 * (1 - nnz(x) / (nrow(x) * ncol(x))), "%\n")
              invisible(x)
          })


#' Sparse Matrix Multiplication
#'
#' @description Multiply two GPU sparse matrices or a sparse matrix with a dense matrix
#'
#' @param x A gpuSparseMatrix object
#' @param y A gpuSparseMatrix, gpuMatrix, or matrix object
#'
#' @return Result of matrix multiplication
#'
#' @export
setMethod("%*%", signature(x="gpuSparseMatrix", y = "gpuSparseMatrix"),
          function(x, y) {
              if (ncol(x) != nrow(y)) {
                  stop("Non-conformable matrices")
              }
              cpp_gpuSparseMat_mult(x@address, y@address)
          })


#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="gpuSparseMatrix", y = "gpuMatrix"),
          function(x, y) {
              if (ncol(x) != nrow(y)) {
                  stop("Non-conformable matrices")
              }
              cpp_gpuSparseGpuDenseMat_mult(x@address, y@address)
          })


#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="gpuMatrix", y = "gpuSparseMatrix"),
          function(x, y) {
              if (ncol(x) != nrow(y)) {
                  stop("Non-conformable matrices")
              }
              cpp_gpuDenseGpuSparseMat_mult(x@address, y@address)
          })


#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="gpuSparseMatrix", y = "matrix"),
          function(x, y) {
              if (ncol(x) != nrow(y)) {
                  stop("Non-conformable matrices")
              }
              y_gpu <- gpuMatrix(y, type = typeof(x), ctx_id = x@.context_index)
              cpp_gpuSparseGpuDenseMat_mult(x@address, y_gpu@address)
          })


#' @rdname grapes-times-grapes-methods
#' @export
setMethod("%*%", signature(x="matrix", y = "gpuSparseMatrix"),
          function(x, y) {
              if (ncol(x) != nrow(y)) {
                  stop("Non-conformable matrices")
              }
              x_gpu <- gpuMatrix(x, type = typeof(y), ctx_id = y@.context_index)
              cpp_gpuDenseGpuSparseMat_mult(x_gpu@address, y@address)
          })


#' Sparse Matrix Transpose
#'
#' @description Transpose a GPU sparse matrix
#'
#' @param x A gpuSparseMatrix object
#'
#' @return Transposed sparse matrix
#'
#' @export
setMethod("t", signature(x="gpuSparseMatrix"),
          function(x) {
              cpp_gpuSparseMat_transpose(x@address)
          })


#' Sparse Matrix Norm
#'
#' @description Compute the Frobenius norm of a GPU sparse matrix
#'
#' @param x A gpuSparseMatrix object
#' @param type Type of norm ("F" for Frobenius)
#'
#' @return The computed norm value
#'
#' @export
setMethod("norm", signature(x="gpuSparseMatrix"),
          function(x, type = "F") {
              if (type != "F") {
                  stop("Only Frobenius norm 'F' is currently supported for sparse matrices")
              }
              cpp_gpuSparseMat_norm_frobenius(x@address)
          })
