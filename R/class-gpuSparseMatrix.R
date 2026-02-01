# Sparse matrix classes for gpuR using ViennaCL

#' @title gpuSparseMatrix Class
#' @description This is the 'mother' class for all
#' gpuSparseMatrix objects. It wraps sparse matrices
#' in compressed row storage (CSR) format for efficient
#' GPU computation. All other gpuSparseMatrix classes
#' inherit from this class.
#' 
#' There are multiple child classes that correspond
#' to the particular data type contained. These include
#' \code{igpuSparseMatrix}, \code{fgpuSparseMatrix}, and 
#' \code{dgpuSparseMatrix} corresponding to integer, float, and
#' double data types respectively.
#'
#' @section Slots:
#'  Common to all gpuSparseMatrix objects in the package
#'  \describe{
#'      \item{\code{address}:}{Pointer to sparse matrix data}
#'      \item{\code{.context_index}:}{Integer index of OpenCL contexts}
#'      \item{\code{.platform_index}:}{Integer index of OpenCL platforms}
#'      \item{\code{.platform}:}{Name of OpenCL platform}
#'      \item{\code{.device_index}:}{Integer index of active device}
#'      \item{\code{.device}:}{Name of active device}
#'  }
#'
#' @note Sparse matrices are stored in Compressed Sparse Row (CSR) format
#' for optimal performance on GPUs. The matrix cannot be directly modified
#' after creation; conversion to dense, modification, and re-sparsification
#' is recommended.
#'
#' @return An object of class 'gpuSparseMatrix' with the specified slots.
#' @name gpuSparseMatrix-class
#' @rdname gpuSparseMatrix-class
#' @author Charles Determan Jr.
#' @seealso \code{\link{igpuSparseMatrix-class}}, 
#' \code{\link{fgpuSparseMatrix-class}},
#' \code{\link{dgpuSparseMatrix-class}}
#' @export
setClass('gpuSparseMatrix', 
         slots = c(address="externalptr",
                   .context_index = "integer",
                   .platform_index = "integer",
                   .platform = "character",
                   .device_index = "integer",
                   .device = "character"))


#' @title igpuSparseMatrix Class
#' @description An integer type sparse matrix in the S4 \code{gpuSparseMatrix}
#' representation using Compressed Sparse Row (CSR) format.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to an integer typed sparse matrix}
#'  }
#' @name igpuSparseMatrix-class
#' @rdname igpuSparseMatrix-class
#' @author Charles Determan Jr.
#' @return If the gpuSparseMatrix object is of type 'integer', returns TRUE, if not, returns an error message.
#' @seealso \code{\link{gpuSparseMatrix-class}}, 
#' \code{\link{fgpuSparseMatrix-class}},
#' \code{\link{dgpuSparseMatrix-class}}
#' @export
setClass("igpuSparseMatrix",
         contains = "gpuSparseMatrix",
         validity = function(object) {
             if( typeof(object) != "integer"){
                 return("igpuSparseMatrix must be of type 'integer'")
             }
             TRUE
         })


#' @title fgpuSparseMatrix Class
#' @description A float type sparse matrix in the S4 \code{gpuSparseMatrix}
#' representation using Compressed Sparse Row (CSR) format.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a float typed sparse matrix.}
#'  }
#' @name fgpuSparseMatrix-class
#' @rdname fgpuSparseMatrix-class
#' @author Charles Determan Jr.
#' @return If the gpuSparseMatrix object is of type 'float', returns TRUE, if not, returns an error message.
#' @seealso \code{\link{gpuSparseMatrix-class}}, 
#' \code{\link{igpuSparseMatrix-class}},
#' \code{\link{dgpuSparseMatrix-class}}
#' @export
setClass("fgpuSparseMatrix",
         contains = "gpuSparseMatrix",
         validity = function(object) {
             if( typeof(object) != "float"){
                 return("fgpuSparseMatrix must be of type 'float'")
             }
             TRUE
         })


#' @title dgpuSparseMatrix Class
#' @description A double type sparse matrix in the S4 \code{gpuSparseMatrix}
#' representation using Compressed Sparse Row (CSR) format.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a double type sparse matrix}
#'  }
#' @name dgpuSparseMatrix-class
#' @rdname dgpuSparseMatrix-class
#' @author Charles Determan Jr.
#' @return If the gpuSparseMatrix object is of type 'double', returns TRUE, if not, returns an error message.
#' @seealso \code{\link{gpuSparseMatrix-class}}, 
#' \code{\link{igpuSparseMatrix-class}},
#' \code{\link{fgpuSparseMatrix-class}}
#' @export
setClass("dgpuSparseMatrix",
         contains = "gpuSparseMatrix",
         validity = function(object) {
             if( typeof(object) != "double"){
                 return("dgpuSparseMatrix must be of type 'double'")
             }
             TRUE
         })


#' @title cgpuSparseMatrix Class
#' @description A complex float type sparse matrix in the S4 \code{gpuSparseMatrix}
#' representation using Compressed Sparse Row (CSR) format.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a complex float sparse matrix.}
#'  }
#' @name cgpuSparseMatrix-class
#' @rdname cgpuSparseMatrix-class
#' @author Charles Determan Jr.
#' @return If the gpuSparseMatrix object is of type 'complex float', returns TRUE, if not, returns an error message.
#' @seealso \code{\link{gpuSparseMatrix-class}}, 
#' \code{\link{igpuSparseMatrix-class}},
#' \code{\link{dgpuSparseMatrix-class}}
#' @export
setClass("cgpuSparseMatrix",
         contains = "gpuSparseMatrix",
         validity = function(object) {
             if( typeof(object) != "fcomplex"){
                 return("cgpuSparseMatrix must be of type 'fcomplex'")
             }
             TRUE
         })


#' @title zgpuSparseMatrix Class
#' @description A complex double type sparse matrix in the S4 \code{gpuSparseMatrix}
#' representation using Compressed Sparse Row (CSR) format.
#' @section Slots:
#'  \describe{
#'      \item{\code{address}:}{Pointer to a complex double sparse matrix.}
#'  }
#' @name zgpuSparseMatrix-class
#' @rdname zgpuSparseMatrix-class
#' @author Charles Determan Jr.
#' @return If the gpuSparseMatrix object is of type 'complex double', returns TRUE, if not, returns an error message.
#' @seealso \code{\link{gpuSparseMatrix-class}}, 
#' \code{\link{igpuSparseMatrix-class}},
#' \code{\link{dgpuSparseMatrix-class}}
#' @export
setClass("zgpuSparseMatrix",
         contains = "gpuSparseMatrix",
         validity = function(object) {
             if( typeof(object) != "dcomplex"){
                 return("zgpuSparseMatrix must be of type 'dcomplex'")
             }
             TRUE
         })
