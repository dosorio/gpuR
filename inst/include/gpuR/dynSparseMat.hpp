#ifndef GPURAM_DYNA_SPARSE_MAT_HPP_
#define GPURAM_DYNA_SPARSE_MAT_HPP_

#include <memory>
#include "gpuR/windows_check.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

#include "viennacl/ocl/backend.hpp"
#include "viennacl/compressed_matrix.hpp"

namespace gpuR {

/** @brief Dynamic sparse matrix wrapper for ViennaCL compressed_matrix
 * 
 * This class wraps ViennaCL's compressed_matrix (CSR format) to provide
 * efficient sparse matrix operations on GPUs. The template parameter T
 * specifies the numeric type (float, double, int, etc.).
 */
template <typename T>
class dynSparseMat {
private:
    std::shared_ptr<viennacl::compressed_matrix<T>> mat_;
    viennacl::context ctx_;
    size_t rows_;
    size_t cols_;
    size_t nnz_;  // number of non-zeros
    
public:
    typedef T value_type;
    typedef viennacl::backend::mem_handle handle_type;
    
    // Constructor - empty matrix
    dynSparseMat() : rows_(0), cols_(0), nnz_(0) {
        mat_ = std::make_shared<viennacl::compressed_matrix<T>>();
    }
    
    // Constructor with context
    dynSparseMat(viennacl::context const & ctx)
        : ctx_(ctx), rows_(0), cols_(0), nnz_(0) {
        mat_ = std::make_shared<viennacl::compressed_matrix<T>>(ctx);
    }
    
    // Constructor with dimensions and nnz
    dynSparseMat(size_t rows, size_t cols, size_t nnz, 
                 viennacl::context const & ctx)
        : ctx_(ctx), rows_(rows), cols_(cols), nnz_(nnz) {
        mat_ = std::make_shared<viennacl::compressed_matrix<T>>(rows, cols, nnz, ctx);
    }
    
    // Get raw matrix pointer
    viennacl::compressed_matrix<T>* getMatrix() {
        return mat_.get();
    }
    
    // Get shared pointer
    std::shared_ptr<viennacl::compressed_matrix<T>> getMatrixPtr() {
        return mat_;
    }
    
    // Matrix dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t nnz() const { return nnz_; }
    
    // Update matrix dimensions after initialization
    void setDimensions(size_t rows, size_t cols, size_t nnz) {
        rows_ = rows;
        cols_ = cols;
        nnz_ = nnz;
    }
    
    // Get context
    viennacl::context getContext() const {
        return ctx_;
    }
    
    // Get context id
    int getContextId() const {
        return 0;  // Default context ID
    }
};

}  // namespace gpuR

#endif // GPURAM_DYNA_SPARSE_MAT_HPP_
