#' @keywords internal
#' @noRd
#' @importFrom utils strOptions packageVersion tail

.onLoad <- function(libname, pkgname) {
    options(gpuR.print.warning=TRUE)
    options(gpuR.default.type = "float")
    # options(gpuR.default.device.type = "gpu")
    
    # Register S4 method qr.Q - wrapped to avoid roxygen2 issues
    tryCatch({
        setMethod("qr.Q", signature(qr = "gpuQR"), 
                  function(qr, complete = FALSE) {
                      type <- typeof(qr$qr)
                      isVCL <- inherits(qr$qr, "vclMatrix")
                      if(isVCL){
                          Q <- vclMatrix(nrow = nrow(qr$qr), ncol = ncol(qr$qr), type = type)
                          R <- vclMatrix(nrow = nrow(qr$qr), ncol = nrow(qr$qr), type = type)    
                      }else{
                          Q <- gpuMatrix(nrow = nrow(qr$qr), ncol = ncol(qr$qr), type = type)
                          R <- vclMatrix(nrow = nrow(qr$qr), ncol = nrow(qr$qr), type = type)
                      }
                      switch(type,
                             "float" = cpp_recover_qr(qr$qr@address, isVCL,
                                                      Q@address, inherits(Q, "vclMatrix"),
                                                      R@address, inherits(R, "vclMatrix"),
                                                      qr$betas, 6L, qr$qr@.context_index - 1),
                             "double" = cpp_recover_qr(qr$qr@address, isVCL,
                                                       Q@address, inherits(Q, "vclMatrix"),
                                                       R@address, inherits(R, "vclMatrix"),
                                                       qr$betas, 8L, qr$qr@.context_index - 1),
                             stop("type not currently supported"))
                      return(Q)
                  },
                  where = asNamespace("gpuR"))
    }, error = function(e) {
        # Silently ignore errors during roxygen2 processing
    })
}

.onAttach <- function(libname, pkgname) {
    # Initialize all possible contexts
    if (!identical(Sys.getenv("APPVEYOR"), "True")) {
        # initialize contexts
      packageStartupMessage(paste0("gpuR ", packageVersion('gpuR')))
      packageStartupMessage(initContexts())
    }
}

.onUnload <- function(libpath) {
    options(gpuR.print.warning=NULL)
    options(gpuR.default.type = NULL)
    # options(gpuR.default.device.type = NULL)
}
