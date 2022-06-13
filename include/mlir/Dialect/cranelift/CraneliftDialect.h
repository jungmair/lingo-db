#ifndef MLIR_DIALECT_CRANELIFT_CRANELIFTDIALECT_H
#define MLIR_DIALECT_CRANELIFT_CRANELIFTDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/cranelift/CraneliftOpsDialect.h.inc"
namespace mlir::cranelift {
std::unique_ptr<mlir::Pass> createToCLIRPass();
}
#endif // MLIR_DIALECT_CRANELIFT_CRANELIFTDIALECT_H
