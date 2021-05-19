#ifndef MLIR_CONVERSION_DBTOARROWSTD_COLLECTIONITERATION_H
#define MLIR_CONVERSION_DBTOARROWSTD_COLLECTIONITERATION_H
#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::db {
class CollectionIterationImpl {
   public:
   virtual std::vector<Value> implementLoop(mlir::ValueRange iterArgs, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, mlir::ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) = 0;
   virtual ~CollectionIterationImpl() {
   }
   static std::unique_ptr<mlir::db::CollectionIterationImpl> getImpl(mlir::Type collectionType, mlir::Value collection, mlir::db::codegen::FunctionRegistry& functionRegistry);
};

} // namespace mlir::db
#endif // MLIR_CONVERSION_DBTOARROWSTD_COLLECTIONITERATION_H
