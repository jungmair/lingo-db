#ifndef LINGODB_CATALOG_MLIRTYPES_H
#define LINGODB_CATALOG_MLIRTYPES_H
#include "mlir/IR/Types.h"

#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

namespace mlir {
class ModuleOp;
class OpBuilder;
}
namespace lingodb::catalog {
class IntTypeInfo;
class DecimalTypeInfo;
class StringTypeInfo;
class CharTypeInfo;
class TimestampTypeInfo;
class DateTypeInfo;
class IntervalTypeInfo;
class MLIRTypeCreator {
   public:
   virtual mlir::Type createType(mlir::MLIRContext* context) = 0;
   virtual ~MLIRTypeCreator() = default;
};
std::shared_ptr<MLIRTypeCreator> createBoolTypeCreator();
std::shared_ptr<MLIRTypeCreator> createIntTypeCreator(std::shared_ptr<catalog::IntTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createFloatTypeCreator();
std::shared_ptr<MLIRTypeCreator> createDoubleTypeCreator();
std::shared_ptr<MLIRTypeCreator> createDecimalTypeCreator(std::shared_ptr<catalog::DecimalTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createDateTypeCreator(std::shared_ptr<catalog::DateTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createTimestampTypeCreator(std::shared_ptr<catalog::TimestampTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createIntervalTypeCreator(std::shared_ptr<catalog::IntervalTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createCharTypeCreator(std::shared_ptr<catalog::CharTypeInfo> info);
std::shared_ptr<MLIRTypeCreator> createStringTypeCreator(std::shared_ptr<catalog::StringTypeInfo> info);

class MLIRFunctionImplementer {
   public:
   virtual mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) = 0;
   virtual ~MLIRFunctionImplementer() = default;
};
} // namespace lingodb::catalog

#endif //LINGODB_CATALOG_MLIRTYPES_H
