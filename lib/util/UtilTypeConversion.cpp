#include "llvm/ADT/Sequence.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {

class PackOpLowering : public ConversionPattern {
   public:
   explicit PackOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::PackOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto constop = mlir::dyn_cast_or_null<mlir::util::PackOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, typeConverter->convertType(constop.tuple().getType()), ValueRange(operands));
      return success();
   }
};
class UndefTupleOpLowering : public ConversionPattern {
   public:
   explicit UndefTupleOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::UndefTupleOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto undefTupleOp = mlir::dyn_cast_or_null<mlir::util::UndefTupleOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::UndefTupleOp>(op, typeConverter->convertType(undefTupleOp.tuple().getType()));
      return success();
   }
};
class SetTupleOpLowering : public ConversionPattern {
   public:
   explicit SetTupleOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::SetTupleOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::SetTupleOpAdaptor setTupleOpAdaptor(operands);
      auto setTupleOp = mlir::dyn_cast_or_null<mlir::util::SetTupleOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::SetTupleOp>(op, typeConverter->convertType(setTupleOp.tuple_out().getType()), setTupleOpAdaptor.tuple(), setTupleOpAdaptor.val(), setTupleOp.offset());
      return success();
   }
};
class GetTupleOpLowering : public ConversionPattern {
   public:
   explicit GetTupleOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::GetTupleOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::GetTupleOpAdaptor getTupleOpAdaptor(operands);
      auto getTupleOp = mlir::dyn_cast_or_null<mlir::util::GetTupleOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::GetTupleOp>(op, typeConverter->convertType(getTupleOp.val().getType()), getTupleOpAdaptor.tuple(), getTupleOp.offset());

      return success();
   }
};
class UnPackOpLowering : public ConversionPattern {
   public:
   explicit UnPackOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::UnPackOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::UnPackOpAdaptor unPackOpAdaptor(operands);

      auto unPackOp = mlir::dyn_cast_or_null<mlir::util::UnPackOp>(op);
      llvm::SmallVector<Type> valTypes;
      for (auto v : unPackOp.vals()) {
         Type converted = typeConverter->convertType(v.getType());
         converted = converted ? converted : v.getType();
         valTypes.push_back(converted);
      }

      rewriter.replaceOpWithNewOp<mlir::util::UnPackOp>(op, valTypes, operands);

      return success();
   }
};
} // end anonymous namespace

class ToGenericMemrefOpLowering : public ConversionPattern {
   public:
   explicit ToGenericMemrefOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::ToGenericMemrefOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::ToGenericMemrefOpAdaptor adaptor(operands);
      auto castedOp = mlir::dyn_cast_or_null<mlir::util::ToGenericMemrefOp>(op);
      auto genericMemrefType = castedOp.generic_memref().getType().cast<mlir::util::GenericMemrefType>();
      auto loweredGenericMemrefType = mlir::util::GenericMemrefType::get(getContext(), typeConverter->convertType(genericMemrefType.getElementType()), genericMemrefType.getSize());

      rewriter.replaceOpWithNewOp<mlir::util::ToGenericMemrefOp>(op, loweredGenericMemrefType, adaptor.memref());

      return success();
   }
};
template <class UtilOp>
class AllocOpLowering : public ConversionPattern {
   public:
   explicit AllocOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, UtilOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      typename UtilOp::Adaptor adaptor(operands);

      auto castedOp = mlir::dyn_cast_or_null<UtilOp>(op);
      auto genericMemrefType = castedOp.generic_memref().getType().template cast<mlir::util::GenericMemrefType>();
      auto loweredGenericMemrefType = mlir::util::GenericMemrefType::get(getContext(), typeConverter->convertType(genericMemrefType.getElementType()), genericMemrefType.getSize());
      rewriter.replaceOpWithNewOp<UtilOp>(op, loweredGenericMemrefType, adaptor.size());
      return success();
   }
};
class StoreOpLowering : public ConversionPattern {
   public:
   explicit StoreOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::StoreOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::StoreOpAdaptor adaptor(operands);
      rewriter.replaceOpWithNewOp<mlir::util::StoreOp>(op, adaptor.val(), adaptor.generic_memref(), adaptor.idx());

      return success();
   }
};
class DeAllocOpLowering : public ConversionPattern {
   public:
   explicit DeAllocOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::DeAllocOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::DeAllocOpAdaptor adaptor(operands);
      rewriter.replaceOpWithNewOp<mlir::util::DeAllocOp>(op, adaptor.generic_memref());

      return success();
   }
};
class SizeOfLowering : public ConversionPattern {
   public:
   explicit SizeOfLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::SizeOfOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto castedOp = mlir::dyn_cast_or_null<mlir::util::SizeOfOp>(op);
      auto converted=typeConverter->convertType(castedOp.type());
      rewriter.replaceOpWithNewOp<mlir::util::SizeOfOp>(op, rewriter.getIndexType(), TypeAttr::get(converted));

      return success();
   }
};
class DimOpLowering : public ConversionPattern {
   public:
   explicit DimOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::DimOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::DimOpAdaptor adaptor(operands);


      rewriter.replaceOpWithNewOp<mlir::util::DimOp>(op, rewriter.getIndexType(), adaptor.generic_memref());

      return success();
   }
};
class LoadOpLowering : public ConversionPattern {
   public:
   explicit LoadOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::LoadOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::LoadOpAdaptor adaptor(operands);
      auto castedOp = mlir::dyn_cast_or_null<mlir::util::LoadOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::LoadOp>(op, typeConverter->convertType(castedOp.val().getType()), adaptor.generic_memref(), adaptor.idx());

      return success();
   }
};
class CastOpLowering : public ConversionPattern {
   public:
   explicit CastOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::GenericMemrefCastOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::GenericMemrefCastOpAdaptor adaptor(operands);
      auto castedOp = mlir::dyn_cast_or_null<mlir::util::GenericMemrefCastOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::GenericMemrefCastOp>(op, typeConverter->convertType(castedOp.res().getType()), adaptor.val());
      return success();
   }
};
//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void mlir::util::populateUtilTypeConversionPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<PackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UnPackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ToGenericMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<StoreOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocOpLowering<util::AllocOp>>(typeConverter, patterns.getContext());
   patterns.add<AllocOpLowering<util::AllocaOp>>(typeConverter, patterns.getContext());
   patterns.add<DeAllocOpLowering>(typeConverter, patterns.getContext());
   patterns.add<DimOpLowering>(typeConverter, patterns.getContext());
   patterns.add<CastOpLowering>(typeConverter, patterns.getContext());

   patterns.add<SizeOfLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadOpLowering>(typeConverter, patterns.getContext());

   typeConverter.addConversion([&](mlir::util::GenericMemrefType genericMemrefType) {
      return mlir::util::GenericMemrefType::get(genericMemrefType.getContext(), typeConverter.convertType(genericMemrefType.getElementType()), genericMemrefType.getSize());
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, mlir::util::GenericMemrefType, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, mlir::util::GenericMemrefType, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
}
