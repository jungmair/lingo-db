#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/cranelift/CraneliftDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"

#include "mlir/Dialect/cranelift/CraneliftOps.h"
#include "mlir/Dialect/util/UtilOps.h"

#include "mlir/Conversion/CraneliftConversions/CraneliftConversions.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

using namespace mlir;

namespace {

class FuncLowering : public OpConversionPattern<mlir::func::FuncOp> {
   public:
   using OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::func::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type, 4> inputTypes, outputTypes;
      assert(typeConverter->convertTypes(op.getFunctionType().getInputs(), inputTypes).succeeded());
      assert(typeConverter->convertTypes(op.getFunctionType().getResults(), outputTypes).succeeded());
      auto funcOp = rewriter.create<mlir::cranelift::FuncOp>(op->getLoc(), op.getSymName(), rewriter.getFunctionType(inputTypes, outputTypes));
      rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(),
                                  funcOp.end());
      TypeConverter::SignatureConversion result(funcOp.getNumArguments());
      auto llvmType = getTypeConverter()->convertSignatureArgs(op.getFunctionType().getInputs(), result);
      if (llvmType.failed())
         return failure();
      if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter, &result)))
         return failure();
      rewriter.eraseOp(op);
      return success();
   }
};
class CallLowering : public OpConversionPattern<mlir::func::CallOp> {
   public:
   using OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type, 4> resTypes;
      typeConverter->convertTypes(op.getResultTypes(), resTypes);
      auto callOp = rewriter.replaceOpWithNewOp<mlir::cranelift::CallOp>(op, resTypes, op.getCallee(), adaptor.operands());
      return success();
   }
};
class ReturnLowering : public OpConversionPattern<mlir::func::ReturnOp> {
   public:
   using OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::func::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::ReturnOp>(op, adaptor.operands());
      return success();
   }
};
class ExtSILowering : public OpConversionPattern<mlir::arith::ExtSIOp> {
   public:
   using OpConversionPattern<mlir::arith::ExtSIOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::ExtSIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto funcOp = rewriter.replaceOpWithNewOp<mlir::cranelift::SExtendOp>(op, op.getType(), adaptor.getIn());
      return success();
   }
};
class ExtUILowering : public OpConversionPattern<mlir::arith::ExtUIOp> {
   public:
   using OpConversionPattern<mlir::arith::ExtUIOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::ExtUIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (adaptor.getIn().getType().isInteger(1)) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::BIntOp>(op, op.getType(), adaptor.getIn());
      } else {
         rewriter.replaceOpWithNewOp<mlir::cranelift::UExtendOp>(op, op.getType(), adaptor.getIn());
      }
      return success();
   }
};
class SelectLowering : public OpConversionPattern<mlir::arith::SelectOp> {
   public:
   using OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (op.getType().isInteger(128)) {
         auto split1 = rewriter.create<mlir::cranelift::ISplitOp>(op->getLoc(), mlir::TypeRange{rewriter.getI64Type(), rewriter.getI64Type()}, adaptor.getTrueValue());
         auto split2 = rewriter.create<mlir::cranelift::ISplitOp>(op->getLoc(), mlir::TypeRange{rewriter.getI64Type(), rewriter.getI64Type()}, adaptor.getFalseValue());
         mlir::Value select1 = rewriter.create<mlir::cranelift::SelectOp>(op->getLoc(), rewriter.getI64Type(), adaptor.getCondition(), split1.getResult(0), split2.getResult(0));
         mlir::Value select2 = rewriter.create<mlir::cranelift::SelectOp>(op->getLoc(), rewriter.getI64Type(), adaptor.getCondition(), split1.getResult(1), split2.getResult(1));
         rewriter.replaceOpWithNewOp<mlir::cranelift::IConcatOp>(op, op.getType(), select1, select2);
      } else {
         rewriter.replaceOpWithNewOp<mlir::cranelift::SelectOp>(op, adaptor.getFalseValue().getType(), adaptor.getCondition(), adaptor.getTrueValue(), adaptor.getFalseValue());
      }
      return success();
   }
};
class TruncILowering : public OpConversionPattern<mlir::arith::TruncIOp> {
   public:
   using OpConversionPattern<mlir::arith::TruncIOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::TruncIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto funcOp = rewriter.replaceOpWithNewOp<mlir::cranelift::IReduceOp>(op, op.getType(), adaptor.getIn());
      return success();
   }
};
static void ensureDiv3(mlir::ModuleOp module, OpBuilder& builder) {
   mlir::cranelift::FuncOp funcOp = module.lookupSymbol<mlir::cranelift::FuncOp>("__divti3");
   if (!funcOp) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      funcOp = builder.create<mlir::cranelift::FuncOp>(module.getLoc(), "__divti3", builder.getFunctionType({builder.getIntegerType(128),builder.getIntegerType(128)}, {builder.getIntegerType(128)}));
   }
}
class DivSILowering : public OpConversionPattern<mlir::arith::DivSIOp> {
   public:
   using OpConversionPattern<mlir::arith::DivSIOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::DivSIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetType=typeConverter->convertType(op.getType());
      if(targetType.isInteger(128)){
         ensureDiv3(op->getParentOfType<mlir::ModuleOp>(),rewriter);
         rewriter.replaceOpWithNewOp<mlir::cranelift::CallOp>(op,targetType,"__divti3",mlir::ValueRange{adaptor.getLhs(),adaptor.getRhs()});
      }else{
         rewriter.replaceOpWithNewOp<mlir::cranelift::SDivOp>(op, targetType, adaptor.getLhs(),adaptor.getRhs());
      }
      return success();
   }
};

template <class From, class To>
class SimpleArithmeticLowering : public OpConversionPattern<From> {
   public:
   using OpConversionPattern<From>::OpConversionPattern;
   LogicalResult matchAndRewrite(From op, typename From::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<To>(op, this->typeConverter->convertType(op.getType()), adaptor.getOperands());
      return success();
   }
};

static size_t getAlignmentOf(mlir::Operation* op, mlir::Type t, const DataLayoutAnalysis& dataLayoutAnalysis) {
   DataLayout defaultLayout;
   const DataLayout* layout = &defaultLayout;
   layout = &dataLayoutAnalysis.getAbove(op);
   if (auto tupleType = t.dyn_cast<mlir::TupleType>()) {
      unsigned structSize = 0;
      unsigned structAlignment = 1;
      for (Type element : tupleType.getTypes()) {
         unsigned elementAlignment = getAlignmentOf(op, element, dataLayoutAnalysis);
         structAlignment = std::max(elementAlignment, structAlignment);
      }
      return structAlignment;
   } else {
      size_t typeAlignment = layout->getTypeABIAlignment(t);
      return typeAlignment;
   }
}
static size_t getSizeOf(mlir::Operation* op, mlir::Type t, const DataLayoutAnalysis& dataLayoutAnalysis) {
   DataLayout defaultLayout;
   const DataLayout* layout = &defaultLayout;
   layout = &dataLayoutAnalysis.getAbove(op);
   if (auto tupleType = t.dyn_cast<mlir::TupleType>()) {
      unsigned structSize = 0;
      unsigned structAlignment = 1;
      for (Type element : tupleType.getTypes()) {
         unsigned elementAlignment = getAlignmentOf(op, element, dataLayoutAnalysis);
         // Add padding to the struct size to align it to the abi alignment of the
         // element type before than adding the size of the element
         structSize = llvm::alignTo(structSize, elementAlignment);
         structSize += getSizeOf(op, element, dataLayoutAnalysis);

         // The alignment requirement of a struct is equal to the strictest alignment
         // requirement of its elements.
         structAlignment = std::max(elementAlignment, structAlignment);
      }
      // At the end, add padding to the struct to satisfy its own alignment
      // requirement. Otherwise structs inside of arrays would be misaligned.
      structSize = llvm::alignTo(structSize, structAlignment);
      return structSize;
   } else {
      size_t typeSize = layout->getTypeSize(t);
      return typeSize;
   }
}

static size_t getTupleOffset(mlir::Operation* op, mlir::TupleType tupleType, size_t offset, const DataLayoutAnalysis& dataLayoutAnalysis) {
   DataLayout defaultLayout;
   const DataLayout* layout = &defaultLayout;
   layout = &dataLayoutAnalysis.getAbove(op);
   unsigned structSize = 0;
   for (size_t i = 0; i < offset; i++) {
      mlir::Type element = tupleType.getType(i);
      unsigned elementAlignment = getAlignmentOf(op, element, dataLayoutAnalysis);
      // Add padding to the struct size to align it to the abi alignment of the
      // element type before than adding the size of the element
      structSize = llvm::alignTo(structSize, elementAlignment);
      structSize += getSizeOf(op, element, dataLayoutAnalysis);
   }
   mlir::Type element = tupleType.getType(offset);
   unsigned elementAlignment = getAlignmentOf(op, element, dataLayoutAnalysis);
   structSize = llvm::alignTo(structSize, elementAlignment);
   return structSize;
}
class SizeOfOpLowering : public ConversionPattern {
   public:
   const DataLayoutAnalysis& dataLayoutAnalysis;
   explicit SizeOfOpLowering(TypeConverter& typeConverter, MLIRContext* context, const DataLayoutAnalysis& dataLayoutAnalysis)
      : ConversionPattern(typeConverter, mlir::util::SizeOfOp::getOperationName(), 1, context), dataLayoutAnalysis(dataLayoutAnalysis) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto sizeOfOp = mlir::dyn_cast_or_null<mlir::util::SizeOfOp>(op);
      Type t = typeConverter->convertType(sizeOfOp.type());

      rewriter.replaceOpWithNewOp<mlir::cranelift::IConstOp>(op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(getSizeOf(op, t, dataLayoutAnalysis)));
      return success();
   }
};
class AllocaOpLowering : public ConversionPattern {
   const mlir::DataLayoutAnalysis& dataLayoutAnalysis;

   public:
   explicit AllocaOpLowering(TypeConverter& typeConverter, MLIRContext* context, const DataLayoutAnalysis& dataLayoutAnalysis)
      : ConversionPattern(typeConverter, mlir::util::AllocaOp::getOperationName(), 1, context), dataLayoutAnalysis(dataLayoutAnalysis) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::AllocaOp allocOp(op);
      auto loc = allocOp->getLoc();
      auto genericMemrefType = allocOp.ref().getType().cast<mlir::util::RefType>();
      int64_t staticSize = 0;
      if (allocOp.size()) {
         if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(allocOp.size().getDefiningOp())) {
            staticSize = constOp.getValue().cast<mlir::IntegerAttr>().getInt();
         }
      } else {
         staticSize = 1;
      }
      Type t = typeConverter->convertType(allocOp.ref().getType().cast<mlir::util::RefType>().getElementType());
      staticSize *= getSizeOf(allocOp.getOperation(), t, dataLayoutAnalysis);
      assert(staticSize > 0);
      rewriter.replaceOpWithNewOp<mlir::cranelift::AllocaOp>(allocOp, rewriter.getI32IntegerAttr(staticSize));

      return success();
   }
};

static void ensureMalloc(mlir::ModuleOp module, OpBuilder& builder) {
   mlir::cranelift::FuncOp funcOp = module.lookupSymbol<mlir::cranelift::FuncOp>("malloc");
   if (!funcOp) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      funcOp = builder.create<mlir::cranelift::FuncOp>(module.getLoc(), "malloc", builder.getFunctionType({builder.getI64Type()}, {builder.getI64Type()}));
   }
}
class AllocOpLowering : public ConversionPattern {
   const mlir::DataLayoutAnalysis& dataLayoutAnalysis;

   public:
   explicit AllocOpLowering(TypeConverter& typeConverter, MLIRContext* context, const DataLayoutAnalysis& dataLayoutAnalysis)
      : ConversionPattern(typeConverter, mlir::util::AllocOp::getOperationName(), 1, context), dataLayoutAnalysis(dataLayoutAnalysis) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::AllocOp allocOp(op);
      mlir::util::AllocOpAdaptor adaptor(operands);
      auto loc = allocOp->getLoc();

      auto genericMemrefType = allocOp.ref().getType().cast<mlir::util::RefType>();
      Value entries;
      if (allocOp.size()) {
         entries = adaptor.size();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }

      auto bytesPerEntry = rewriter.create<mlir::cranelift::IConstOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(getSizeOf(allocOp.getOperation(), typeConverter->convertType(genericMemrefType.getElementType()), dataLayoutAnalysis)));
      Value sizeInBytes = rewriter.create<mlir::arith::MulIOp>(loc, rewriter.getI64Type(), entries, bytesPerEntry);
      ensureMalloc(allocOp->getParentOfType<mlir::ModuleOp>(), rewriter);
      rewriter.replaceOpWithNewOp<mlir::cranelift::CallOp>(allocOp, mlir::TypeRange{rewriter.getI64Type()}, llvm::StringRef("malloc"), sizeInBytes);
      return success();
   }
};
class ConstLowering : public OpConversionPattern<mlir::arith::ConstantOp> {
   public:
   using OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto convertedType = typeConverter->convertType(op.getType());
      if (op.getType().isInteger(1)) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::BConstOp>(op, op.getValue().cast<mlir::IntegerAttr>().getInt() != 0);
         return success();
      } else if (op.getType().isInteger(128)) {
         auto intVal = op.getValue().cast<mlir::IntegerAttr>().getValue();
         if (intVal.getBitWidth() > 64) {
            int64_t low = *intVal.getLoBits(64).getRawData();
            int64_t high = *op.getValue().cast<mlir::IntegerAttr>().getValue().getHiBits(intVal.getBitWidth() - 64).getRawData();
            mlir::Value lowV = rewriter.create<mlir::cranelift::IConstOp>(op.getLoc(), rewriter.getI64Type(), low);
            mlir::Value highV = rewriter.create<mlir::cranelift::IConstOp>(op.getLoc(), rewriter.getI64Type(), high);
            rewriter.replaceOpWithNewOp<mlir::cranelift::IConcatOp>(op, convertedType, lowV, highV);
         } else {
            assert(false);
         }
         return success();
      } else if (op.getType().isIntOrIndex()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::IConstOp>(op, convertedType, op.getValue().cast<mlir::IntegerAttr>().getInt());
         return success();
      } else if (op.getType().isF32()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::F32ConstOp>(op, convertedType, op.getValue().cast<mlir::FloatAttr>());
         return success();
      } else if (op.getType().isF64()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::F64ConstOp>(op, convertedType, op.getValue().cast<mlir::FloatAttr>());
         return success();
      }
      return failure();
   }
};

class UndefLowering : public OpConversionPattern<mlir::util::UndefOp> {
   public:
   using OpConversionPattern<mlir::util::UndefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::UndefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetType = typeConverter->convertType(op.getType());
      if (targetType.isInteger(1)) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::BConstOp>(op, 0);
         return success();
      } else if (targetType.isInteger(128)) {
         mlir::Value zeroV = rewriter.create<mlir::cranelift::IConstOp>(op.getLoc(), rewriter.getI64Type(), 0);
         rewriter.replaceOpWithNewOp<mlir::cranelift::IConcatOp>(op, targetType, zeroV, zeroV);
         return success();
      } else if (targetType.isIntOrIndex()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::IConstOp>(op, targetType, 0);
         return success();
      }
      return failure();
   }
};
class ArrayElementPtrOpLowering : public ConversionPattern {
   public:
   const DataLayoutAnalysis& dataLayoutAnalysis;
   explicit ArrayElementPtrOpLowering(TypeConverter& typeConverter, MLIRContext* context, const DataLayoutAnalysis& dataLayoutAnalysis)
      : ConversionPattern(typeConverter, mlir::util::ArrayElementPtrOp::getOperationName(), 1, context), dataLayoutAnalysis(dataLayoutAnalysis) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto arrayElementPtrOp = mlir::dyn_cast_or_null<mlir::util::ArrayElementPtrOp>(op);
      mlir::util::ArrayElementPtrOpAdaptor adaptor(operands);
      mlir::Value bytesPerEntry = rewriter.create<mlir::cranelift::IConstOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(getSizeOf(op, typeConverter->convertType(arrayElementPtrOp.ref().getType().cast<mlir::util::RefType>().getElementType()), dataLayoutAnalysis)));
      mlir::Value ptr = adaptor.ref();
      mlir::Value off = adaptor.idx();
      mlir::Value byteOff = rewriter.create<mlir::cranelift::IMulOp>(op->getLoc(), off, bytesPerEntry);
      mlir::Value res = rewriter.create<mlir::cranelift::IAddOp>(op->getLoc(), ptr, byteOff);
      rewriter.replaceOp(op, res);
      return success();
   }
};

class TupleElementPtrOpLowering : public ConversionPattern {
   public:
   const DataLayoutAnalysis& dataLayoutAnalysis;
   explicit TupleElementPtrOpLowering(TypeConverter& typeConverter, MLIRContext* context, const DataLayoutAnalysis& dataLayoutAnalysis)
      : ConversionPattern(typeConverter, mlir::util::TupleElementPtrOp::getOperationName(), 1, context), dataLayoutAnalysis(dataLayoutAnalysis) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto tupleElementPtrOp = mlir::dyn_cast_or_null<mlir::util::TupleElementPtrOp>(op);
      mlir::util::TupleElementPtrOpAdaptor adaptor(operands);
      Type t = typeConverter->convertType(tupleElementPtrOp.ref().getType().cast<mlir::util::RefType>().getElementType());

      mlir::Value offset = rewriter.create<mlir::cranelift::IConstOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(getTupleOffset(op, t.cast<mlir::TupleType>(), tupleElementPtrOp.idx(), dataLayoutAnalysis)));
      rewriter.replaceOpWithNewOp<mlir::cranelift::IAddOp>(op, adaptor.ref(), offset);
      return success();
   }
};
class CreateConstVarLenLowering : public OpConversionPattern<mlir::util::CreateConstVarLen> {
   public:
   using OpConversionPattern<mlir::util::CreateConstVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::CreateConstVarLen op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      size_t len = op.str().size();

      mlir::Value p1, p2;
      mlir::Value const0 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 0));

      uint64_t first4 = 0;
      memcpy(&first4, op.str().data(), std::min(4ul, len));
      size_t c1 = (first4 << 32) | len;
      p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(c1));
      if (len <= 12) {
         uint64_t last8 = 0;
         if (len > 4) {
            memcpy(&last8, op.str().data() + 4, std::min(8ul, len - 4));
         }
         p2 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(last8));
      } else {
         static size_t globalStrConstId = 0;
         mlir::cranelift::GlobalOp globalOp;
         std::string name = "global_str_const_" + std::to_string(globalStrConstId++);
         {
            auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            globalOp = rewriter.create<mlir::cranelift::GlobalOp>(op->getLoc(), name, op.strAttr().str());
         }
         p2 = rewriter.create<mlir::cranelift::AddressOfOp>(op->getLoc(), name);
      }
      rewriter.replaceOpWithNewOp<mlir::cranelift::IConcatOp>(op, rewriter.getIntegerType(128), p1, p2);
      return success();
   }
};

class CreateVarLenLowering : public OpConversionPattern<mlir::util::CreateVarLen> {
   public:
   using OpConversionPattern<mlir::util::CreateVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::CreateVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Type i128Ty = rewriter.getIntegerType(128);
      Value lazymask = rewriter.create<mlir::cranelift::IConstOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x80000000));
      Value len64 = rewriter.create<mlir::cranelift::UExtendOp>(op->getLoc(), rewriter.getI64Type(), adaptor.len());
      Value lazylen = rewriter.create<mlir::cranelift::BOrOp>(op->getLoc(), lazymask, len64);
      rewriter.replaceOpWithNewOp<mlir::cranelift::IConcatOp>(op, rewriter.getIntegerType(128), lazylen, adaptor.ref());
      return success();
   }
};

class VarLenGetLenLowering : public OpConversionPattern<mlir::util::VarLenGetLen> {
   public:
   using OpConversionPattern<mlir::util::VarLenGetLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::VarLenGetLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value len = rewriter.create<cranelift::IReduceOp>(op->getLoc(), rewriter.getI64Type(), adaptor.varlen());
      Value mask = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x7FFFFFFF));
      Value castedLen = rewriter.create<cranelift::BAndOp>(op->getLoc(), len, mask);
      rewriter.replaceOp(op, castedLen);
      return success();
   }
};

class StoreOpLowering : public OpConversionPattern<mlir::util::StoreOp> {
   public:
   using OpConversionPattern<mlir::util::StoreOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.ref();
      if (adaptor.idx()) {
         elementPtr = rewriter.create<util::ArrayElementPtrOp>(op->getLoc(), op.ref().getType(), op.ref(), op.idx());
      }
      rewriter.replaceOpWithNewOp<cranelift::StoreOp>(op, adaptor.val(), rewriter.getRemappedValue(elementPtr));
      return success();
   }
};
class LoadOpLowering : public OpConversionPattern<mlir::util::LoadOp> {
   public:
   using OpConversionPattern<mlir::util::LoadOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.ref();
      if (adaptor.idx()) {
         elementPtr = rewriter.create<util::ArrayElementPtrOp>(op->getLoc(), op.ref().getType(), op.ref(), op.idx());
      }
      rewriter.replaceOpWithNewOp<cranelift::LoadOp>(op, typeConverter->convertType(op.getType()), rewriter.getRemappedValue(elementPtr));
      return success();
   }
};
class IndexCastOpLowering : public OpConversionPattern<mlir::arith::IndexCastOp> {
   public:
   using OpConversionPattern<mlir::arith::IndexCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::IndexCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetType = typeConverter->convertType(op.getType()).cast<mlir::IntegerType>();
      auto sourceType = adaptor.getIn().getType().cast<mlir::IntegerType>();
      if (targetType == sourceType) {
         rewriter.replaceOp(op, adaptor.getIn());
      } else if (targetType.getWidth() < sourceType.getWidth()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::IReduceOp>(op, targetType, adaptor.getIn());
      } else if (targetType.getWidth() > sourceType.getWidth()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::UExtendOp>(op, targetType, adaptor.getIn());
      }
      return success();
   }
};
class GenericMemrefCastLowering : public OpConversionPattern<mlir::util::GenericMemrefCastOp> {
   public:
   using OpConversionPattern<mlir::util::GenericMemrefCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::GenericMemrefCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, adaptor.val());
      return success();
   }
};
class ToMemrefLowering : public OpConversionPattern<mlir::util::ToMemrefOp> {
   public:
   using OpConversionPattern<mlir::util::ToMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::ToMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, adaptor.ref());
      return success();
   }
};
class FuncConstLowering : public OpConversionPattern<mlir::func::ConstantOp> {
   public:
   using OpConversionPattern<mlir::func::ConstantOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::func::ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::FuncAddrOp>(op, op.getValue());
      return success();
   }
};
class CondBranchLowering : public OpConversionPattern<mlir::cf::CondBranchOp> {
   public:
   using OpConversionPattern<mlir::cf::CondBranchOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::CondBranchOp>(op, adaptor.getCondition(), adaptor.getTrueDestOperands(), adaptor.getFalseDestOperands(), op.getTrueDest(), op.getFalseDest());
      return success();
   }
};
class BranchLowering : public OpConversionPattern<mlir::cf::BranchOp> {
   public:
   using OpConversionPattern<mlir::cf::BranchOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::cf::BranchOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::BranchOp>(op, adaptor.getDestOperands(), op.getDest());
      return success();
   }
};
class ExtFOpLowering : public OpConversionPattern<mlir::arith::ExtFOp> {
   public:
   using OpConversionPattern<mlir::arith::ExtFOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::ExtFOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::FPromoteOp>(op, op.getType(), adaptor.getIn());
      return success();
   }
};
class CmpIOpLowering : public OpConversionPattern<mlir::arith::CmpIOp> {
   public:
   using OpConversionPattern<mlir::arith::CmpIOp>::OpConversionPattern;
   mlir::cranelift::ICmpPredicate translatePredicate(mlir::arith::CmpIPredicate pred) const {
      switch (pred) {
         case mlir::arith::CmpIPredicate::eq: return mlir::cranelift::ICmpPredicate::eq;
         case mlir::arith::CmpIPredicate::ne: return mlir::cranelift::ICmpPredicate::ne;
         case mlir::arith::CmpIPredicate::slt: return mlir::cranelift::ICmpPredicate::slt;
         case mlir::arith::CmpIPredicate::sle: return mlir::cranelift::ICmpPredicate::sle;
         case mlir::arith::CmpIPredicate::sgt: return mlir::cranelift::ICmpPredicate::sgt;
         case mlir::arith::CmpIPredicate::sge: return mlir::cranelift::ICmpPredicate::sge;
         case mlir::arith::CmpIPredicate::ult: return mlir::cranelift::ICmpPredicate::ult;
         case mlir::arith::CmpIPredicate::ule: return mlir::cranelift::ICmpPredicate::ule;
         case mlir::arith::CmpIPredicate::ugt: return mlir::cranelift::ICmpPredicate::ugt;
         case mlir::arith::CmpIPredicate::uge: return mlir::cranelift::ICmpPredicate::uge;
      }
      assert(false && "unsupported predicate");
   }
   LogicalResult matchAndRewrite(mlir::arith::CmpIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (adaptor.getLhs().getType().isInteger(1)) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::ICmpOp>(op, translatePredicate(op.getPredicate()), rewriter.create<mlir::cranelift::BIntOp>(op->getLoc(), rewriter.getI8Type(), adaptor.getLhs()), rewriter.create<mlir::cranelift::BIntOp>(op->getLoc(), rewriter.getI8Type(), adaptor.getRhs()));
      } else {
         rewriter.replaceOpWithNewOp<mlir::cranelift::ICmpOp>(op, translatePredicate(op.getPredicate()), adaptor.getLhs(), adaptor.getRhs());
      }
      return success();
   }
};

class Hash64Lowering : public OpConversionPattern<mlir::util::Hash64> {
   public:
   using OpConversionPattern<mlir::util::Hash64>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::Hash64 op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(11400714819323198549ull));
      Value m1 = rewriter.create<cranelift::IMulOp>(op->getLoc(), p1, adaptor.val());
      Value m2 = rewriter.create<cranelift::UMulHiOp>(op->getLoc(), p1, adaptor.val());
      Value result = rewriter.create<cranelift::BXOrOp>(op->getLoc(), m1, m2);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashCombineLowering : public OpConversionPattern<mlir::util::HashCombine> {
   public:
   using OpConversionPattern<mlir::util::HashCombine>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::HashCombine op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value c32 = rewriter.create<mlir::cranelift::IConstOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(32));
      Value left = rewriter.create<mlir::cranelift::IShlOp>(op->getLoc(), adaptor.h1(), c32);
      Value right = rewriter.create<mlir::cranelift::UShrOp>(op->getLoc(), adaptor.h1(), c32);
      Value combined = rewriter.create<mlir::cranelift::BOrOp>(op->getLoc(), left, right);
      Value result = rewriter.create<cranelift::BXOrOp>(op->getLoc(), adaptor.h2(), combined);
      rewriter.replaceOp(op, result);
      return success();
   }
};

static void ensureHashVarlenFn(mlir::ModuleOp module, OpBuilder& builder) {
   mlir::cranelift::FuncOp funcOp = module.lookupSymbol<mlir::cranelift::FuncOp>("hashVarLenData");
   if (!funcOp) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      funcOp = builder.create<mlir::cranelift::FuncOp>(module.getLoc(), "hashVarLenData", builder.getFunctionType({builder.getIntegerType(128)}, {builder.getI64Type()}));
   }
}
class HashVarLenLowering : public OpConversionPattern<mlir::util::HashVarLen> {
   public:
   using OpConversionPattern<mlir::util::HashVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::HashVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      ensureHashVarlenFn(op->getParentOfType<ModuleOp>(), rewriter);
      rewriter.replaceOpWithNewOp<mlir::cranelift::CallOp>(op, mlir::TypeRange{rewriter.getI64Type()}, llvm::StringRef("hashVarLenData"), adaptor.val());
      return success();
   }
};
class FilterTaggedPtrLowering : public OpConversionPattern<mlir::util::FilterTaggedPtr> {
   public:
   using OpConversionPattern<mlir::util::FilterTaggedPtr>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::FilterTaggedPtr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto tagMask = rewriter.create<mlir::cranelift::IConstOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xffff000000000000ull));
      auto ptrMask = rewriter.create<mlir::cranelift::IConstOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x0000ffffffffffffull));
      Value maskedHash = rewriter.create<cranelift::BAndOp>(loc, adaptor.hash(), tagMask);
      Value maskedPtr = rewriter.create<cranelift::BAndOp>(loc, adaptor.ref(), ptrMask);
      Value ored = rewriter.create<cranelift::BOrOp>(loc, adaptor.ref(), maskedHash);
      Value contained = rewriter.create<cranelift::ICmpOp>(loc, cranelift::ICmpPredicate::eq, ored, adaptor.ref());
      Value nullPtr = rewriter.create<mlir::cranelift::IConstOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value filtered = rewriter.create<mlir::cranelift::SelectOp>(loc, rewriter.getI64Type(), contained, maskedPtr, nullPtr);
      rewriter.replaceOp(op, filtered);
      return success();
   }
};
class IsRefValidOpLowering : public OpConversionPattern<mlir::util::IsRefValidOp> {
   public:
   using OpConversionPattern<mlir::util::IsRefValidOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::IsRefValidOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::ICmpOp>(op, mlir::cranelift::ICmpPredicate::ne, adaptor.ref(), rewriter.create<mlir::cranelift::IConstOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0)));
      return success();
   }
};
class InvalidRefOpLowering : public OpConversionPattern<mlir::util::InvalidRefOp> {
   public:
   using OpConversionPattern<mlir::util::InvalidRefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::InvalidRefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::IConstOp>(op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      return success();
   }
};

class AtomicRmwOpLowering : public OpConversionPattern<mlir::memref::AtomicRMWOp> {
   mlir::cranelift::AtomicRmwOpType translateOp(mlir::arith::AtomicRMWKind kind) const {
      switch (kind) {
         case mlir::arith::AtomicRMWKind::assign: return mlir::cranelift::AtomicRmwOpType::Xchg;
      }
      assert(false && "unsupported type");
   }

   public:
   using OpConversionPattern<mlir::memref::AtomicRMWOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::memref::AtomicRMWOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::AtomicRmwOp>(op, adaptor.value().getType(), translateOp(op.kind()), adaptor.memref(), adaptor.value());
      return success();
   }
};

} // end anonymous namespace

namespace {
mlir::TupleType convertTuple(mlir::TypeConverter& converter, mlir::TupleType t) {
   std::vector<mlir::Type> types;
   for (auto subT : t.getTypes()) {
      types.push_back(converter.convertType(subT));
   }
   return mlir::TupleType::get(t.getContext(), types);
}
struct ToCraneliftPass
   : public PassWrapper<ToCraneliftPass, OperationPass<ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "to-cl"; }

   ToCraneliftPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<mlir::cranelift::CraneliftDialect>();
   }
   void runOnOperation() final {
      auto module = getOperation();

      // Define Conversion Target
      ConversionTarget target(getContext());
      target.addLegalOp<ModuleOp>();
      target.addLegalDialect<mlir::BuiltinDialect>();
      target.addLegalOp<UnrealizedConversionCastOp>();

      target.addLegalDialect<cranelift::CraneliftDialect>();

      RewritePatternSet patterns(&getContext());
      const auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();
      mlir::TypeConverter typeConverter;
      typeConverter.addConversion([](mlir::util::VarLen32Type t) { return mlir::IntegerType::get(t.getContext(), 128); });
      typeConverter.addConversion([](mlir::util::RefType t) { return mlir::IntegerType::get(t.getContext(), 64); });
      typeConverter.addConversion([](mlir::MemRefType t) { return mlir::IntegerType::get(t.getContext(), 64); });
      typeConverter.addConversion([](mlir::IntegerType t) { return t; });
      typeConverter.addConversion([](mlir::FloatType t) { return t; });
      typeConverter.addConversion([](mlir::IndexType t) { return mlir::IntegerType::get(t.getContext(), 64); });
      typeConverter.addConversion([](mlir::FunctionType t) { return mlir::IntegerType::get(t.getContext(), 64); });

      typeConverter.addConversion([&typeConverter](mlir::TupleType t) { return convertTuple(typeConverter, t); });
      typeConverter.addSourceMaterialization([&](mlir::OpBuilder&, mlir::util::RefType type, mlir::ValueRange valueRange, mlir::Location loc) {
         return valueRange.front();
      });
      typeConverter.addTargetMaterialization([&](mlir::OpBuilder&, mlir::util::RefType type, mlir::ValueRange valueRange, mlir::Location loc) {
         return valueRange.front();
      });
      patterns.add<FuncLowering>(typeConverter, patterns.getContext());
      patterns.add<FuncConstLowering>(typeConverter, patterns.getContext());
      patterns.add<ConstLowering>(typeConverter, patterns.getContext());
      patterns.add<UndefLowering>(typeConverter, patterns.getContext());
      patterns.add<CallLowering>(typeConverter, patterns.getContext());
      patterns.add<BranchLowering>(typeConverter, patterns.getContext());
      patterns.add<CondBranchLowering>(typeConverter, patterns.getContext());
      patterns.add<CmpIOpLowering>(typeConverter, patterns.getContext());
      patterns.add<IndexCastOpLowering>(typeConverter, patterns.getContext());
      patterns.add<GenericMemrefCastLowering>(typeConverter, patterns.getContext());
      patterns.add<ExtFOpLowering>(typeConverter, patterns.getContext());
      patterns.add<ReturnLowering>(typeConverter, patterns.getContext());
      patterns.add<ExtSILowering>(typeConverter, patterns.getContext());
      patterns.add<ExtUILowering>(typeConverter, patterns.getContext());
      patterns.add<SelectLowering>(typeConverter, patterns.getContext());
      patterns.add<LoadOpLowering>(typeConverter, patterns.getContext());
      patterns.add<StoreOpLowering>(typeConverter, patterns.getContext());

      patterns.add<TruncILowering>(typeConverter, patterns.getContext());
      patterns.add<CreateConstVarLenLowering>(typeConverter, patterns.getContext());
      patterns.add<CreateVarLenLowering>(typeConverter, patterns.getContext());
      patterns.add<VarLenGetLenLowering>(typeConverter, patterns.getContext());
      patterns.add<AllocOpLowering>(typeConverter, patterns.getContext(), dataLayoutAnalysis);
      patterns.add<ArrayElementPtrOpLowering>(typeConverter, patterns.getContext(), dataLayoutAnalysis);

      patterns.add<Hash64Lowering>(typeConverter, patterns.getContext());
      patterns.add<FilterTaggedPtrLowering>(typeConverter, patterns.getContext());
      patterns.add<IsRefValidOpLowering>(typeConverter, patterns.getContext());
      patterns.add<InvalidRefOpLowering>(typeConverter, patterns.getContext());
      patterns.add<HashCombineLowering>(typeConverter, patterns.getContext());
      patterns.add<HashVarLenLowering>(typeConverter, patterns.getContext());
      patterns.add<ToMemrefLowering>(typeConverter, patterns.getContext());
      patterns.add<AtomicRmwOpLowering>(typeConverter, patterns.getContext());
      patterns.add<DivSILowering>(typeConverter, patterns.getContext());

      patterns.add<SizeOfOpLowering>(typeConverter, patterns.getContext(), dataLayoutAnalysis);
      patterns.add<TupleElementPtrOpLowering>(typeConverter, patterns.getContext(), dataLayoutAnalysis);
      patterns.add<AllocaOpLowering>(typeConverter, patterns.getContext(), dataLayoutAnalysis);
      patterns.add<SimpleArithmeticLowering<mlir::arith::ShRUIOp, mlir::cranelift::UShrOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::AddIOp, mlir::cranelift::IAddOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::SubIOp, mlir::cranelift::ISubOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::MulIOp, mlir::cranelift::IMulOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::DivUIOp, mlir::cranelift::UDivOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::RemUIOp, mlir::cranelift::URemOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::RemSIOp, mlir::cranelift::SRemOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::OrIOp, mlir::cranelift::BOrOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::XOrIOp, mlir::cranelift::BXOrOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::AndIOp, mlir::cranelift::BAndOp>>(typeConverter, patterns.getContext());

      patterns.add<SimpleArithmeticLowering<mlir::arith::AddFOp, mlir::cranelift::FAddOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::SubFOp, mlir::cranelift::FSubOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::MulFOp, mlir::cranelift::FMulOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::DivFOp, mlir::cranelift::FDivOp>>(typeConverter, patterns.getContext());

      if (failed(applyFullConversion(module, target, std::move(patterns))))
         signalPassFailure();
      OpBuilder builder(&getContext());
      builder.setInsertionPointToStart(module.getBody());
      builder.create<mlir::cranelift::GlobalOp>(builder.getUnknownLoc(), "execution_context", "abcdefgh");
      if (mlir::Operation* setECFunc = module.lookupSymbol("rt_set_execution_context")) {
         auto* setFnBlock = new Block;
         mlir::Value arg = setFnBlock->addArgument(builder.getI64Type(), builder.getUnknownLoc());
         mlir::cast<mlir::cranelift::FuncOp>(setECFunc).body().push_back(setFnBlock);
         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(setFnBlock);
            mlir::Value addr = builder.create<mlir::cranelift::AddressOfOp>(builder.getUnknownLoc(), "execution_context");
            builder.create<mlir::cranelift::StoreOp>(builder.getUnknownLoc(), arg, addr);
            builder.create<mlir::cranelift::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
         }
      }
      if (mlir::Operation* getECFunc = module.lookupSymbol("rt_get_execution_context")) {
         auto* getFnBlock = new Block;
         mlir::cast<mlir::cranelift::FuncOp>(getECFunc).body().push_back(getFnBlock);
         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(getFnBlock);
            mlir::Value addr = builder.create<mlir::cranelift::AddressOfOp>(builder.getUnknownLoc(), "execution_context");
            mlir::Value loaded = builder.create<mlir::cranelift::LoadOp>(builder.getUnknownLoc(), builder.getI64Type(), addr);
            builder.create<mlir::cranelift::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{loaded});
         }
      }
   }
};
}

void mlir::cranelift::registerCraneliftConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return std::make_unique<ToCraneliftPass>();
   });
}
std::unique_ptr<::mlir::Pass> mlir::cranelift::createLowerToCraneliftPass() {
   return std::make_unique<ToCraneliftPass>();
}