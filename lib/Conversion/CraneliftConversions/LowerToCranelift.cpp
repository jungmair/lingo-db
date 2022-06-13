#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
      auto funcOp = rewriter.create<mlir::cranelift::FuncOp>(op->getLoc(), op.getSymName(), op.getFunctionType());
      rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(),
                                  funcOp.end());
      /*TypeConverter::SignatureConversion result(funcOp.getNumArguments());
      if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter, &result)))
         return failure();*/
      rewriter.eraseOp(op);
      return success();
   }
};
class CallLowering : public OpConversionPattern<mlir::func::CallOp> {
   public:
   using OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type,4> resTypes;
      typeConverter->convertTypes(op.getResultTypes(),resTypes);
      auto callOp = rewriter.replaceOpWithNewOp<mlir::cranelift::CallOp>(op, resTypes, op.getCallee(), adaptor.operands());
      callOp.dump();
      return success();
   }
};
class ReturnLowering : public OpConversionPattern<mlir::func::ReturnOp> {
   public:
   using OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::func::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto funcOp = rewriter.replaceOpWithNewOp<mlir::cranelift::ReturnOp>(op, adaptor.operands());
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
class SelectLowering : public OpConversionPattern<mlir::arith::SelectOp> {
   public:
   using OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::cranelift::SelectOp>(op, adaptor.getFalseValue().getType(), adaptor.getCondition(), adaptor.getTrueValue(), adaptor.getFalseValue());
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
template <class From, class To>
class SimpleArithmeticLowering : public OpConversionPattern<From> {
   public:
   using OpConversionPattern<From>::OpConversionPattern;
   LogicalResult matchAndRewrite(From op, typename From::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<To>(op, op.getType(), adaptor.getOperands());
      return success();
   }
};
class ConstLowering : public OpConversionPattern<mlir::arith::ConstantOp> {
   public:
   using OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
            rewriter.replaceOpWithNewOp<mlir::cranelift::IConcatOp>(op, op.getType(), lowV, highV);
         } else {
            rewriter.replaceOpWithNewOp<mlir::cranelift::IConstOp>(op, op.getType(), op.getValue().cast<mlir::IntegerAttr>().getInt());
         }
         return success();
      } else if (op.getType().isIntOrIndex()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::IConstOp>(op, op.getType(), op.getValue().cast<mlir::IntegerAttr>().getInt());
         return success();
      } else if (op.getType().isF32()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::F32ConstOp>(op, op.getType(), op.getValue().cast<mlir::FloatAttr>());
         return success();
      } else if (op.getType().isF64()) {
         rewriter.replaceOpWithNewOp<mlir::cranelift::F64ConstOp>(op, op.getType(), op.getValue().cast<mlir::FloatAttr>());
         return success();
      }
      return failure();
   }
};

class CreateConstVarLenLowering : public OpConversionPattern<mlir::util::CreateConstVarLen> {
   public:
   using OpConversionPattern<mlir::util::CreateConstVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::CreateConstVarLen op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      size_t len = op.str().size();

      mlir::Type i128Ty = rewriter.getIntegerType(128);
      mlir::Value p1, p2;

      uint64_t first4 = 0;
      memcpy(&first4, op.str().data(), std::min(4ul, len));
      size_t c1 = (first4 << 32) | len;
      p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, c1));
      if (len <= 12) {
         uint64_t last8 = 0;
         if (len > 4) {
            memcpy(&last8, op.str().data() + 4, std::min(8ul, len - 4));
         }
         p2 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, last8));
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
      auto const64 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
      auto shlp2 = rewriter.create<mlir::cranelift::IShlOp>(op->getLoc(), p2, const64);
      rewriter.replaceOpWithNewOp<mlir::cranelift::BOrOp>(op, p1, shlp2);
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

} // end anonymous namespace

namespace {
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
      mlir::TypeConverter typeConverter;
      typeConverter.addConversion([](mlir::util::VarLen32Type t) { return mlir::IntegerType::get(t.getContext(), 128); });
      typeConverter.addConversion([](mlir::IntegerType t) { return t; });
      typeConverter.addConversion([](mlir::IndexType t) { return mlir::IntegerType::get(t.getContext(), 64);  });

      patterns.add<FuncLowering>(typeConverter, patterns.getContext());
      patterns.add<ConstLowering>(typeConverter, patterns.getContext());
      patterns.add<CallLowering>(typeConverter, patterns.getContext());
      patterns.add<ReturnLowering>(typeConverter, patterns.getContext());
      patterns.add<ExtSILowering>(typeConverter, patterns.getContext());
      patterns.add<SelectLowering>(typeConverter, patterns.getContext());

      patterns.add<TruncILowering>(typeConverter, patterns.getContext());
      patterns.add<CreateConstVarLenLowering>(typeConverter, patterns.getContext());
      patterns.add<VarLenGetLenLowering>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::ShRUIOp, mlir::cranelift::UShrOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::AddIOp, mlir::cranelift::IAddOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::SubIOp, mlir::cranelift::ISubOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::MulIOp, mlir::cranelift::IMulOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::DivSIOp, mlir::cranelift::SDivOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::DivUIOp, mlir::cranelift::UDivOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::RemUIOp, mlir::cranelift::URemOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::RemSIOp, mlir::cranelift::SRemOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::OrIOp, mlir::cranelift::BOrOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::XOrIOp, mlir::cranelift::BXOrOp>>(typeConverter, patterns.getContext());
      patterns.add<SimpleArithmeticLowering<mlir::arith::AndIOp, mlir::cranelift::BAndOp>>(typeConverter, patterns.getContext());

      if (failed(applyFullConversion(module, target, std::move(patterns))))
         signalPassFailure();
   }
};
}

void mlir::cranelift::registerCraneliftConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return std::make_unique<ToCraneliftPass>();
   });
}