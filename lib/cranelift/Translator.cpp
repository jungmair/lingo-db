#include "ccranelift.h"
#include "mlir/Dialect/cranelift/CraneliftDialect.h"
#include "mlir/Dialect/cranelift/CraneliftOps.h"
#include <iostream>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"

namespace {

class Translator {
   ModuleData* mod;

   public:
   Translator() {
      mod = cranelift_module_new(
         "x86_64-pc-linux", "is_pic,enable_simd,enable_atomics,enable_verifier,enable_llvm_abi_extensions", "mymodule", 0, [](uintptr_t userdata, const char* err, const char* fn) {
            printf("Error %s: %s\n", fn, err);
         }, [](uintptr_t userdata, const char* err, const char* fn) { printf("Message %s: %s\n", fn, err); });
   }
   static uint8_t translateType(mlir::Type t) {
      return ::llvm::TypeSwitch<mlir::Type, uint8_t>(t)
         .Case<mlir::IntegerType>([&](mlir::IntegerType integerType) {
            switch (integerType.getWidth()) {
               case 1: return TypeB1;
               case 8: return TypeI8;
               case 16: return TypeI16;
               case 32: return TypeI32;
               case 64: return TypeI64;
               case 128: return TypeI128;
            }
            assert(false && "unknown type");
         })
         .Default([](mlir::Type) { return 0; });
   }
   void translate(mlir::cranelift::FuncOp fn) {
      if(fn.getBody().empty()){
         return;
      }
      cranelift_signature_builder_reset(mod,CraneliftCallConv::CraneliftCallConvSystemV);

      for (auto t : fn.getArgumentTypes()) {
         cranelift_signature_builder_add_param(mod, translateType(t));
      }
      for (auto t : fn.getResultTypes()) {
         cranelift_signature_builder_add_result(mod, translateType(t));
      }
      cranelift_build_function(mod, (uintptr_t) fn.getOperation(), [](uintptr_t userdata, FunctionData* fd) {
         mlir::cranelift::FuncOp fn = mlir::cast<mlir::cranelift::FuncOp>((mlir::Operation*) userdata);
         std::unordered_map<mlir::Block*, BlockCode> translatedBlocks;
         for (auto& x : fn.body().getBlocks()) {
            translatedBlocks.insert({&x, cranelift_create_block(fd)});
         }
         for (auto& x : fn.body().getBlocks()) {
            auto blockCode = translatedBlocks.at(&x);
            cranelift_switch_to_block(fd, blockCode);
            for (auto blockParamType : x.getArgumentTypes()) {
               cranelift_append_block_param(fd, blockCode, translateType(blockParamType));
            }
            std::vector<ValueCode> blockParams(x.getNumArguments());
            cranelift_block_params(fd, blockCode, blockParams.data());
            llvm::DenseMap<mlir::Value, ValueCode> varMapping;
            assert(x.getNumArguments() == cranelift_block_params_count(fd, blockCode));
            for (size_t i = 0; i < x.getNumArguments(); i++) {
               varMapping.insert({x.getArgument(i), blockParams[i]});
            }
            auto store=[&](mlir::Value v,ValueCode vc){
               varMapping.insert({v, vc});
            };
            auto load=[&](mlir::Value v){return varMapping[v];};
            for (auto& currentOp : x) {
               llvm::TypeSwitch<mlir::Operation*>(&currentOp)
                  .Case([&](mlir::cranelift::IConstOp op){store(op,cranelift_iconst(fd,translateType(op.getType()),op.value()));})
                  .Case([&](mlir::cranelift::F32ConstOp op){store(op,cranelift_f32const(fd,op.value().convertToFloat()));})
                  .Case([&](mlir::cranelift::F64ConstOp op){store(op,cranelift_f64const(fd,op.value().convertToDouble()));})
                  .Case([&](mlir::cranelift::BConstOp op){store(op,cranelift_bconst(fd,TypeB1,op.value()));})
                  .Case([&](mlir::cranelift::IAddOp op){store(op,cranelift_iadd(fd,load(op.lhs()),load(op.rhs())));})
                  .Case([&](mlir::cranelift::ISubOp op){store(op,cranelift_isub(fd,load(op.lhs()),load(op.rhs())));})
                  .Case([&](mlir::cranelift::IMulOp op){store(op,cranelift_imul(fd,load(op.lhs()),load(op.rhs())));})
                  .Case([&](mlir::cranelift::UShrOp op){store(op,cranelift_ushr(fd,load(op.lhs()),load(op.rhs())));})
                  .Case([&](mlir::cranelift::IShlOp op){store(op,cranelift_ishl(fd,load(op.lhs()),load(op.rhs())));})
                  .Case([&](mlir::cranelift::BOrOp op){store(op,cranelift_bor(fd,load(op.lhs()),load(op.rhs())));})
                  .Case([&](mlir::cranelift::BXOrOp op){store(op,cranelift_bxor(fd,load(op.lhs()),load(op.rhs())));})
                  .Case([&](mlir::cranelift::BAndOp op){store(op,cranelift_band(fd,load(op.lhs()),load(op.rhs())));})

                  .Case([&](mlir::cranelift::SExtendOp op){store(op,cranelift_sextend(fd,translateType(op.getType()),load(op.value())));})
                  .Case([&](mlir::cranelift::IConcatOp op){store(op,cranelift_iconcat(fd,load(op.lower()),load(op.higher())));})
                  .Case([&](mlir::cranelift::IReduceOp op){store(op,cranelift_ireduce(fd,translateType(op.getType()),load(op.value())));})
                  .Case([&](mlir::cranelift::UDivOp op){store(op,cranelift_udiv(fd,load(op.lhs()),load(op.rhs())));})//todo: use immediate if possible?
                  .Case([&](mlir::cranelift::SDivOp op){store(op,cranelift_sdiv(fd,load(op.lhs()),load(op.rhs())));})//todo: use immediate if possible?
                  .Case([&](mlir::cranelift::URemOp op){store(op,cranelift_urem(fd,load(op.lhs()),load(op.rhs())));})//todo: use immediate if possible?
                  .Case([&](mlir::cranelift::SRemOp op){store(op,cranelift_srem(fd,load(op.lhs()),load(op.rhs())));})//todo: use immediate if possible?
                  .Case([&](mlir::cranelift::ReturnOp op){
                     std::vector<ValueCode> returnVals;
                     for (auto x : op.operands()) {
                        returnVals.push_back(varMapping[x]);
                     }
                     cranelift_return(fd, returnVals.size(), returnVals.data());
                  })
                  .Case([&](mlir::cranelift::CallOp op){
                     //op.
                     std::vector<uint8_t> argTypes;
                     std::vector<uint8_t> resTypes;
                     std::vector<ValueCode> args;

                     for(auto t:op.getOperandTypes()){
                        argTypes.push_back(translateType(t));
                     }
                     for(auto t:op.getResultTypes()){
                        resTypes.push_back(translateType(t));
                     }
                     for(auto v:op.operands()){
                        args.push_back(load(v));
                     }
                     auto importedFunc=cranelift_import_func(fd,op.callee().data(),CraneliftCallConv::CraneliftCallConvSystemV,argTypes.size(),argTypes.data(),resTypes.size(),resTypes.data());
                     auto declaredFn=cranelift_declare_func_in_current_func(fd,importedFunc);
                     auto inst=cranelift_call(fd,declaredFn,op.getNumOperands(),args.data());
                     for(size_t i=0;i<op.getNumResults();i++){
                        store(op.getResult(i),cranelift_inst_result(fd,inst,i));
                     }
                  })
                  .Case([&](mlir::cranelift::BranchOp op){
                     std::vector<ValueCode> returnVals;
                     for (auto x : op.destOperands()) {
                        returnVals.push_back(varMapping[x]);
                     }
                     cranelift_ins_jump(fd, translatedBlocks.at(op.dest()), returnVals.size(), returnVals.data());
                  })
                  .Case([&](mlir::cranelift::CondBranchOp op){
                     std::vector<ValueCode> falseDestVals;
                     for (auto x : op.falseDestOperands()) {
                        falseDestVals.push_back(varMapping[x]);
                     }
                     std::vector<ValueCode> trueDestVals;
                     for (auto x : op.trueDestOperands()) {
                        trueDestVals.push_back(varMapping[x]);
                     }
                     cranelift_ins_brnz(fd, load(op.condition()), translatedBlocks.at(op.trueDest()), trueDestVals.size(), trueDestVals.data());

                     cranelift_ins_jump(fd, translatedBlocks.at(op.falseDest()), falseDestVals.size(), falseDestVals.data());
                  })
                  .Default([](mlir::Operation* op) {op->dump();assert(false&&"unknown operation"); });
            }
         }
         cranelift_seal_all_blocks(fd);
      });
      cranelift_function_to_string(mod, 0, [](uintptr_t ud, const char* data) {
         printf("\n%s\n", data);
      });
      uint32_t id;
      cranelift_declare_function(mod, fn.sym_name().data(), CraneliftLinkage::Export, &id);
      cranelift_define_function(mod, id);
      if(fn.sym_name().str()=="main") {
         auto fn_ptr = cranelift_jit(mod, id);

         typedef int64_t (*mainFn)();
         auto res = ((mainFn) fn_ptr)();
         std::cout << "result:" << res << std::endl;
      }
      //cranelift_signature_builder_reset(mod,CraneliftCallConv::CraneliftCallConvSystemV);
      cranelift_clear_context(mod);
   }
   void translate(mlir::ModuleOp module) {
      module.walk([this](mlir::cranelift::FuncOp funcOp) {
         translate(funcOp);
      });
      std::cout<<"done"<<std::endl;
      /*
      cranelift_signature_builder_add_param(mod, TypeI128);
      cranelift_signature_builder_add_param(mod, TypeI128);
      cranelift_signature_builder_add_result(mod, TypeI128);
      cranelift_build_function(mod, 0, [](uintptr_t userdata, FunctionData* fd) {
         BlockCode entry = cranelift_create_block(fd);
         BlockCode larger = cranelift_create_block(fd);
         BlockCode exit = cranelift_create_block(fd);

         cranelift_append_block_params_for_function_params(fd, entry);

         cranelift_switch_to_block(fd, entry);

         ValueCode res[2];
         cranelift_block_params(fd, entry, res);

         VariableCode a = cranelift_declare_var(fd, TypeI128);
         cranelift_def_var(fd, a, res[0]);

         VariableCode b = cranelift_declare_var(fd, TypeI128);
         cranelift_def_var(fd, b, res[1]);

         VariableCode retvar = cranelift_declare_var(fd, TypeI128);
         cranelift_def_var(fd, retvar, cranelift_iconst(fd, TypeI128, 0));

         cranelift_ins_br_icmp(fd, CraneliftIntCC::CraneliftIntCCSignedLessThan, a, b, larger, 0, NULL);

         cranelift_def_var(fd, retvar, cranelift_use_var(fd, a));

         cranelift_ins_jump(fd, exit, 0, NULL);

         cranelift_switch_to_block(fd, larger);

         cranelift_def_var(fd, retvar, cranelift_use_var(fd, b));

         cranelift_ins_jump(fd, exit, 0, NULL);

         cranelift_switch_to_block(fd, exit);

         ValueCode retval = cranelift_use_var(fd, retvar);
         cranelift_return(fd, 1, &retval);

         cranelift_seal_all_blocks(fd);
      });
      cranelift_function_to_string(mod, 0, [](uintptr_t ud, const char* data) {
         printf("\n%s\n", data);
      });
      uint32_t id;
      cranelift_declare_function(mod, "main", CraneliftLinkage::Export, &id);
      cranelift_define_function(mod, id);*/
   }
};

class ToCLIRPass : public mlir::PassWrapper<ToCLIRPass, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "to-cl-ir"; }

   public:
   void runOnOperation() override {
      Translator translator;
      getOperation().dump();
      translator.translate(getOperation());
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::cranelift::createToCLIRPass() {
   return std::make_unique<ToCLIRPass>();
}