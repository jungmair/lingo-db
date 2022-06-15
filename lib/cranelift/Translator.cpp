#include "mlir/Dialect/cranelift/CraneliftDialect.h"
#include "mlir/Dialect/cranelift/CraneliftExecutionEngine.h"
#include "mlir/Dialect/cranelift/CraneliftOps.h"

#include <iostream>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"
#include <chrono>
mlir::cranelift::CraneliftExecutionEngine::CraneliftExecutionEngine(mlir::ModuleOp module) : moduleOp(module) {
   mod = cranelift_module_new(
      "x86_64-pc-linux", "is_pic,enable_simd,enable_atomics,enable_llvm_abi_extensions", "mymodule", 0,
      [](uintptr_t userdata, const char* err, const char* fn) {
         printf("Error %s: %s\n", fn, err);
      },
      [](uintptr_t userdata, const char* err, const char* fn) { printf("Message %s: %s\n", fn, err); });
   auto start = std::chrono::high_resolution_clock::now();

   translate(module);
   auto end = std::chrono::high_resolution_clock::now();
   cranelift_compile(mod);
   jitTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
uint8_t mlir::cranelift::CraneliftExecutionEngine::translateType(mlir::Type t) {
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
      .Case<mlir::Float32Type>([&](mlir::Float32Type) { return TypeF32; })
      .Case<mlir::Float64Type>([&](mlir::Float64Type) { return TypeF64; })
      .Default([](mlir::Type) { return 0; });
}
uint8_t mlir::cranelift::CraneliftExecutionEngine::translateFuncType(mlir::Type t) {
   if (t.isInteger(1)) {
      return TypeI8;
   } else {
      return translateType(t);
   }
}

void mlir::cranelift::CraneliftExecutionEngine::translate(mlir::cranelift::FuncOp fn) {
   if (fn.getBody().empty()) {
      return;
   }
   cranelift_signature_builder_reset(mod, CraneliftCallConv::CraneliftCallConvSystemV);

   for (auto t : fn.getArgumentTypes()) {
      cranelift_signature_builder_add_param(mod, translateType(t));
   }
   for (auto t : fn.getResultTypes()) {
      cranelift_signature_builder_add_result(mod, translateType(t));
   }
   struct Context {
      CraneliftExecutionEngine* thisPtr;
      mlir::Operation* currFunc;
   };
   Context ctxt{this, fn.getOperation()};

   cranelift_build_function(mod, (uintptr_t) &ctxt, [](uintptr_t userdata, FunctionData* fd) {
      auto ctxt = (Context*) userdata;
      mlir::cranelift::FuncOp fn = mlir::cast<mlir::cranelift::FuncOp>(ctxt->currFunc);
      std::unordered_map<mlir::Block*, BlockCode> translatedBlocks;
      for (auto& x : fn.body().getBlocks()) {
         translatedBlocks.insert({&x, cranelift_create_block(fd)});
      }
      llvm::DenseMap<mlir::Value, ValueCode> varMapping;
      for (auto& x : fn.body().getBlocks()) {
         auto blockCode = translatedBlocks.at(&x);
         cranelift_switch_to_block(fd, blockCode);
         for (auto blockParamType : x.getArgumentTypes()) {
            cranelift_append_block_param(fd, blockCode, translateType(blockParamType));
         }
         std::vector<ValueCode> blockParams(x.getNumArguments());
         cranelift_block_params(fd, blockCode, blockParams.data());
         assert(x.getNumArguments() == cranelift_block_params_count(fd, blockCode));
         for (size_t i = 0; i < x.getNumArguments(); i++) {
            varMapping.insert({x.getArgument(i), blockParams[i]});
         }
         auto store = [&](mlir::Value v, ValueCode vc) {
            varMapping.insert({v, vc});
         };
         auto load = [&](mlir::Value v) {
            assert(varMapping.count(v));
            return varMapping[v];
         };
         for (auto& currentOp : x) {
            llvm::TypeSwitch<mlir::Operation*>(&currentOp)
               .Case([&](mlir::cranelift::IConstOp op) { store(op, cranelift_iconst(fd, translateType(op.getType()), op.value())); })
               .Case([&](mlir::cranelift::F32ConstOp op) { store(op, cranelift_f32const(fd, op.value().convertToFloat())); })
               .Case([&](mlir::cranelift::F64ConstOp op) { store(op, cranelift_f64const(fd, op.value().convertToDouble())); })
               .Case([&](mlir::cranelift::BConstOp op) { store(op, cranelift_bconst(fd, TypeB1, op.value())); })
               .Case([&](mlir::cranelift::IAddOp op) { store(op, cranelift_iadd(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::BIntOp op) { store(op, cranelift_bint(fd, translateType(op.getType()), load(op.value()))); })
               .Case([&](mlir::cranelift::ISubOp op) { store(op, cranelift_isub(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::IMulOp op) { store(op, cranelift_imul(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::UMulHiOp op) { store(op, cranelift_umulhi(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::FAddOp op) { store(op, cranelift_fadd(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::FSubOp op) { store(op, cranelift_fsub(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::FMulOp op) { store(op, cranelift_fmul(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::FDivOp op) { store(op, cranelift_fdiv(fd, load(op.lhs()), load(op.rhs()))); }) //todo: use immediate if possible?

               .Case([&](mlir::cranelift::UShrOp op) { store(op, cranelift_ushr(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::IShlOp op) { store(op, cranelift_ishl(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::BOrOp op) { store(op, cranelift_bor(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::BXOrOp op) {
                  assert(!op.getType().isInteger(128));
                  store(op, cranelift_bxor(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::BAndOp op) { store(op, cranelift_band(fd, load(op.lhs()), load(op.rhs()))); })
               .Case([&](mlir::cranelift::SelectOp op) {
                  assert(!op.getType().isInteger(128));
                  store(op, cranelift_select(fd, load(op.condition()), load(op.trueVal()), load(op.falseVal()))); })
               .Case([&](mlir::cranelift::ICmpOp op) { store(op, cranelift_icmp(fd, static_cast<CraneliftIntCC>(op.predicate()), load(op.lhs()), load(op.rhs()))); })

               .Case([&](mlir::cranelift::SExtendOp op) {
                  //assert(!op.getType().isInteger(128));
                  store(op, cranelift_sextend(fd, translateType(op.getType()), load(op.value()))); })
               .Case([&](mlir::cranelift::UExtendOp op) { store(op, cranelift_uextend(fd, translateType(op.getType()), load(op.value()))); })
               .Case([&](mlir::cranelift::FPromoteOp op) { store(op, cranelift_fpromote(fd, translateType(op.getType()), load(op.value()))); })
               .Case([&](mlir::cranelift::FDemoteOp op) { store(op, cranelift_fdemote(fd, translateType(op.getType()), load(op.value()))); })
               .Case([&](mlir::cranelift::IConcatOp op) { store(op, cranelift_iconcat(fd, load(op.lower()), load(op.higher()))); })
               .Case([&](mlir::cranelift::ISplitOp op) {
                  ValueCode lower, higher;
                  cranelift_isplit(fd, load(op.val()), &lower, &higher);

                  store(op.getResult(0), lower);
                  store(op.getResult(1), higher);
               })
               .Case([&](mlir::cranelift::IReduceOp op) {
                  store(op, cranelift_ireduce(fd, translateType(op.getType()), load(op.value()))); })
               .Case([&](mlir::cranelift::UDivOp op) { store(op, cranelift_udiv(fd, load(op.lhs()), load(op.rhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::SDivOp op) {
                  store(op, cranelift_sdiv(fd, load(op.lhs()), load(op.rhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::URemOp op) { store(op, cranelift_urem(fd, load(op.lhs()), load(op.rhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::SRemOp op) { store(op, cranelift_srem(fd, load(op.lhs()), load(op.rhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::AtomicRmwOp op) {
                  assert(op.rmw_op() == AtomicRmwOpType::Xchg);
                  store(op, cranelift_atomic_rmw(fd, translateType(op.getType()), "xchg", load(op.p()), load(op.x())));
               })

               .Case([&](mlir::cranelift::StoreOp op) {
                  if (op.x().getType().isInteger(1)) {
                     cranelift_store(fd, 0, cranelift_bint(fd, TypeI8, load(op.x())), load(op.p()), 0);
                  }else {
                     cranelift_store(fd, 0, load(op.x()), load(op.p()), 0);
                  }
               }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::LoadOp op) {
                  if (op.getType().isInteger(1)) {
                     auto loaded = cranelift_load(fd, TypeI8, 0, load(op.p()), 0);
                     store(op, cranelift_icmp_imm(fd, CraneliftIntCC::CraneliftIntCCNotEqual, loaded, 0));
                  } else {
                     store(op, cranelift_load(fd, translateType(op.getType()), 0, load(op.p()), 0));
                  }
               }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::AllocaOp op) {
                  auto stackSlot = cranelift_create_stack_slot(fd, op.size());
                  store(op.ref(), cranelift_stack_addr(fd, TypeI64, stackSlot, 0));
               })
               .Case([&](mlir::cranelift::AddressOfOp op) {
                  auto dataId = cranelift_declare_data_in_current_func(fd, ctxt->thisPtr->dataIds.at(op.symbol_name().str()));
                  store(op, cranelift_symbol_value(fd, translateType(op.getType()), dataId));
               })
               .Case([&](mlir::cranelift::ReturnOp op) {
                  std::vector<ValueCode> returnVals;
                  for (auto x : op.operands()) {
                     returnVals.push_back(varMapping[x]);
                  }
                  cranelift_return(fd, returnVals.size(), returnVals.data());
               })
               .Case([&](mlir::cranelift::CallOp op) {
                  uint32_t funcId;
                  if (ctxt->thisPtr->functionIds.contains(op.callee().str())) {
                     funcId = ctxt->thisPtr->functionIds[op.callee().str()];
                  } else {
                     std::vector<uint8_t> argTypes;
                     std::vector<uint8_t> resTypes;

                     for (auto t : op.getOperandTypes()) {
                        argTypes.push_back(translateType(t));
                     }
                     for (auto t : op.getResultTypes()) {
                        resTypes.push_back(translateType(t));
                     }

                     funcId = cranelift_import_func(fd, op.callee().data(), CraneliftCallConv::CraneliftCallConvSystemV, argTypes.size(), argTypes.data(), resTypes.size(), resTypes.data());
                     ctxt->thisPtr->functionIds[op.callee().str()] = funcId;
                  }
                  std::vector<ValueCode> args;
                  for (auto v : op.operands()) {
                     args.push_back(load(v));
                  }
                  auto declaredFn = cranelift_declare_func_in_current_func(fd, funcId);
                  auto inst = cranelift_call(fd, declaredFn, op.getNumOperands(), args.data());
                  for (size_t i = 0; i < op.getNumResults(); i++) {
                     store(op.getResult(i), cranelift_inst_result(fd, inst, i));
                  }
               })
               .Case([&](mlir::cranelift::FuncAddrOp op) {
                  auto declaredFn = cranelift_declare_func_in_current_func(fd, ctxt->thisPtr->functionIds.at(op.callee().str()));
                  store(op, cranelift_func_addr(fd, TypeI64, declaredFn));
               })
               .Case([&](mlir::cranelift::BranchOp op) {
                  std::vector<ValueCode> returnVals;
                  for (auto x : op.destOperands()) {
                     returnVals.push_back(varMapping[x]);
                  }
                  cranelift_ins_jump(fd, translatedBlocks.at(op.dest()), returnVals.size(), returnVals.data());
               })
               .Case([&](mlir::cranelift::CondBranchOp op) {
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

   /*cranelift_function_to_string(mod, 0, [](uintptr_t ud, const char* data) {
      printf("\n%s\n", data);
   });*/

   uint32_t id;
   cranelift_declare_function(mod, fn.sym_name().data(), CraneliftLinkage::Export, &id);
   cranelift_define_function(mod, id);
   functionIds[fn.sym_name().str()] = id;
   cranelift_clear_context(mod);
}

void mlir::cranelift::CraneliftExecutionEngine::translate(mlir::ModuleOp module) {
   for (mlir::Operation& op : module.getBody()->getOperations()) {
      if (auto globalOp = mlir::dyn_cast_or_null<mlir::cranelift::GlobalOp>(&op)) {
         uint32_t id2;
         cranelift_set_data_value(mod, (const uint8_t*) globalOp.value().data(), globalOp.value().size());
         cranelift_define_data(mod, globalOp.symbol_name().data(), CraneliftLinkage::Export, CraneliftDataFlags::Writable, 0, &id2);
         cranelift_assign_data_to_global(mod, id2);
         cranelift_clear_data(mod);
         dataIds[globalOp.symbol_name().str()] = id2;
      } else if (auto funcOp = mlir::dyn_cast_or_null<mlir::cranelift::FuncOp>(&op)) {
         translate(funcOp);
      }
   }
   success = true;
}

const uint8_t* mlir::cranelift::CraneliftExecutionEngine::getFunction(std::string name) {
   auto funcId = functionIds.at(name);
   return cranelift_get_compiled_fun(mod, funcId);
}
size_t mlir::cranelift::CraneliftExecutionEngine::getJitTime() const {
   return jitTime;
}
