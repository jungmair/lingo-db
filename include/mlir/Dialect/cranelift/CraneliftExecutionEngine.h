#ifndef DB_DIALECTS_CRANELIFTEXECUTIONENGINE_H
#define DB_DIALECTS_CRANELIFTEXECUTIONENGINE_H
#include "ccranelift.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/cranelift/CraneliftOps.h"

namespace mlir::cranelift {
class CraneliftExecutionEngine {
   mlir::ModuleOp moduleOp;
   ModuleData* mod;
   size_t jitTime;
   std::unordered_map<std::string, uint32_t> functionIds;
   std::unordered_map<std::string, uint32_t> dataIds;

   bool success=true;
   public:
   CraneliftExecutionEngine(mlir::ModuleOp module);
   static uint8_t translateType(mlir::Type t);
   static uint8_t translateFuncType(mlir::Type t);

   void translate(mlir::cranelift::FuncOp fn);
   void translate(mlir::ModuleOp module);
   const uint8_t* getFunction(std::string name);
   bool succeeded(){
      return success;
   }
   size_t getJitTime() const;
};
}
#endif //DB_DIALECTS_CRANELIFTEXECUTIONENGINE_H
