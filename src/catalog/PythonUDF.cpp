#include "lingodb/catalog/PythonUDF.h"

#include "json.h"

#include <llvm/Support/SourceMgr.h>

#include <fstream>
#include <iostream>
#include <lingodb/compiler/Dialect/DB/IR/DBOps.h>
#include <lingodb/execution/Backend.h>
#include <lingodb/execution/Frontend.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

namespace mlir {
class ModuleOp;
}
namespace {
class PythonUDFImplementer : public lingodb::catalog::MLIRFunctionImplementer {
   // properties from the catalog
   std::string funcName;
   std::string pythonCode;
   std::vector<lingodb::catalog::Type> argumentTypes;
   lingodb::catalog::Type returnType;

   public:
   PythonUDFImplementer(std::string funcName, std::string pythonCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType)
      : funcName(std::move(funcName)), pythonCode(std::move(pythonCode)), argumentTypes(std::move(argumentTypes)), returnType(std::move(returnType)) {}

   mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) override {
      using namespace lingodb::compiler::dialect;

      mlir::OwningOpRef<mlir::ModuleOp> module;

      std::string tempFilePath, outputFilePath;
      try {
         // Create a temporary file path for the Python code
         char tempFileTemplate[] = "/tmp/python_udf_XXXXXX";
         int fd = mkstemp(tempFileTemplate);
         if (fd == -1) {
            throw std::runtime_error("Failed to create temporary file.");
         }
         tempFilePath = tempFileTemplate;

         // Write Python code to the temporary file
         std::ofstream tempFile(tempFilePath, std::ios::out | std::ios::trunc);
         if (!tempFile.is_open()) {
            throw std::runtime_error("Failed to open temporary file for writing Python code.");
         }
         tempFile << pythonCode;
         tempFile.close();

         // Create a temporary file path for the output
         char outputFileTemplate[] = "/tmp/python_udf_out_XXXXXX";
         int fdOut = mkstemp(outputFileTemplate);
         if (fdOut == -1) {
            throw std::runtime_error("Failed to create temporary output file.");
         }
         outputFilePath = outputFileTemplate;

         // Prepare JSON argument types
         nlohmann::json jsonArgs = nlohmann::json::array();
         for (const auto& argType : argumentTypes) {
            switch (argType.getTypeId()) {
               case lingodb::catalog::LogicalTypeId::BOOLEAN:
                  jsonArgs.push_back("bool");
                  break;
               case lingodb::catalog::LogicalTypeId::DOUBLE:
                  jsonArgs.push_back("float");
                  break;
               case lingodb::catalog::LogicalTypeId::STRING:
                  jsonArgs.push_back("str");
                  break;
               default:
                  throw std::runtime_error("Unsupported argument type for Python UDF: " + argType.toString());
            }
         }
         std::string jsonArgsStr = jsonArgs.dump();

         // Step 2: Invoke the external script
         std::ostringstream command;
         command << "~/projects/hipy/venv/bin/python3 vendored/hipy/compile.py " << tempFilePath << " " << funcName << " '" << jsonArgsStr << "' " << outputFilePath << " 2>&1";
         std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.str().c_str(), "r"), pclose);
         if (!pipe) {
            throw std::runtime_error("Failed to execute compile.py script.");
         }

         std::string output;
         char buffer[128];
         while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            output += buffer;
         }

         int returnCode = pclose(pipe.release());
         if (returnCode != 0) {
            throw std::runtime_error("compile.py script failed with return code: " + std::to_string(returnCode) + "\nOutput:\n" + output);
         }
         // Step 3: Read back the output file
         llvm::SourceMgr sourceMgr;
         mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, builder.getContext());
         llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(outputFilePath);
         if (std::error_code ec = fileOrErr.getError()) {
            throw std::runtime_error("Could not open input file: " + ec.message() + "\n");
         }

         // Parse the input mlir.
         sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
         module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, builder.getContext());
         if (!module) {
            throw std::runtime_error("Error can't load file " + outputFilePath + "\n");
         }
      } catch (const std::exception& e) {
         throw std::runtime_error(std::string("Error during compilation: ") + e.what());
      }
      std::vector<mlir::Operation*> toMove;
      for (auto& op : module->getOps()) {
         toMove.push_back(&op);
      }
      for (auto* op : toMove) {
         op->remove();
         if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
            funcOp.setSymVisibility("private");
         }
         moduleOp.getBody()->push_back(op);
      }
      std::vector<mlir::Value> values;
      std::vector<mlir::Value> isNull;
      for (auto arg : args) {
         values.push_back(arg);
         if (mlir::isa<db::NullableType>(arg.getType())) {
            isNull.push_back(builder.create<db::IsNullOp>(loc, arg));
         }
      }
      if (isNull.size() > 0) {
         auto allNotNull = builder.create<db::OrOp>(loc, isNull);
         auto* elseBlock = new mlir::Block;
         mlir::Type resType;
         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(elseBlock);
            std::vector<mlir::Value> notNullValues;
            for (auto v : values) {
               notNullValues.push_back(mlir::isa<db::NullableType>(v.getType()) ? builder.create<db::NullableGetVal>(loc, mlir::cast<db::NullableType>(v.getType()).getType(), v) : v);
            }
            auto func = mlir::cast<mlir::func::FuncOp>(moduleOp.lookupSymbol(funcName));
            auto res = builder.create<mlir::func::CallOp>(loc, func, notNullValues).getResult(0);
            mlir::Value resNullable = builder.create<db::AsNullableOp>(loc, db::NullableType::get(res.getType()), res);

            resType = resNullable.getType();
            builder.create<mlir::scf::YieldOp>(loc, resNullable);
         }
         auto* thenBlock = new mlir::Block;

         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(thenBlock);
            mlir::Value res = builder.create<db::NullOp>(loc, resType);
            builder.create<mlir::scf::YieldOp>(loc, res);
         }
         auto ifOp = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{resType}, allNotNull, false);
         ifOp.getThenRegion().getBlocks().clear();
         ifOp.getThenRegion().push_back(thenBlock);
         ifOp.getElseRegion().getBlocks().clear();
         ifOp.getElseRegion().push_back(elseBlock);
         return ifOp.getResult(0);
      }
      auto func = mlir::cast<mlir::func::FuncOp>(moduleOp.lookupSymbol(funcName));

      return builder.create<mlir::func::CallOp>(loc, func, values).getResult(0);
   }
};
} // namespace

std::shared_ptr<lingodb::catalog::MLIRFunctionImplementer> lingodb::compiler::frontend::createPythonUDFImplementer(
   std::string funcName, std::string pythonCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType) {
   return std::make_shared<PythonUDFImplementer>(std::move(funcName), std::move(pythonCode), std::move(argumentTypes), std::move(returnType));
}