//
// Created by michael on 7/4/25.
//

#ifndef PYTHONUDF_H
#define PYTHONUDF_H

#include "lingodb/catalog/MLIRTypes.h"

#include <lingodb/catalog/Types.h>

namespace lingodb::compiler::frontend {
std::shared_ptr<lingodb::catalog::MLIRFunctionImplementer> createPythonUDFImplementer(
   std::string funcName, std::string pythonCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType);
} // namespace lingodb::compiler::frontend

#endif //PYTHONUDF_H
