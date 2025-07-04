#include "lingodb/catalog/Functions.h"

#include <lingodb/catalog/PythonUDF.h>
#include <lingodb/compiler/frontend/SQL/Parser.h>
#include <lingodb/utility/Serialization.h>

namespace lingodb::catalog {

void PyFunctionCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, name);
   serializer.writeProperty(2, argumentTypes);
   serializer.writeProperty(3, returnType);
   serializer.writeProperty(4, pythonCode);
}

std::shared_ptr<PyFunctionCatalogEntry> PyFunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(1);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(2);
   auto returnType = deserializer.readProperty<Type>(3);
   auto pythonCode = deserializer.readProperty<std::string>(4);
   return std::make_shared<PyFunctionCatalogEntry>(name, argumentTypes, returnType, pythonCode);
}
std::shared_ptr<MLIRFunctionImplementer> PyFunctionCatalogEntry::getImplementer() {
   return lingodb::compiler::frontend::createPythonUDFImplementer(name, pythonCode, argumentTypes, returnType);
}

} // namespace lingodb::catalog