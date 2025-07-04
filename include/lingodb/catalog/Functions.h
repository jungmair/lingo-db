#ifndef LINGODB_CATALOG_FUNCTIONS_H
#define LINGODB_CATALOG_FUNCTIONS_H
#include "Catalog.h"
#include "Column.h"

#include <vector>

namespace lingodb::catalog {
class MLIRFunctionImplementer;
class FunctionCatalogEntry : public CatalogEntry {
   protected:
   std::string name;
   std::vector<Type> argumentTypes;
   Type returnType;

   static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::PY_FUNCTION_ENTRY};

   public:
   FunctionCatalogEntry(CatalogEntryType entryType, std::string name, std::vector<Type> argumentTypes, Type returnType)
      : CatalogEntry(entryType), name(std::move(name)), argumentTypes(std::move(argumentTypes)), returnType(std::move(returnType)) {}
   virtual std::shared_ptr<MLIRFunctionImplementer> getImplementer() = 0;
   std::string getName() override { return name; }
};

class PyFunctionCatalogEntry : public FunctionCatalogEntry {
   std::string pythonCode;

   public:
   PyFunctionCatalogEntry(std::string name, std::vector<Type> argumentTypes, Type returnType, std::string pythonCode)
      : FunctionCatalogEntry(CatalogEntryType::PY_FUNCTION_ENTRY, std::move(name), std::move(argumentTypes), std::move(returnType)), pythonCode(pythonCode) {}
   static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::PY_FUNCTION_ENTRY};

   void serializeEntry(lingodb::utility::Serializer& serializer) const override;
   static std::shared_ptr<PyFunctionCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
   std::shared_ptr<MLIRFunctionImplementer> getImplementer() override;
};

} // namespace lingodb::catalog

#endif //LINGODB_CATALOG_FUNCTIONS_H
