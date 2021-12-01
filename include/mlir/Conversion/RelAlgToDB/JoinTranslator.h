#ifndef MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#include "Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include <mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h>
#include <mlir/Dialect/RelAlg/IR/RelationalAttribute.h>

namespace mlir::relalg {
class JoinTranslator : public Translator {
   protected:
   Operator joinOp;
   mlir::relalg::Translator* builderChild;
   mlir::relalg::Translator* lookupChild;
   std::vector<size_t> customLookupBuilders;

   JoinTranslator(Operator joinOp, Value builderChild, Value lookupChild) : Translator({builderChild, lookupChild}), joinOp(joinOp) {
      this->builderChild = children[0].get();
      this->lookupChild = children[1].get();
      this->op = joinOp;
   }
   void addJoinRequiredAttributes() {
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      if (joinOp->hasAttr("mapping") && joinOp->getAttr("mapping").isa<mlir::ArrayAttr>()) {
         for (mlir::Attribute attr : joinOp->getAttr("mapping").cast<mlir::ArrayAttr>()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if (this->requiredAttributes.contains(defAttr)) {
               auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
               const auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
               this->requiredAttributes.insert(refAttr);
            }
         }
      }
   }
   void handleMappingNull(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope) {
      if (joinOp->hasAttr("mapping") && joinOp->getAttr("mapping").isa<mlir::ArrayAttr>()) {
         for (mlir::Attribute attr : joinOp->getAttr("mapping").cast<mlir::ArrayAttr>()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if (this->requiredAttributes.contains(defAttr)) {
               auto nullValue = builder.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
               context.setValueForAttribute(scope, defAttr, nullValue);
            }
         }
      }
   }
   void handleMapping(OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope) {
      if (joinOp->hasAttr("mapping") && joinOp->getAttr("mapping").isa<mlir::ArrayAttr>()) {
         for (mlir::Attribute attr : joinOp->getAttr("mapping").cast<mlir::ArrayAttr>()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if (this->requiredAttributes.contains(defAttr)) {
               auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
               const auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
               auto value = context.getValueForAttribute(refAttr);
               if (refAttr->type != defAttr->type) {
                  mlir::Value tmp = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
                  value = tmp;
               }
               context.setValueForAttribute(scope, defAttr, value);
            }
         }
      }
   }
   void handlePotentialMatch(OpBuilder& builder, TranslatorContext& context, Value matches, mlir::function_ref<void(OpBuilder&, TranslatorContext& context, TranslatorContext::AttributeResolverScope&)> onMatch = nullptr) {
      auto scope = context.createScope();
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), matches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
               if(onMatch){
                  onMatch(builder1,context,scope);
               }
               consumer->consume(this, builder1, context);
               builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); }, requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
      setRequiredBuilderValues(context, ifOp.getResults());
   }
   std::vector<mlir::Type> getRequiredBuilderTypesCustom(TranslatorContext& context) {
      auto requiredBuilderTypes = getRequiredBuilderTypes(context);
      for (auto x : customLookupBuilders) {
         requiredBuilderTypes.push_back(context.builders[x].getType());
      }
      return requiredBuilderTypes;
   }
   std::vector<mlir::Value> getRequiredBuilderValuesCustom(TranslatorContext& context) {
      auto requiredBuilderValues = getRequiredBuilderValues(context);
      for (auto x : customLookupBuilders) {
         requiredBuilderValues.push_back(context.builders[x]);
      }
      return requiredBuilderValues;
   }
   void setRequiredBuilderValuesCustom(TranslatorContext& context, mlir::ValueRange values) {
      size_t i = 0;
      for (auto x : requiredBuilders) {
         context.builders[x] = values[i++];
      }
      for (auto y : customLookupBuilders) {
         context.builders[y] = values[i++];
      }
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[1]->addRequiredBuilders(requiredBuilders);
   }

   public:
   virtual void addAdditionalRequiredAttributes() {}
   virtual void handleLookup(Value matched, Value markerBefore, TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(TranslatorContext& context, mlir::OpBuilder& builder) {}
   virtual void afterLookup(TranslatorContext& context, mlir::OpBuilder& builder) {}
   virtual void handleScanned(Value marker, TranslatorContext& context, mlir::OpBuilder& builder) {
   }
   virtual void after(TranslatorContext& context, mlir::OpBuilder& builder) {
   }
   virtual Value evaluatePredicate(TranslatorContext& context, mlir::OpBuilder& builder, TranslatorContext::AttributeResolverScope& scope) {
      bool hasRealPredicate = false;
      if (joinOp->getNumRegions() == 1 && joinOp->getRegion(0).hasOneBlock()) {
         auto terminator = mlir::cast<mlir::relalg::ReturnOp>(joinOp->getRegion(0).front().getTerminator());
         hasRealPredicate = !terminator.results().empty();
      }
      if (hasRealPredicate) {
         return mergeRelationalBlock(
            builder.getInsertionBlock(), joinOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
      } else {
         return builder.create<mlir::db::ConstantOp>(joinOp.getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
      }
   }
   virtual mlir::Value getFlag() { return Value(); }
};
} // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
