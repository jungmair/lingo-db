#include "mlir/Conversion/RelAlgToDB/Translator.h"
using namespace mlir::relalg;
std::vector<mlir::Value> mlir::relalg::Translator::mergeRelationalBlock(mlir::Block* dest, mlir::Operation* op, mlir::function_ref<mlir::Block*(mlir::Operation*)> getBlockFn, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope) {
   // Splice the operations of the 'source' block into the 'dest' block and erase
   // it.
   llvm::iplist<mlir::Operation> translated;
   std::vector<mlir::Operation*> toErase;
   auto* cloned = op->clone();
   mlir::Block* source = getBlockFn(cloned);
   auto* terminator = source->getTerminator();

   source->walk([&](mlir::relalg::GetAttrOp getAttrOp) {
      getAttrOp.replaceAllUsesWith(context.getValueForAttribute(&getAttrOp.attr().getRelationalAttribute()));
      toErase.push_back(getAttrOp.getOperation());
   });
   for (auto addAttrOp : source->getOps<mlir::relalg::AddAttrOp>()) {
      context.setValueForAttribute(scope, &addAttrOp.attr().getRelationalAttribute(), addAttrOp.val());
      toErase.push_back(addAttrOp.getOperation());
   }

   dest->getOperations().splice(dest->end(), source->getOperations());
   for (auto* op : toErase) {
      op->dropAllUses();
      op->erase();
   }
   auto returnOp = mlir::cast<mlir::relalg::ReturnOp>(terminator);
   std::vector<Value> res(returnOp.results().begin(), returnOp.results().end());
   terminator->erase();
   return res;
}

void Translator::propagateInfo() {
   for (auto& c : children) {
      auto available = c->getAvailableAttributes();
      mlir::relalg::Attributes toPropagate = requiredAttributes.intersect(available);
      c->setInfo(this, toPropagate);
   }
}
std::vector<mlir::Value> Translator::getRequiredBuilderValues(TranslatorContext& context) {
   std::vector<mlir::Value> res;
   for (auto x : requiredBuilders) {
      res.push_back(context.builders[x]);
   }
   return res;
}
void Translator::setRequiredBuilderValues(TranslatorContext& context, const mlir::ValueRange& values) {
   size_t i = 0;
   for (auto x : requiredBuilders) {
      context.builders[x] = values[i++];
   }
}
mlir::Value Translator::packValues(TranslatorContext& context, OpBuilder builder, const std::vector<const mlir::relalg::RelationalAttribute*>& attrs, const std::vector<Value>& additional) {
   auto loc = builder.getUnknownLoc();
   std::vector<Value> values(additional);
   for (const auto* attr : attrs) {
      values.push_back(context.getValueForAttribute(attr));
   }
   if (values.size() == 0) {
      return builder.create<mlir::util::UndefTupleOp>(loc, mlir::TupleType::get(builder.getContext()));
   }
   return builder.create<mlir::util::PackOp>(loc, values);
}
mlir::Value Translator::packValues(TranslatorContext& context, OpBuilder builder, const mlir::relalg::Attributes& attrs) {
   auto loc = builder.getUnknownLoc();
   std::vector<Value> values;
   for (const auto* attr : attrs) {
      values.push_back(context.getValueForAttribute(attr));
   }
   if (values.size() == 0) {
      return builder.create<mlir::util::UndefTupleOp>(loc, mlir::TupleType::get(builder.getContext()));
   }
   return builder.create<mlir::util::PackOp>(loc, values);
}
std::vector<mlir::Type> Translator::getRequiredBuilderTypes(TranslatorContext& context) {
   std::vector<mlir::Type> res;
   for (auto x : requiredBuilders) {
      res.push_back(context.builders[x].getType());
   }
   return res;
}
mlir::relalg::Translator::Translator(Operator op) : op(op) {
   for (auto child : op.getChildren()) {
      children.push_back(mlir::relalg::Translator::createTranslator(child.getOperation()));
   }
}

mlir::relalg::Translator::Translator(mlir::ValueRange potentialChildren) : op() {
   for (auto child : potentialChildren) {
      if (child.getType().isa<mlir::relalg::TupleStreamType>()) {
         children.push_back(mlir::relalg::Translator::createTranslator(child.getDefiningOp()));
      }
   }
}
void Translator::addRequiredBuilders(std::vector<size_t> requiredBuilders) {
   this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   for (auto& child : children) {
      child->addRequiredBuilders(requiredBuilders);
   }
}

void Translator::setFlag(mlir::Value flag) {
   this->flag = flag;
   for (auto& child : children) {
      child->setFlag(flag);
   }
}
void Translator::setInfo(mlir::relalg::Translator* consumer, mlir::relalg::Attributes requiredAttributes) {
   this->consumer = consumer;
   this->requiredAttributes = requiredAttributes;
   if (op) {
      this->requiredAttributes.insert(op.getUsedAttributes());
      propagateInfo();
   }
}
mlir::relalg::Attributes Translator::getAvailableAttributes() {
   return op.getAvailableAttributes();
};
std::unique_ptr<mlir::relalg::Translator> Translator::createTranslator(mlir::Operation* operation) {
   return ::llvm::TypeSwitch<mlir::Operation*, std::unique_ptr<mlir::relalg::Translator>>(operation)
      .Case<BaseTableOp>([&](auto x) { return createBaseTableTranslator(x); })
      .Case<ConstRelationOp>([&](auto x) { return createConstRelTranslator(x); })
      .Case<MaterializeOp>([&](auto x) { return createMaterializeTranslator(x); })
      .Case<SelectionOp>([&](auto x) { return createSelectionTranslator(x); })
      .Case<MapOp>([&](auto x) { return createMapTranslator(x); })
      .Case<CrossProductOp>([&](auto x) { return createCrossProductTranslator(x); })
      .Case<SortOp>([&](auto x) { return createSortTranslator(x); })
      .Case<AggregationOp>([&](auto x) { return createAggregationTranslator(x); })
      .Case<InnerJoinOp>([&](auto x) { return createInnerJoinTranslator(x); })
      .Case<SemiJoinOp>([&](auto x) { return createSemiJoinTranslator(x); })
      .Case<AntiSemiJoinOp>([&](auto x) { return createAntiSemiJoinTranslator(x); })
      .Case<RenamingOp>([&](auto x) { return createRenamingTranslator(x); })
      .Case<ProjectionOp>([&](auto x) { return createProjectionTranslator(x); })
      .Case<LimitOp>([&](auto x) { return createLimitTranslator(x); })
      .Case<OuterJoinOp>([&](auto x) { return createOuterJoinTranslator(x); })
      .Case<SingleJoinOp>([&](auto x) { return createSingleJoinTranslator(x); })
      .Case<MarkJoinOp>([&](auto x) { return createMarkJoinTranslator(x); })
      .Case<TmpOp>([&](auto x) { return createTmpTranslator(x); })
      .Case<CollectionJoinOp>([&](auto x) { return createCollectionJoinTranslator(x); })
      .Default([](auto x) { assert(false&&"should not happen"); return std::unique_ptr<Translator>(); });
}