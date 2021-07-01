#include "mlir/Conversion/RelAlgToDB/HashJoinUtils.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/IR/BlockAndValueMapping.h>

class NLSingleJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::SingleJoinOp joinOp;
   mlir::Value matchFoundFlag;

   public:
   NLSingleJoinLowering(mlir::relalg::SingleJoinOp singleJoinOp) : mlir::relalg::ProducerConsumerNode({singleJoinOp.left(), singleJoinOp.right()}), joinOp(singleJoinOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
         }
      }
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return joinOp.getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
         children[1]->produce(context, builder);
         mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
         mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);
         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::relalg::ProducerConsumerBuilder builder2(ifOp.elseRegion());
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         for (mlir::Attribute attr : joinOp.mapping()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if (this->requiredAttributes.contains(defAttr)) {
               auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);

               context.setValueForAttribute(scope, defAttr, nullValue);
            }
         }
         consumer->consume(this, builder1, context);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, ifOp.getResults());
      } else if (child == this->children[1].get()) {
         mlir::relalg::SingleJoinOp clonedSingleJoinOp = mlir::dyn_cast<mlir::relalg::SingleJoinOp>(joinOp->clone());
         mlir::Block* block = &clonedSingleJoinOp.predicate().getBlocks().front();
         auto* terminator = block->getTerminator();
         mlir::Value condition;
         bool hasCondition = !mlir::cast<mlir::relalg::ReturnOp>(terminator).results().empty();
         if (hasCondition) {
            builder.mergeRelatinalBlock(block, context, scope);
            condition = mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0];
         } else {
            condition = builder.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         }

         auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), condition);
         mlir::Block* ifBlock = new mlir::Block;

         ifOp.thenRegion().push_back(ifBlock);

         mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
         if (!requiredBuilders.empty()) {
            mlir::Block* elseBlock = new mlir::Block;
            ifOp.elseRegion().push_back(elseBlock);
            mlir::relalg::ProducerConsumerBuilder builder2(ifOp.elseRegion());
            builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
         }
         for (mlir::Attribute attr : joinOp.mapping()) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
            auto* defAttr = &relationDefAttr.getRelationalAttribute();
            if (this->requiredAttributes.contains(defAttr)) {
               auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
               auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
               auto value = context.getValueForAttribute(refAttr);
               if (refAttr->type != defAttr->type) {
                  mlir::Value tmp = builder1.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
                  value = tmp;
               }
               context.setValueForAttribute(scope, defAttr, value);
            }
         }
         consumer->consume(this, builder1, context);
         auto trueVal = builder1.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder1.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, trueVal);
         builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

         size_t i = 0;
         for (auto b : requiredBuilders) {
            context.builders[b] = ifOp.getResult(i++);
         }
         terminator->erase();
         clonedSingleJoinOp->destroy();
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLSingleJoinLowering() {}
};
class ConstantSingleJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::SingleJoinOp joinOp;
   mlir::Value matchFoundFlag;
   std::vector<const mlir::relalg::RelationalAttribute*> attrs;
   std::vector<const mlir::relalg::RelationalAttribute*> origAttrs;
   std::vector<mlir::Type> types;
   size_t builderId;

   public:
   ConstantSingleJoinLowering(mlir::relalg::SingleJoinOp singleJoinOp) : mlir::relalg::ProducerConsumerNode({singleJoinOp.left(), singleJoinOp.right()}), joinOp(singleJoinOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
            origAttrs.push_back(refAttr);
            attrs.push_back(defAttr);
            types.push_back(defAttr->type);
         }
      }
      propagateInfo();
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return joinOp.getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         auto unpacked = builder.create<mlir::util::UnPackOp>(joinOp->getLoc(), types, context.builders[builderId]);
         for (size_t i = 0; i < attrs.size(); i++) {
            context.setValueForAttribute(scope, attrs[i], unpacked.getResult(i));
         }
         consumer->consume(this, builder, context);
      } else if (child == this->children[1].get()) {
         std::vector<mlir::Value> values;
         for (size_t i = 0; i < origAttrs.size(); i++) {
            mlir::Value value = context.getValueForAttribute(origAttrs[i]);
            if (origAttrs[i]->type != attrs[i]->type) {
               mlir::Value tmp = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), attrs[i]->type, value);
               value = tmp;
            }
            values.push_back(value);
         }
         context.builders[builderId] = builder.create<mlir::util::PackOp>(joinOp->getLoc(), mlir::TupleType::get(builder.getContext(), types), values);
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      std::vector<mlir::Value> values;
      for (auto type : types) {
         values.push_back(builder.create<mlir::db::NullOp>(joinOp.getLoc(), type));
      }
      builderId = context.getBuilderId();
      context.builders[builderId] = builder.create<mlir::util::PackOp>(joinOp->getLoc(), mlir::TupleType::get(builder.getContext(), types), values);
      children[1]->addRequiredBuilders({builderId});
      children[1]->produce(context, builder);
      children[0]->produce(context, builder);
   }

   virtual ~ConstantSingleJoinLowering() {}
};
class HashSingleJoinLowering : public mlir::relalg::HJNode<mlir::relalg::SingleJoinOp> {
   mlir::Value matchFoundFlag;

   public:
   HashSingleJoinLowering(mlir::relalg::SingleJoinOp innerJoinOp) : mlir::relalg::HJNode<mlir::relalg::SingleJoinOp>(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }
   void addAdditionalRequiredAttributes() override {
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
         }
      }
   }

   virtual void handleLookup(mlir::Value matched, mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), matched);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::relalg::ProducerConsumerBuilder builder3(ifOp.elseRegion());
         builder3.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            auto value = context.getValueForAttribute(refAttr);
            if (refAttr->type != defAttr->type) {
               mlir::Value tmp = builder1.create<mlir::db::CastOp>(builder.getUnknownLoc(), defAttr->type, value);
               value = tmp;
            }
            context.setValueForAttribute(scope, defAttr, value);
         }
      }
      consumer->consume(this, builder1, context);

      auto trueVal = builder1.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
      builder1.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, trueVal);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
   }

   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);
      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::relalg::ProducerConsumerBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      }
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
            context.setValueForAttribute(scope, defAttr, nullValue);
         }
      }
      consumer->consume(this, builder1, context);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
      setRequiredBuilderValues(context, ifOp.getResults());
   }
   virtual ~HashSingleJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredSingleJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::SingleJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashSingleJoinLowering>(joinOp);
         } else if (impl.getValue() == "constant") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<ConstantSingleJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLSingleJoinLowering>(joinOp);
});