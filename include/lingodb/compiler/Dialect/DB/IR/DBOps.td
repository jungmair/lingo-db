#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "lingodb/compiler/Dialect/DB/IR/DBInterfaces.td"
include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgInterfaces.td"
include "lingodb/compiler/Dialect/Arrow/IR/ArrowBase.td"

//===----------------------------------------------------------------------===//
// DB dialect definition.
//===----------------------------------------------------------------------===//

def DB_Dialect : Dialect {
    let summary = "Imperative Database Dialect";
    let description = [{
        Imperative database dialect, defining abstract/non-trivial types and high-level operations which more closely resemble SQL semantics, e.g. by supporting `NULL` values.

        Supported types are integers (`i32`/`i64`), floats, bools (`i1`), `Date`, `Timestamp`, `Interval`, `Decimal`, `Char` (fixed-size buffers), and `String` (variable-length buffers). Nullable types are supported using `nullable<...>`; plain `NULL` values can be represented using `nullable<none>`.
    }];

    let name = "db";
    let cppNamespace = "::lingodb::compiler::dialect::db";
    let extraClassDeclaration = [{
        private:
        void registerTypes();
        std::shared_ptr<lingodb::compiler::dialect::db::RuntimeFunctionRegistry> runtimeFunctionRegistry;

        public:
            std::shared_ptr<lingodb::compiler::dialect::db::RuntimeFunctionRegistry> getRuntimeFunctionRegistry(){
                return runtimeFunctionRegistry;
            }
    }];
    let useDefaultTypePrinterParser=1;
      let hasConstantMaterializer = 1;

}

class DB_Op<string mnemonic, list<Trait> traits = []> : Op<DB_Dialect, mnemonic, traits>{
}


class DB_Type<string name, string typeMnemonic> : TypeDef<DB_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def NullableType : DB_Type<"Nullable","nullable"> {
    let summary = "nullable";

    let parameters = (ins "mlir::Type":$type);
    let assemblyFormat = " `<` $type `>`";

    let builders = [
        TypeBuilderWithInferredContext<(ins "mlir::Type":$type), [{
            return $_get(type.getContext(), type);
        }]>
    ];
    // XXX: remove builder that takes context parameter
    // let skipDefaultBuilders = 1;
}


def DB_CharType : DB_Type<"Char","char"> {
    let summary = "char";
    let description = [{
        Fixed-size array of chars.
    }];

    let parameters = (ins "unsigned":$len);
    let assemblyFormat="`<` $len `>` ";
}

def DB_DateUnit_DAY  : I64EnumAttrCase<"day", 0>;
def DB_DateUnit_MS : I64EnumAttrCase<"millisecond", 1>;
def DB_DateUnitAttr : I64EnumAttr<"DateUnitAttr", "",
        [DB_DateUnit_DAY,DB_DateUnit_MS]> {
    let cppNamespace = "::lingodb::compiler::dialect::db";
}
def DB_DateType : DB_Type<"Date","date"> {
    let summary = "date";
    let parameters = (ins "DateUnitAttr":$unit);
    let assemblyFormat="`<` $unit  `>` ";
}

def DB_IntervalUnit_MONTH  : I64EnumAttrCase<"months", 0>;
def DB_IntervalUnit_DAYTIME : I64EnumAttrCase<"daytime", 1>;
def DB_IntervalUnitAttr : I64EnumAttr<"IntervalUnitAttr", "",
        [DB_IntervalUnit_MONTH,DB_IntervalUnit_DAYTIME]> {
    let cppNamespace = "::lingodb::compiler::dialect::db";
}
def DB_IntervalType : DB_Type<"Interval","interval"> {
    let summary = "interval";
    let parameters = (ins "IntervalUnitAttr":$unit);
    let assemblyFormat="`<` $unit  `>` ";
}

def DB_TimeUnit_S : I64EnumAttrCase<"second", 0>;
def DB_TimeUnit_MS : I64EnumAttrCase<"millisecond", 1>;
def DB_TimeUnit_MUS : I64EnumAttrCase<"microsecond", 2>;
def DB_TimeUnit_NS : I64EnumAttrCase<"nanosecond", 3>;
def DB_TimeUnitAttr : I64EnumAttr<"TimeUnitAttr", "",
        [DB_TimeUnit_S,DB_TimeUnit_MS,DB_TimeUnit_MUS,DB_TimeUnit_NS]> {
    let cppNamespace = "::lingodb::compiler::dialect::db";
}
def DB_TimestampType : DB_Type<"Timestamp","timestamp"> {
    let summary = "timestamp";
    let parameters = (ins "TimeUnitAttr":$unit);
    let assemblyFormat="`<` $unit`>` ";
}

def DB_DecimalType : DB_Type<"Decimal","decimal"> {
    let summary = "decimal";
    let description = [{
        Decimal type with `p` digits in total and `s` digits after the decimal point.
    }];

    let parameters = (ins "int":$p,"int":$s);
    let assemblyFormat="`<` $p `,` $s  `>` ";
}

def DB_StringType : DB_Type<"String","string"> {
    let summary = "string";
    let description = [{
        Variable-length string.
    }];
}

def AnyBool: AnyTypeOf<[I1,NullableType]>;
def NullableDBType: AnyTypeOf<[AnyType,NullableType]>;

def DB_ConstantOp : DB_Op<"constant", [ConstantLike,Pure]> {
    let summary = "Create constant value";

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "`(` $value `)` `:` type($result) attr-dict";
    let hasFolder = 1;
}

def DB_NullOp : DB_Op<"null", [Pure]> {
    let summary = "Create null value";

    let results = (outs NullableType:$res);
    let assemblyFormat = "`:` type($res) attr-dict";
}

def DB_DeriveTruth : DB_Op<"derive_truth", [Pure]> {
    let summary = "Get whether a value is true-ish";

    let arguments  = (ins AnyType:$val);
    let results = (outs I1:$res);
    let assemblyFormat = "$val `:` type($val) attr-dict";
}

def DB_SortCompare : DB_Op<"sort_compare", [Pure]> {
    let summary = "Get whether a value is true-ish";

    let arguments  = (ins AnyType:$left, AnyType: $right);
    let results = (outs I8:$res);
    let assemblyFormat = "$left `:` type($left) `,` $right `:` type($right) attr-dict";
}


def DB_AsNullableOp : DB_Op<"as_nullable", [Pure]> {
    let summary = "Convert to nullable type";

    let arguments = (ins AnyType: $val,Optional<I1>: $null);
    let results = (outs NullableType : $res);
    let assemblyFormat = "$val `:` type($val) ( `,` $null^ )?  `->` type($res) attr-dict";
}

def DB_IsNullOp : DB_Op<"isnull", [Pure]> {
    let summary = "Indicate whether a nullable value is null";

    let arguments = (ins NullableType:$val);
    let results = (outs I1:$result);
    let assemblyFormat = "$val `:` type($val) attr-dict";
     let hasFolder = 1;
}

def DB_NullableGetVal : DB_Op<"nullable_get_val", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Get value of nullable type";

    let arguments = (ins NullableType: $val);
    let results = (outs AnyType : $res);
    let assemblyFormat = "$val `:` type($val)  attr-dict";
    let extraClassDefinition = [{
        LogicalResult $cppClass::inferReturnTypes(
                MLIRContext* context, std::optional<Location> location,
                ValueRange operands, DictionaryAttr attributes, OpaqueProperties opaqueProperties,
                RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
            inferredReturnTypes.assign({operands[0].getType().cast<lingodb::compiler::dialect::db::NullableType>().getType()});
            return success();
        }
    }];
         let hasFolder = 1;

}

def DB_RuntimeCall : DB_Op<"runtime_call", [DB_MayNeedNullWrap,DB_MaySupportInvalidValues]> {
    let summary = "Call runtime function";

    let arguments = (ins StrAttr:$fn,Variadic<AnyType>: $args);
    let results = (outs Optional<AnyType>:$res);
    let assemblyFormat = "$fn  `(` $args `)` `:`  functional-type($args,$res)  attr-dict";

    let hasVerifier=1;
    let hasFolder = 1;
}

def DB_Hash : DB_Op<"hash", [Pure]> {
    let summary = "Compute hash";

    let arguments = (ins AnyType: $val);
    let results = (outs  Index : $hash);
    let assemblyFormat = "$val `:` type($val) attr-dict";
}

class DB_BinaryOp<string mnemonic, list<Trait> traits = []> :
        Op<DB_Dialect, mnemonic,
            !listconcat(traits, [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,DB_NeedsNullWrap])>,Arguments<(ins NullableDBType:$left, NullableDBType:$right)> {
    let results = (outs AnyType:$result);
    let extraClassDefinition = [{
        LogicalResult $cppClass::inferReturnTypes(
        MLIRContext* context, std::optional<Location> location, ValueRange operands,
        DictionaryAttr attributes,OpaqueProperties opaqueProperties, RegionRange regions,
        SmallVectorImpl<Type>& inferredReturnTypes) {
            return inferReturnType(context,location,operands,inferredReturnTypes);
        }
    }];
    let assemblyFormat = "$left `:` type($left) `,` $right `:` type($right) attr-dict";
}

def DB_AddOp : DB_BinaryOp<"add",[DB_SupportsInvalidValues]> {
    let summary = "Add int/float/decimal";
    let hasFolder = 1;
}
def DB_SubOp : DB_BinaryOp<"sub",[DB_SupportsInvalidValues]> {
    let summary = "Subtract int/float/decimal";
    let hasFolder = 1;
}
def DB_MulOp : DB_BinaryOp<"mul",[DB_SupportsInvalidValues]> {
    let summary = "Multiply int/float/decimal";

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, std::optional<Location> location, ValueRange operands,
   DictionaryAttr attributes,OpaqueProperties opaqueProperties, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
         return inferMulReturnType(context,location,operands,inferredReturnTypes);
    }
  }];
}
def DB_DivOp : DB_BinaryOp<"div"> {
    let summary = "Divide int/float/decimal";

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, std::optional<Location> location, ValueRange operands,
   DictionaryAttr attributes,OpaqueProperties opaqueProperties, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
         return inferDivReturnType(context,location,operands,inferredReturnTypes);
    }
  }];
}
def DB_ModOp : DB_BinaryOp<"mod"> {
    let summary = "Remainder of int/float/decimal";

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, std::optional<Location> location, ValueRange operands,
   DictionaryAttr attributes,OpaqueProperties opaqueProperties, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
         return inferRemReturnType(context,location,operands,inferredReturnTypes);
    }
  }];
}

class DB_LogicalOp<string mnemonic, list<Trait> traits = []> :
        Op<DB_Dialect, mnemonic,
            !listconcat(traits, [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>])> {
    let results = (outs AnyBool:$res);
    let extraClassDefinition = [{
        LogicalResult $cppClass::inferReturnTypes(MLIRContext* context, std::optional<Location>, ValueRange operands,
                DictionaryAttr,OpaqueProperties opaqueProperties, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
            mlir::Type i1 = IntegerType::get(context, 1);
            inferredReturnTypes.assign({wrapNullableType(context, i1, operands)});
            return success();
        }
    }];
}

def DB_CMP_P_EQ  : I64EnumAttrCase<"eq", 0>;
def DB_CMP_P_NEQ  : I64EnumAttrCase<"neq", 1>;
def DB_CMP_P_LT : I64EnumAttrCase<"lt", 2>;
def DB_CMP_P_LTE : I64EnumAttrCase<"lte", 3>;
def DB_CMP_P_GT : I64EnumAttrCase<"gt", 4>;
def DB_CMP_P_GTE : I64EnumAttrCase<"gte", 5>;
def DB_CMP_P_ISA  : I64EnumAttrCase<"isa", 6>;

def DB_CmpPredicateAttr : I64EnumAttr<"DBCmpPredicate", "",
        [DB_CMP_P_EQ, DB_CMP_P_NEQ, DB_CMP_P_LT, DB_CMP_P_LTE, DB_CMP_P_GT,DB_CMP_P_GTE,DB_CMP_P_ISA]> {
    let cppNamespace = "::lingodb::compiler::dialect::db";
}

def DB_CmpOp : DB_LogicalOp<"compare",
        [DeclareOpInterfaceMethods<Relalg_CmpOpInterface>,DB_NeedsNullWrap,DB_MaySupportInvalidValues]> {
    let summary = "Compare int/Decimal/String";

    let arguments = (ins DB_CmpPredicateAttr:$predicate, NullableDBType:$left, NullableDBType:$right);
    let assemblyFormat = "$predicate $left `:` type($left) `,` $right `:` type($right) attr-dict";
        let extraClassDefinition = [{
            LogicalResult $cppClass::inferReturnTypes(MLIRContext* context, std::optional<Location>, ValueRange operands,
                    DictionaryAttr attributes,OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
                mlir::Type i1 = IntegerType::get(context, 1);
                  CmpOpAdaptor adaptor(operands, attributes, properties, regions);

                if(adaptor.getPredicate()==lingodb::compiler::dialect::db::DBCmpPredicate::isa){
                   inferredReturnTypes.assign({i1});
                   return success();
                }
                inferredReturnTypes.assign({wrapNullableType(context, i1, operands)});
                return success();
            }
        }];
}

def DB_BetweenOp : DB_LogicalOp<"between",[DB_NeedsNullWrap]> {
    let summary = "whether value is between `lower` and `upper`";
    let description = [{
        Determine whether a value is in a given range. This is preferred over (and also internally derived from) two comparisons to simplify optimizations depending on information about a value range.
    }];

    let arguments = (ins NullableDBType:$val, NullableDBType:$lower, NullableDBType:$upper, I1Attr:$lowerInclusive, I1Attr:$upperInclusive);
    let assemblyFormat = "$val `:` type ($val) `between` $lower `:` type($lower) `,` $upper `:` type($upper) `,` `lowerInclusive` `:` $lowerInclusive `,` `upperInclusive` `:` $upperInclusive attr-dict";
}

def DB_OneOfOp : DB_LogicalOp<"oneof",[DB_NeedsNullWrap]> {
    let summary = "whether `val` is included in `vals`";
    let description = [{
        Determine whether `val` is included in `vals`; similar to `IN` in SQL.
    }];

    let arguments = (ins NullableDBType:$val,Variadic<NullableDBType>:$vals );
    let assemblyFormat = "$val `:` type ($val) `?` $vals `:` type ($vals) attr-dict";
}

def AndOp : DB_LogicalOp<"and"> {
    let summary = "Logical And";

    let arguments = (ins Variadic<AnyBool>: $vals);
    let hasCanonicalizeMethod = 1;
    let assemblyFormat = "$vals `:` type($vals) attr-dict";
}

def OrOp : DB_LogicalOp<"or"> {
    let summary = "Logical Or";

    let arguments = (ins Variadic<AnyType>: $vals);
    let hasCanonicalizeMethod = 1;
    let assemblyFormat = "$vals `:` type($vals) attr-dict";
}

def NotOp : DB_Op<"not", [Pure,SameOperandsAndResultType,DB_NeedsNullWrap,DB_SupportsInvalidValues]> {
    let summary = "Logical Not";

    let arguments = (ins AnyBool: $val);
    let results = (outs AnyBool : $res);
    let assemblyFormat = "$val `:` type($val) attr-dict";
}

def CastOp : DB_Op<"cast",[DB_NeedsNullWrap,DB_MaySupportInvalidValues,Pure]> {
    let summary = "Cast value to different type";

    let arguments = (ins NullableDBType:$val);
    let results = (outs NullableDBType:$res);
    let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
    let hasFolder = 1;
}
def LoadArrowOp : DB_Op<"arrow.load"> {
    let summary = "loads value from arrow array";

    let arguments = (ins Arrow_Array:$array, Index: $offset);
    let results = (outs NullableDBType:$res);
    let assemblyFormat = "$array `,` $offset `->` type($res) attr-dict";
}

def AppendArrowOp : DB_Op<"arrow.append"> {
    let summary = "append value to arrow array ";
    let arguments = (ins Arrow_ArrayBuilder:$builder, NullableDBType:$value); //todo: add valid flag, or is nullable type enough?
    let assemblyFormat = "$builder `,` $value `->` type($value) attr-dict";
}

#endif// OPS
