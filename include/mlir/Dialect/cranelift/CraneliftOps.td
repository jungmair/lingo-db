#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


def Cranelift_Dialect : Dialect {
let name = "cranelift";
let cppNamespace = "::mlir::cranelift";
let extraClassDeclaration = [{
private:
void registerTypes();
public:
}];
}

class Cranelift_Op<string mnemonic, list<Trait> traits = []> :
Op<Cranelift_Dialect, mnemonic, traits>{

}

class Cranelift_Type<string name, string typeMnemonic> : TypeDef<Cranelift_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def FuncOp : Cranelift_Op<"func", [
  AffineScope, IsolatedFromAbove, Symbol,FunctionOpInterface
]> {

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type);
  let regions = (region AnyRegion:$body);


  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
          /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    FunctionType getFunctionType() { return function_type();}
  }];
}

def ReturnOp : Cranelift_Op<"return", [NoSideEffect, HasParent<"FuncOp">,Terminator]> {


  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def CallOp : Cranelift_Op<"call"> {
  let summary = "call operation";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def BranchOp : Cranelift_Op<"br", [NoSideEffect, Terminator]> {

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

def CondBranchOp : Cranelift_Op<"cond_br",
    [AttrSizedOperandSegments,NoSideEffect, Terminator]> {

  let arguments = (ins I1:$condition,
                       Variadic<AnyType>:$trueDestOperands,
                       Variadic<AnyType>:$falseDestOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

def IConstOp : Cranelift_Op<"iconst"> {
  let arguments = (ins I64Attr:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($res)";
}
def F32ConstOp : Cranelift_Op<"f32const"> {
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$res);
  let assemblyFormat = "attr-dict $value `:` type($res)";
}
def F64ConstOp : Cranelift_Op<"f64const"> {
  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$res);
  let assemblyFormat = "attr-dict $value `:` type($res)";
}
def GlobalOp : Cranelift_Op<"global"> {
  let arguments = (ins StrAttr:$symbol_name,StrAttr:$value);
  let assemblyFormat = "attr-dict $symbol_name `=` $value";
}
def AddressOfOp : Cranelift_Op<"addressof"> {
  let arguments = (ins StrAttr:$symbol_name);
  let results = (outs I64:$ptr);
  let assemblyFormat = "attr-dict $symbol_name";
}
def BConstOp : Cranelift_Op<"bconst"> {
  let arguments = (ins I1Attr:$value);
  let results = (outs I1:$res);
  let assemblyFormat = "attr-dict $value";
}
def SelectOp : Cranelift_Op<"select"> {
  let arguments = (ins I1:$condition,AnyType:$trueVal, AnyType:$falseVal);
  let results = (outs AnyType:$res);
  let assemblyFormat = "attr-dict $condition `?` $trueVal `:` type($trueVal) `,` $falseVal `:` type($falseVal) `->` type($res)";
}


def Cranelift_ICmpPredicateAttr : I64EnumAttr<
    "ICmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"slt", 2>,
      I64EnumAttrCase<"sle", 5>,
      I64EnumAttrCase<"sgt", 4>,
      I64EnumAttrCase<"sge", 3>,
      I64EnumAttrCase<"ult", 6>,
      I64EnumAttrCase<"ule", 7>,
      I64EnumAttrCase<"ugt", 8>,
      I64EnumAttrCase<"uge", 9>,
      I64EnumAttrCase<"of", 10>,
      I64EnumAttrCase<"nof", 11>,

    ]> {
  let cppNamespace = "::mlir::cranelift";
}
def Cranelift_CmpIOp : Cranelift_Op<"icmp"> {
  let arguments = (ins Cranelift_ICmpPredicateAttr:$predicate,
                       SignlessIntegerLike:$lhs,
                       SignlessIntegerLike:$rhs);
  let results = (outs I1:$res);
    let assemblyFormat = "$predicate $lhs `:` type($lhs) `,` $rhs  `:` type($rhs) attr-dict";

}

class Cranelift_ArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Cranelift_Op<mnemonic, traits # [SameOperandsAndResultType]>;
class Cranelift_Arith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Cranelift_ArithmeticOp<mnemonic, traits> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
class Cranelift_Arith_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Cranelift_Arith_BinaryOp<mnemonic, traits>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)>,
    Results<(outs SignlessIntegerLike:$result)>;
def IAddOp : Cranelift_Arith_IntBinaryOp<"iadd">;
def ISubOp : Cranelift_Arith_IntBinaryOp<"isub">;
def IMulOp : Cranelift_Arith_IntBinaryOp<"imul">;
def UDivOp : Cranelift_Arith_IntBinaryOp<"udiv">;
def SDivOp : Cranelift_Arith_IntBinaryOp<"sdiv">;
def URemOp : Cranelift_Arith_IntBinaryOp<"urem">;
def SRemOp : Cranelift_Arith_IntBinaryOp<"srem">;
def IShlOp : Cranelift_Arith_IntBinaryOp<"ishl">;
def SShrOp : Cranelift_Arith_IntBinaryOp<"sshr">;
def UShrOp : Cranelift_Arith_IntBinaryOp<"ushr">;
def BOrOp : Cranelift_Arith_IntBinaryOp<"bor">;
def BXOrOp : Cranelift_Arith_IntBinaryOp<"bxor">;
def BAndOp : Cranelift_Arith_IntBinaryOp<"band">;


def UExtendOp : Cranelift_Op<"uextend"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def SExtendOp : Cranelift_Op<"sextend"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def IReduceOp : Cranelift_Op<"ireduce"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def IConcatOp : Cranelift_Op<"iconcat"> {
  let arguments = (ins SignlessIntegerLike:$lower,SignlessIntegerLike:$higher);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $lower `,` $higher  `:` type($lower) `,` type($higher) `->` type($res)";
}
#endif// OPS
