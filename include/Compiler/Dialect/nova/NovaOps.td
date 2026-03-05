#ifndef NOVA_OPS
#define NOVA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "NovaDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/BuiltinTypes.td" 

//traits
def NoComplexOperands : NativeOpTrait<"NoComplexOperands">;
def NoIntegerOperands : NativeOpTrait<"NoIntegerOperands">;

 //attributes
 def Nova_DeviceAttr : AttrDef<Nova_Dialect, "NovaDevice"> {
  let mnemonic = "device"; // This makes it #nova.device<...>
  
  // Parameters for the attribute (e.g., "gpu", "cpu")
  let parameters = (ins "StringAttr":$value);

  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// GPU Lowering Config Attribute
//===----------------------------------------------------------------------===//

// Nova_LoweringConfigAttr — carries tiling levels, MMA kind, promoted operands.
// Stored as a DictionaryAttr with well-known keys:
//   "workgroup"         : I64ArrayAttr  — workgroup tile sizes per loop dim
//   "reduction"         : I64ArrayAttr  — reduction tile sizes per loop dim
//   "mma_kind"          : I32Attr       — NVMMAIntrinsic enum value
//   "promoted_operands" : I64ArrayAttr  — operand indices to promote to smem
//
// Example in IR:
//   {lowering_config = #nova.lowering_config<{
//       workgroup = [128, 128, 0], reduction = [0, 0, 32],
//       mma_kind = 1 : i32, promoted_operands = [0, 1]}>}
def Nova_LoweringConfigAttr : AttrDef<Nova_Dialect, "LoweringConfig"> {
  let mnemonic = "lowering_config";
  let summary = "Drive GPU lowering of an op (tile sizes, MMA kind, promotion).";
  let description = [{
    GPU-specific lowering configuration attached to linalg ops.
    Mirrors IREE's IREEGPU_LoweringConfigAttr but self-contained in Nova.
    The `attributes` dictionary holds all configuration fields.
  }];
  let parameters = (ins
    AttrParameter<"DictionaryAttr",
        "Configured fields (tiling levels, mma_kind, promoted_operands, ...)">:$attributes
  );
  let assemblyFormat = "`<` $attributes `>`";
}

//===----------------------------------------------------------------------===//
// NVIDIA MMA Intrinsic Kind Enum
//===----------------------------------------------------------------------===//

// Identifies a concrete NVIDIA hardware MMA intrinsic.
// Used as the value of the "mma_kind" key in Nova_LoweringConfigAttr.
def Nova_NVMMAIntrinsicKind : I32EnumAttr<"NVMMAIntrinsic",
    "NVIDIA hardware MMA intrinsic", [
  // No MMA — fall back to SIMT / vectorization pipeline.
  I32EnumAttrCase<"NONE",               0, "none">,
  // Volta/Turing WMMA: C(16x16) += A(16x16) * B(16x16), f16 inputs, f32 acc.
  I32EnumAttrCase<"WMMA_F32_16x16x16",  1, "wmma_f32_16x16x16">,
  // Turing WMMA: C(16x16) += A(16x16) * B(16x16), f16 inputs, f16 acc.
  I32EnumAttrCase<"WMMA_F16_16x16x16",  2, "wmma_f16_16x16x16">,
  // Ampere mma.sync: C(16x8) += A(16x16) * B(16x8), f16 inputs, f32 acc.
  I32EnumAttrCase<"MMA_SYNC_F16_16x8x16", 3, "mma_sync_f16_16x8x16">,
  // Ampere mma.sync: C(16x8) += A(16x16) * B(16x8), bf16 inputs, f32 acc.
  I32EnumAttrCase<"MMA_SYNC_BF16_16x8x16", 4, "mma_sync_bf16_16x8x16">,
]> {
  let cppNamespace = "::mlir::nova";
}

// data types
def I4 : Type<CPred<"$_self.isInteger(4)">, "4-bit integer">,
         BuildableType<"$_builder.getIntegerType(4)">;

def Nova_ElementType : AnyTypeOf<[
  I1,I4, I8, I16, I32, I64, 
  F16, BF16, F32, F64,
  Complex<F64>,Complex<F32>]>;

def Nova_FloatType:AnyTypeOf<[F16, BF16, F32, F64]>;
def Nova_IntegerType:AnyTypeOf<[I1,I4, I8, I16, I32, I64]>;
def Nova_ComplexType:AnyTypeOf<[Complex<F64>,Complex<F32>]>;

def Nova_Tensor : RankedTensorOf<[Nova_ElementType]>;
def Nova_FloatTensor : RankedTensorOf<[Nova_FloatType]>;
def Nova_IntegarTensor : RankedTensorOf<[Nova_IntegerType]>;
def Nova_ComplexTensor:RankedTensorOf<[Nova_ComplexType]>;


  // operations start
//enum 
def Nova_Comparisonenum:I32EnumAttr<"ComparisonType","How operands should be compared",[
  I32EnumAttrCase<"EQ",0,"eq">,
  I32EnumAttrCase<"NEQ",1,"neq">,
  I32EnumAttrCase<"LT",2,"lt">,
  I32EnumAttrCase<"GT",3,"gt">,
  I32EnumAttrCase<"LE",4,"le">,
  I32EnumAttrCase<"GE",5,"ge">
  ]>{
    let cppNamespace="::mlir::nova";
  }


//comparison defintion
def Nova_CompareOp:Op<Nova_Dialect,"compare",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary="compare operation";

  let description="elementwise comparison";
  let arguments =(ins
  Nova_Comparisonenum:$kind,
  Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results =(outs Nova_Tensor:$result);
  let assemblyFormat=[{
    `<`$kind `>`$lhs`,`$rhs attr-dict `:`type($lhs)`,`type($rhs)}];
 let builders=[
   OpBuilder <(ins "Type":$resultType,"Value":$lhs,"Value":$rhs,"ComparisonType":$kind)>
 ];
 let skipDefaultBuilders=1;
  let hasVerifier=1;
  let hasCanonicalizer=1; 
  let extraClassDeclaration = [{
  }];

}
def Nova_ToDeviceOp : Op<Nova_Dialect, "to_device", [Pure]> {
  let hasFolder = 1;
  let summary = "Explicitly copy a tensor to a different device";
  let description = [{
    The `nova.to_device` operation copies a tensor from one device to another.
    The source and destination device attributes must be different.
    Example:
    ```mlir
    %0 = nova.to_device %input : tensor<8x8xf32, #nova.device<"0">> -> tensor<8x8xf32, #nova.device<"1">>
    ```
  }];

  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}



def Nova_EmptyTensorOp : Op<Nova_Dialect,"empty_tensor", []> {
  let summary = "allocate an uninitialized tensor with a specified shape";
  let description = [{
    The `nova.empty_tensor` operation allocates a tensor with the specified shape 
    and element type. The memory is uninitialized.
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "`(`$dynamicSizes`)` attr-dict `:` type($result)";
}
//---------------------------------------reduction operations---------------------------------------------------------------------------------------

def Nova_ReductionKind : I32EnumAttr<"ReductionKind",
    "Reduction operation kind", [
  I32EnumAttrCase<"MAX", 0, "max">,
  I32EnumAttrCase<"MIN", 1, "min">,
  I32EnumAttrCase<"SUM", 2, "sum">,
  I32EnumAttrCase<"PRODUCT", 3, "product">,
  I32EnumAttrCase<"MEAN", 8, "mean">,
  I32EnumAttrCase<"ALL", 7, "all">,
  I32EnumAttrCase<"ANY", 4, "any">

]> 
{
  let cppNamespace = "::mlir::nova";
}
def Nova_ReduceOp : Op<Nova_Dialect,"reduce", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let summary = "Generic reduction operation";
  let description = [{
    Performs reduction operation across specified dimensions of a tensor.
    The reduction kind is specified by the 'kind' attribute.
    
    Examples:
    ```mlir
    // Reduce all dimensions with max
    %result = nova.reduce<max> %input : tensor<4x8xf32> -> tensor<f32>
    
    // Reduce along specific dimension with sum
    %result = nova.reduce<sum> %input dimension = [0] : tensor<4x8xf32> -> tensor<8xf32>
    
    // Keep reduced dimensions with min
    %result = nova.reduce<min> %input dimension = [1] keepdims = true ignore_nan=true :  
              tensor<4x8xf32> -> tensor<4x1xf32>
    
    // Product reduction
    %result = nova.reduce<product> %input : tensor<4x8xf32> -> tensor<f32>
    ```}];
      let extraClassDeclaration = [{
  }];
    let hasCanonicalizer = 1;

  let arguments = (ins 
    Nova_ReductionKind:$kind,
    Nova_Tensor:$input,
    OptionalAttr<I64ArrayAttr>:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    DefaultValuedAttr<BoolAttr, "false">:$ignore_nan
  );

    let results = (outs Nova_Tensor:$output);
  let builders = [
    OpBuilder<(ins "ReductionKind":$kind, "Value":$input, "Type":$resultType,CArg<"bool","false">:$keepdims,
                   CArg<"ArrayRef<int64_t>","{0}">:$dimension,CArg<"bool","false">:$ignore_nan)>
  ];
 let skipDefaultBuilders=1;
  let hasVerifier=1;
  let assemblyFormat = [{
  `<` $kind `>` $input (`dimension` `=` $dimension^)? (`keepdims` `=` $keepdims^)? (`ignore_nan` `=`$ignore_nan^)?
    attr-dict `:` type($input) 
  }];
}

//---------------------------ARGMAX-------------------------------

def Nova_ArgmaxOp:Op <Nova_Dialect,"argmax",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
    let arguments = (ins 
    Nova_Tensor:$input,
    OptionalAttr<I64Attr>:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    DefaultValuedAttr<BoolAttr, "false">:$ignore_nan

  );
      let results = (outs Nova_Tensor:$output);
 let skipDefaultBuilders=1;
  let builders = [
    OpBuilder<(ins "Value":$input, 
                   "int64_t":$dimension, "bool":$keepdims,"bool":$ignore_nan, "Type":$resultType)>
  ];
        let extraClassDeclaration = [{
  }];
    let assemblyFormat = [{
     $input (`dimension` `=` $dimension^)? (`keepdims` `=` $keepdims^)?(`ignore_nan` `=` $ignore_nan^)?
    attr-dict `:` type($input) 
  }];
}

//------------------------------ARGMIN----------------------------------------

def Nova_ArgMinOp:Op <Nova_Dialect,"argmin",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
    let arguments = (ins 
    Nova_Tensor:$input,
    OptionalAttr<I32Attr>:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    DefaultValuedAttr<BoolAttr, "false">:$ignore_nan
  ); 
      let results = (outs Nova_Tensor:$output);
 let skipDefaultBuilders=1;
  let builders = [
    OpBuilder<(ins "Value":$input, 
                   "int32_t":$dimension, "bool":$keepdims, "bool":$ignore_nan, "Type":$resultType)>
  ];
        let extraClassDeclaration = [{
  }];
    let assemblyFormat = [{
     $input (`dimension` `=` $dimension^)? (`keepdims` `=` $keepdims^)? (`ignore_nan` `=` $ignore_nan^)?
    attr-dict `:` type($input) 
  }];
}
//-----------------------------------------------------------
// Nova broadcasting.
//-----------------------------------------------------------

// TODO : Dont add elementwise int the operation. We will add it in lowering pass

def Nova_BroadcastInDimOp : Op<Nova_Dialect, "broadcast_in_dim", [
    Pure
  ]> {
  let summary = "Broadcast operation";
  let description = [{
    Expands dimensions of input tensor by replicating data.
    
    The `broadcast_dimensions` attribute maps operand dimensions to result dimensions.
    
    Example:
      // Broadcast [3] to [2, 3]
      %result = nova.broadcast_in_dim %input, dims = [1] : (tensor<3xf32>) 
  }];

  let arguments = (ins
    Nova_Tensor:$operand,
    I64ArrayAttr:$broadcast_dimensions
  );
  
  let results = (outs Nova_Tensor:$result);
  
  let assemblyFormat = [{
    $operand `,` `dims` `=` $broadcast_dimensions attr-dict `:`
    `(` type($operand) `)` `->` type($result)
  }];

  let hasVerifier = 1;
}
//-------------------------------------------
//Base class for unary
//------------------------------------
class Nova_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Op<Nova_Dialect,mnemonic, traits # [Pure]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}


def Nova_ConstantOp : Nova_UnaryOp<"constant", [Pure]> {
  let summary = "Constant tensor operation";
  let description = [{
    Creates a constant tensor with a specified value.
    
    Example:
    ```mlir
    %0 = nova.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    ```
  }];
  
  let arguments = (ins ElementsAttr:$value);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
  
  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    /// Set a new value attribute for the constant op.
    void setValue(::mlir::Attribute newValue) {
      (*this)->setAttr("value", newValue);
    }
  }];
}


//-----------ARITHMETIC OPERATIONS-----------------------------

def Nova_AbsOp : Nova_UnaryOp<"abs", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
        let extraClassDeclaration = [{
  }];
}
include "mlir/Interfaces/SideEffectInterfaces.td"



def Nova_SquareOp : Nova_UnaryOp<"square", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]>{
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def Nova_SqrtOp : Nova_UnaryOp<"sqrt", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let extraClassDeclaration = [{
  }];
}

def Nova_RsqrtOp : Nova_UnaryOp<"rsqrt", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let extraClassDeclaration = [{
  }];
}
def Nova_SignOp : Nova_UnaryOp<"sign", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
      let extraClassDeclaration = [{
  }];
}
def Nova_NegOp : Nova_UnaryOp<"neg", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]>{
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  
}

def Nova_ReciprocalOp : Nova_UnaryOp<"reciprocal", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}
//-----------------------------------------------------------------
// EXPONENTS
//-----------------------------------------------------------------
//Exponents base class
class Nova_ExponentsOp<string mnemonic, list<Trait> traits = [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> :
    Op<Nova_Dialect,mnemonic, traits # [Pure]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
let extraClassDeclaration = [{
  }];
}
 def Nova_ExpOp : Nova_ExponentsOp<"exp"> {
    let hasCanonicalizer = 1;
  } 
 def Nova_Exp2Op : Nova_ExponentsOp<"exp2"> {} 
 def Nova_LogOp : Nova_ExponentsOp<"log"> {
    let hasCanonicalizer = 1;
  } 
 def Nova_Log2Op : Nova_ExponentsOp<"log2"> {} 
 def Nova_Log10Op : Nova_ExponentsOp<"log10"> {} 

//----------------------------------------------------------------------
// TRIGNOMETRY
//----------------------------------------------------------------------
 def Nova_SinOp : Nova_UnaryOp<"sin", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 

 def Nova_CosOp : Nova_UnaryOp<"cos", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 

 def Nova_TanOp : Nova_UnaryOp<"tan", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 
 def Nova_AsinOp : Nova_UnaryOp<"asin", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
}  
def Nova_AcosOp : Nova_UnaryOp<"acos", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 

 def Nova_AtanOp : Nova_UnaryOp<"atan", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 
 def Nova_SinhOp : Nova_UnaryOp<"sinh", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
   
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 

 def Nova_CoshOp : Nova_UnaryOp<"cosh", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 
 def Nova_TanhOp : Nova_UnaryOp<"tanh", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 
 def Nova_AsinhOp : Nova_UnaryOp<"asinh", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 

 def Nova_AcoshOp : Nova_UnaryOp<"acosh", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
   
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 

 def Nova_AtanhOp : Nova_UnaryOp<"atanh", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
 
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
  }];
} 
//--------------------------SPECIAL FUNCTIONS----------------------------

def Nova_ReluOp : Nova_UnaryOp<"relu", [Pure,NoComplexOperands, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Base class for binary elementwise tensor ops
//===----------------------------------------------------------------------===//

class Nova_BinaryOp<string mnemonic, list<Trait> commonTraits = []>: 
  Op<Nova_Dialect, mnemonic,
     commonTraits # [Pure,
                     DeclareOpInterfaceMethods<InferTypeOpInterface>,
                     ]> {

  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
  }];
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//
def Nova_AddOp : Nova_BinaryOp<"add", [Commutative]> {
  let summary = "Elementwise tensor addition";
  let description = [{
    Adds two tensors elementwise with broadcasting support.
    
    Example:
      %r = nova.add %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
  let hasCanonicalizer =1;
}

def Nova_SubOp : Nova_BinaryOp<"sub", []> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise tensor subtraction";
  let description = [{
    Subtracts two tensors elementwise with broadcasting support.

    Example:
      %r = nova.sub %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_MulOp : Nova_BinaryOp<"mul", [Commutative]> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise tensor multiplication";
  let description = [{
    Multiplies two tensors elementwise with broadcasting support.
    Example: [1,2,3] * [3,4,5] = [3,8,15]

    Example:
      %r = nova.mul %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_DivOp : Nova_BinaryOp<"div", []> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise tensor division";
  let description = [{
    Divides two tensors (dividend and divisor) elementwise with broadcasting support.
    
    Example:
      %r = nova.div %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_ModOp : Nova_BinaryOp<"mod", [NoComplexOperands]> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise tensor remainder";
  let description = [{
    Performs element-wise remainder of dividend lhs and divisor rhs.
    Computation example: a=[2,7,10] b=[2,2,4] result=[0,1,2]
  let hasVerifier=1;  
    Example:
      %r = nova.mod %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_PowOp : Nova_BinaryOp<"pow", []> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise tensor power";
  let description = [{
    Performs element-wise power: lhs^rhs.
    Computation example: a=[1,2,3] b=[2,2,4] result=[1,4,81]
  
    Example:
      %r = nova.pow %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_MaxOp : Nova_BinaryOp<"max", [Commutative,NoComplexOperands]> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise tensor maximum";
  let description = [{
    Performs element-wise max operation on tensors.
    Example: a=[1,2,3] b=[2,4,2] result=[2,4,3]
  let hasVerifier=1;  
    Example:
      %r = nova.max %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_MinOp : Nova_BinaryOp<"min", [Commutative,NoComplexOperands]> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise tensor minimum";
  let description = [{
    Performs element-wise min operation on tensors.
    Example: a=[1,2,3] b=[2,4,2] result=[1,2,2]
  let hasVerifier=1;  
    Example:
      %r = nova.min %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_AndOp : Nova_BinaryOp<"and", [Commutative,NoComplexOperands]> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise logical AND";
  let description = [{
    Performs element-wise logical AND operation on tensors.
    Example: a=[true,false,true] b=[true,true,false] result=[true,false,false]
  let hasVerifier=1;  
    Example:
      %r = nova.and %a, %b : tensor<2x3xi1>, tensor<3xi1> -> tensor<2x3xi1>
  }];
}

def Nova_OrOp : Nova_BinaryOp<"or", [Commutative,NoComplexOperands]> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise logical OR";
  let description = [{
    Performs element-wise logical OR operation on tensors.
    Example: a=[true,false,true] b=[true,true,false] result=[true,true,true]
  let hasVerifier=1;  
    Example:
      %r = nova.or %a, %b : tensor<2x3xi1>, tensor<3xi1> -> tensor<2x3xi1>
  }];
}

def Nova_XorOp : Nova_BinaryOp<"xor", [Commutative,NoComplexOperands]> {
  let hasCanonicalizer = 1;
  let summary = "Elementwise logical XOR";
  let description = [{
    Performs element-wise logical XOR operation on tensors.
    Example: a=[true,false,true] b=[true,true,false] result=[false,true,true]
  let hasVerifier=1;  
    Example:
      %r = nova.xor %a, %b : tensor<2x3xi1>, tensor<3xi1> -> tensor<2x3xi1>
  }];
}

def Nova_NotOp : Nova_UnaryOp<"not", [Commutative,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let summary = "Elementwise logical XOR";
  let description = [{
    Performs element-wise logical NOT operation on tensors.
    Example: a=[true,false,true] result=[false,true,false]
  let hasVerifier=1;  
    Example:
      %r = nova.not %a tensor<3xi1> -> tensor<2x3xi1>
  }];
   let extraClassDeclaration = [{
  }];
}

def Nova_MatmulOp : Op<Nova_Dialect,"matmul", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);
  let hasVerifier=1;  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];
}

//-------------------------------------ACTIVATION FUNCTIONS-------------------------------------

def Nova_SigmoidOp : Op<Nova_Dialect,"sigmoid", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$lhs);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $lhs attr-dict `:` type($lhs) }];
}
def Nova_GeluOp : Op<Nova_Dialect,"gelu", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$lhs);
  let results = (outs Nova_Tensor:$result);
   
  let assemblyFormat = [{
    $lhs attr-dict `:` type($lhs) }];
  
}
def Nova_GeluBackwardOp : Op<Nova_Dialect,"gelu_backward", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$grad_out, Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$result);
   
  let assemblyFormat = [{
    $grad_out `,` $input attr-dict `:` type($grad_out) `,` type($input) }];
}
def Nova_SoftmaxOp:Op<Nova_Dialect,"softmax",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]>{
  let arguments=(ins Nova_Tensor:$input,
   OptionalAttr<I32Attr>:$dimension);
  let results =(outs Nova_Tensor:$result);
 

  let assemblyFormat = [{
    $input (`dimension` `=` $dimension^)? attr-dict `:` type($input) }];
  let builders = [
    OpBuilder<(ins "ReductionKind":$kind, "Value":$input, "Type":$resultType,CArg<"bool","false">:$keepdims,
                   CArg<"int32_t","-1">:$dimension,CArg<"bool","false">:$ignore_nan)>
  ];
}




def Nova_TransposeOp : Op<Nova_Dialect,"transpose",[Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input,
      DefaultValuedAttr<I32Attr, "-1">:$axes1,
    DefaultValuedAttr<I32Attr, "-2">:$axes2);
  let results = (outs Nova_Tensor:$result);
  
  let assemblyFormat = [{
     $input (`axes1` `=` $axes1^)?(`axes2` `=` $axes2^)? attr-dict `:`   type($input) 
  }];
  
}
//=========LOSSES===================
def Nova_MaeOp :Op<Nova_Dialect,"mae",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]>{
    let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);
    let summary = "Mean absolute error loss operation";

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];
}
def Nova_MseOp :Op<Nova_Dialect,"mse",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]>{
    let summary = "Mean Square error loss operation";
  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];
}
def Nova_CceOp :Op<Nova_Dialect,"cce",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]>{
  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let summary = "Categorical Cross Entropy loss operation";
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];
}
def Nova_BceOp :Op<Nova_Dialect,"bce",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]>{
  let summary = "Binary Cross Entropy loss operation";
  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];
}
def Nova_SceOp :Op<Nova_Dialect,"sce",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]>{
  let summary = "Sparse Cross Entropy loss operation";
  let description = [{
    Computes the sparse cross entropy loss between logits and target indices.
    This is commonly used for multi-class classification where targets are class indices.
    
    Example:
      %loss = nova.sce %logits, %targets : tensor<4x10xf32>, tensor<4xi32>
  }];
  
  let arguments = (ins Nova_Tensor:$logits, Nova_Tensor:$targets);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $logits `,` $targets attr-dict `:` type($logits) `,` type($targets) 
  }];
}

def Nova_CastOp : Op<Nova_Dialect, "cast", [Pure]> {
  let summary = "Cast operation";
  let description = [{
    Casts a tensor from one element type to another.
    Example:
    ```mlir
    %0 = nova.cast %input : tensor<8xf32> -> tensor<8xi32>
    ```
  }];

  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}
//==============================Weight initialier operations======================
def Nova_Rndm2DOp:Op<Nova_Dialect,"rndm2d",[Pure]>{
  
    let arguments = (ins F64:$min, F64:$max);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $min `,` $max attr-dict `:` type($result) 
  }];
}

def AdamOp : Op<Nova_Dialect,"adam",[Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]>{
  let summary = "Adam optimizer stepping";
  let description = [{
    Implements adam optimizer step 
    Returns updated params and moments 
    Inputs:
     - param: Current parameters
    - m: Current first moment
    - v: Current second moment
    - grad: Gradient with respect to parameters
    - lr: Learning rate
    - t: Time step (for bias correction)

    Attributes:
    - beta1: Exponential decay rate for first moment (default 0.9)
    - beta2: Exponential decay rate for second moment (default 0.999)
    - epsilon: Small constant for numerical stability (default 1e-8)
    
      Returns:
    - new_param: Updated parameters
    - new_m: Updated first moment
    - new_v: Updated second moment

    Example:
    ```mlir
    %np, %nm, %nv = nova.adam %p, %m, %v, %g 
      {beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8, lr = 0.001, t = 1} 
      : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
    ```
  }];
  let arguments = (ins
    Nova_Tensor:$param,
    Nova_Tensor:$m,
    Nova_Tensor:$v,
    Nova_Tensor:$grad,
    
    // Hyperparameters as attributes for now
    DefaultValuedAttr<F32Attr, "0.001f">:$lr,
    DefaultValuedAttr<F32Attr, "0.9f">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999f">:$beta2,
    DefaultValuedAttr<F32Attr, "1.0e-8f">:$epsilon,
    I32Attr:$t
  );
  
  let results = (outs 
    Nova_Tensor:$new_param,
    Nova_Tensor:$new_m,
    Nova_Tensor:$new_v
  );
  
  let assemblyFormat = [{
    $param `,` $m `,` $v `,` $grad attr-dict `:` type($param) `,` type($m) `,` type($v) `,` type($grad) `->` type($new_param) `,` type($new_m) `,` type($new_v)
  }];

  let extraClassDeclaration = [{
  }];
}
def GatherOp : Op<Nova_Dialect, "gather", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Gather Operation";
  let description = [{The nova gather operation takes an input tensor and extracts slices along the specified `axis` using the provided `indices`}];
  let arguments = (ins
    Nova_Tensor:$input,
    Nova_Tensor:$indices,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );
  let results = (outs Nova_Tensor:$output);
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = [{
    $input ` ` `[` $indices `]` (`axis` `=` $axis^)? attr-dict `:` type($input) `,` type($indices) `->` type($output)
  }];
}

def ScatterAddOp : Op<Nova_Dialect, "scatter_add", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ScatterAdd operation";
  let arguments = (ins Nova_Tensor:$input, Nova_Tensor:$indices, Nova_Tensor:$src, DefaultValuedAttr<I64Attr, "0">:$axis);
  let results = (outs Nova_Tensor:$output);

  let hasVerifier = 1;
  let hasFolder = 1;

  let assemblyFormat = [{
    $input `,` $indices `,` $src (`axis` `=` $axis^)? attr-dict `:` type($input) `,` type($indices) `,` type($src) `->` type($output)
  }];
}
def Nova_ReshapeOp : Op<Nova_Dialect, "reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes the input tensor to a new shape.
    Example:
      %result = nova.reshape %input : (tensor<4x4xf32>) -> tensor<2x8xf32>
  }];

  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

//===============================
//MLP OPERATIONS
//===============================
def LayerNormOp :Op<Nova_Dialect,"layer_norm",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary ="Layer norm operation y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta";
  let arguments = (ins Nova_Tensor:$input, Nova_Tensor:$gamma, Nova_Tensor:$beta);
  let results=(outs Nova_Tensor:$output);
  //let hasVerifier=1;
    let assemblyFormat = [{
    $input `,` $gamma `,` $beta attr-dict `:` type($input) `,` type($gamma) `,` type($beta) 
  }];
}
def LayerNormBackwardOp :Op<Nova_Dialect,"layer_norm_backward",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary ="Layer norm backward has calculating gradient for gamma,beta and input";
  let arguments = (ins Nova_Tensor:$grad_y, Nova_Tensor:$x,  Nova_Tensor:$gamma);
  let results=(outs Nova_Tensor:$grad_x, Nova_Tensor:$grad_gamma, Nova_Tensor:$grad_beta);
  //let hasVerifier=1;
    let assemblyFormat = [{
    $grad_y `,` $x `,`  $gamma attr-dict `:` type($grad_y) `,` type($x) `,` type($gamma) 
  }];
}
def LinearOp :Op<Nova_Dialect,"linear",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary ="Linear operation(matmul+add)";
  let arguments = (ins Nova_Tensor:$input, Nova_Tensor:$weight, Nova_Tensor:$bias);
  let results=(outs Nova_Tensor:$output);
  //let hasVerifier=1;
    let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict `:` type($input) `,` type($weight) `,` type($bias) 
  }];
}
def LinearBackwardOp :Op<Nova_Dialect,"linear_backward",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary ="Linear backward operation calculating gradient for input, weight, and bias";
  let arguments = (ins Nova_Tensor:$grad_out, Nova_Tensor:$input, Nova_Tensor:$weight);
  let results=(outs Nova_Tensor:$grad_input, Nova_Tensor:$grad_weight, Nova_Tensor:$grad_bias);
  //let hasVerifier=1;
    let assemblyFormat = [{
    $grad_out `,` $input `,` $weight attr-dict `:` type($grad_out) `,` type($input) `,` type($weight) 
  }];
}
def Nova_SceBackwardOp : Op<Nova_Dialect, "sce_backward",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Sparse Cross Entropy backward (gradient w.r.t. logits)";
  let description = [{
    Computes the gradient of the sparse cross-entropy loss w.r.t. logits:
      grad_logits = (softmax(logits, dim) - one_hot(targets)) / N
    where N = total number of samples (product of non-class dimensions).

    The dim attribute specifies which dimension is the class dimension
    Negative values index from the end.

    logits: tensor<[..., C, ...] x float>
    targets: tensor<[...] x int>           

    Example:
      %grad = nova.sce_backward %logits, %targets {dim = -1}
        : tensor<4x128x50304xf32>, tensor<4x128xi32>
  }];

  let arguments = (ins Nova_Tensor:$logits, Nova_Tensor:$targets,
                       DefaultValuedAttr<I64Attr, "-1">:$dim);
  let results = (outs Nova_Tensor:$grad_logits);

  let assemblyFormat = [{
    $logits `,` $targets attr-dict `:` type($logits) `,` type($targets)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// FusionBarrierOp
//===----------------------------------------------------------------------===//
// A tensor-level scheduling fence that prevents loop fusion analysis from
// merging the global-to-workgroup linalg.copy (Stage 1) with the
// thread-local promoted linalg.copy (Stage 2) into the same loop body.
//
// This op is a pure identity: its result equals its source at runtime.
// It is erased at the very beginning of bufferization once all tiling and
// fusion decisions have been made.
//
// Mirrors IREE's iree_codegen.fusion_barrier (IREECodegenOps.td).
//
// Usage in NovaGPUPromoteMatmulOperands:
//   %alloc  = bufferization.alloc_tensor {workgroup}
//   %stage1 = linalg.copy(%global -> %alloc)   <- fills shared memory
//   %fence  = nova.fusion_barrier %stage1       <- fusion cannot cross here
//   %stage2 = linalg.copy(%fence -> %empty)     <- per-thread promoted copy
//
// Usage at bufferization (NovaGPUBufferize / Passes.cpp):
//   Erase all nova.fusion_barrier by replacing with their source value.

def Nova_FusionBarrierOp : Op<Nova_Dialect, "fusion_barrier", [
    SameOperandsAndResultType, Pure]> {
  let summary = "Prevents fusion through a ranked tensor value";
  let description = [{
    Identity operation on a ranked tensor that acts as an opaque fence for
    the fusion analysis. The fusion/tiling passes cannot see through this op,
    so producer operations on the source side and consumer operations on the
    result side will be placed in separate loops.

    This op must be erased before bufferization. Example erasure:
    ```cpp
    funcOp.walk([&](nova::FusionBarrierOp barrier) {
      rewriter.replaceOp(barrier, barrier.getSource());
    });
    ```
  }];
  let arguments = (ins AnyRankedTensor:$source);
  let results   = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict $source `:` type($result)";
}

#endif // NOVA_OPS

