#ifndef NOVA_DIALECT_VECTOREXT_OPS
#define NOVA_DIALECT_VECTOREXT_OPS

include "mlir/Interfaces/VectorInterfaces.td"
include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtBase.td"
include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtAttrs.td"
include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class NovaVectorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<NovaVectorExt_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Layout ops.
//===----------------------------------------------------------------------===//

def NovaVectorExt_ToLayoutOp : NovaVectorExt_PureOp<"to_layout", [
  Pure,
  AllTypesMatch<["input", "output"]>
  ]> {
  let summary = [{Layout conversion operator.}];
  let description = [{
    The layout conversion operator takes a shaped value and a layout and
    transforms the value to have that layout.

    This is a pure passthrough that carries the NestedLayoutAttr as an anchor
    for VectorDistribute to consume. If shared_memory_conversion is set, the
    layout change must be materialized through shared memory.
  }];
  let arguments = (ins
    AnyShaped:$input,
    VectorLayoutInterface:$layout,
    DefaultValuedAttr<UnitAttr, "false">:$shared_memory_conversion,
    OptionalAttr<AnyAttr>:$mma_kind
  );
  let results = (outs
    AnyShaped:$output
  );
  let builders = [
    OpBuilder<(ins "Value":$input,
                   "VectorLayoutInterface":$layout,
                   "Attribute":$mma_kind_attr,
                   CArg<"bool", "false">:$shared_memory_conversion), [{
      UnitAttr defaultSharedMemoryConversion;
      if (shared_memory_conversion) {
        defaultSharedMemoryConversion = UnitAttr::get(input.getContext());
      }
      build($_builder, $_state, input.getType(), input, layout,
            defaultSharedMemoryConversion, mma_kind_attr);
    }]>,
    OpBuilder<(ins "Value":$input,
                   "VectorLayoutInterface":$layout), [{
      UnitAttr defaultSharedMemoryConversion;
      Attribute emptyIntrinsic;
      build($_builder, $_state, input.getType(), input, layout,
            defaultSharedMemoryConversion, emptyIntrinsic);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool hasTensorSemantics() {
      return isa<RankedTensorType>(getOutput().getType());
    }
  }];
  let assemblyFormat = "$input `to` `layout` `(` $layout `)` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def NovaVectorExt_ToSIMDOp : NovaVectorExt_PureOp<"to_simd",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = [{SIMT to SIMD conversion operation.}];
  let description = [{
    Temporary operation for source/target materializations when doing type
    conversions between distributed and non-distributed vectors.
  }];
  let arguments = (ins AnyVectorOfAnyRank:$input);
  let results = (outs AnyVectorOfAnyRank:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def NovaVectorExt_ToSIMTOp : NovaVectorExt_PureOp<"to_simt",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = [{SIMD to SIMT conversion operation.}];
  let description = [{
    Temporary operation for source/target materializations when doing type
    conversions between distributed and non-distributed vectors.
  }];
  let arguments = (ins AnyVectorOfAnyRank:$input);
  let results = (outs AnyVectorOfAnyRank:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

#endif  // NOVA_DIALECT_VECTOREXT_OPS
