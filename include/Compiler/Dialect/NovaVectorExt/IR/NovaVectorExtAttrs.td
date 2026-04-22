#ifndef NOVA_DIALECT_VECTOREXT_ATTRS
#define NOVA_DIALECT_VECTOREXT_ATTRS

include "Compiler/Dialect/NovaVectorExt/IR/NovaVectorExtBase.td"

//===---------------------------------------------------------------------===//
// Vector layout attributes
//===---------------------------------------------------------------------===//

def NestedLayoutAttr : NovaVectorExt_Attr<"NestedLayout",
      [ DeclareAttrInterfaceMethods<VectorLayoutInterface> ]> {
  let mnemonic = "nested_layout";
  let summary = [{A layout representing a mapping from GPU thread hierarchy to a shape.}];
  let description = [{
    This layout explicitly defines how the shape of the associated vector
    is mapped to a compute hierarchy with four levels:

    1. Subgroups per workgroup
    2. Batches per subgroup  (instruction unrolling)
    3. Outers per batch      (thread layout duplication)
    4. Threads per subgroup
    5. Elements per thread   (native intrinsic tile)

    A vector<n x f16> is conceptually viewed as
    vector<subgroup x batch x outer x thread x element>.
    Thread/subgroup indexing uses virtual IDs based on strides:

      virtual_subgroup_id[i] = (subgroup_id / subgroup_stride[i]) % subgroup_tile[i]
      virtual_thread_id[i]   = (thread_id   / thread_stride[i])   % thread_tile[i]

    Stride 0 means that dimension is not distributed on that hierarchy.
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"int64_t", "subgroup_tile">:$subgroupTile,
    OptionalArrayRefParameter<"int64_t", "batch_tile">:$batchTile,
    OptionalArrayRefParameter<"int64_t", "outer_tile">:$outerTile,
    OptionalArrayRefParameter<"int64_t", "thread_tile">:$threadTile,
    OptionalArrayRefParameter<"int64_t", "element_tile">:$elementTile,

    OptionalArrayRefParameter<"int64_t", "subgroup_strides">:$subgroupStrides,
    OptionalArrayRefParameter<"int64_t", "thread_strides">:$threadStrides
  );

  let assemblyFormat = [{
    `<` `subgroup_tile`     `=` `[` (`]`) : ($subgroupTile^ `]`)? `,`
        `batch_tile`        `=` `[` (`]`) : ($batchTile^ `]`)? `,`
        `outer_tile`        `=` `[` (`]`) : ($outerTile^ `]`)? `,`
        `thread_tile`       `=` `[` (`]`) : ($threadTile^ `]`)? `,`
        `element_tile`      `=` `[` (`]`) : ($elementTile^ `]`)? `,`
        `subgroup_strides`  `=` `[` (`]`) : ($subgroupStrides^ `]`)? `,`
        `thread_strides`    `=` `[` (`]`) : ($threadStrides^ `]`)?
    `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$subgroupTile,
                     "ArrayRef<int64_t>":$batchTile,
                     "ArrayRef<int64_t>":$outerTile,
                     "ArrayRef<int64_t>":$threadTile,
                     "ArrayRef<int64_t>":$elementTile,
                     "ArrayRef<int64_t>":$subgroupStrides,
                     "ArrayRef<int64_t>":$threadStrides)>,
    AttrBuilder<(ins "NestedLayoutAttr":$source,
                     "ArrayRef<int64_t>":$appendSubGroupLens,
                     "ArrayRef<int64_t>":$appendBatchLens,
                     "ArrayRef<int64_t>":$appendOuterLens,
                     "ArrayRef<int64_t>":$appendThreadLens,
                     "ArrayRef<int64_t>":$appendElementLens,
                     "ArrayRef<int64_t>":$appendSubgroupStrides,
                     "ArrayRef<int64_t>":$appendThreadStrides)>
  ];

  let extraClassDeclaration = [{
    // Delinearize a flat thread ID into virtual subgroup/thread ids.
    // Returns empty on failure.
    SmallVector<Value> computeThreadIds(Value threadId, int64_t subgroupSize,
                                        RewriterBase &rewriter) const;

    // Get the undistributed shape: subgroup x batch x outer x thread x element
    SmallVector<int64_t> getUndistributedPackedShape() const;

    // Get the packed shape for a single undistributed dim
    SmallVector<int64_t> getPackedShapeForUndistributedDim(int64_t dim) const;

    // Get the distributed shape with same rank as undistributed shape
    SmallVector<int64_t> getDistributedUnpackedShape() const;
  }];

  let genVerifyDecl = 1;
}

#endif // NOVA_DIALECT_VECTOREXT_ATTRS
