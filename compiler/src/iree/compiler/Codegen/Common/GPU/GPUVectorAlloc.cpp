// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits>
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPromotionAnalysis.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-alloc"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

static llvm::cl::opt<bool> clEnableDMASwizzle(
    "iree-codegen-enable-dma-swizzle",
    llvm::cl::desc("Enable XOR swizzling for DMA-promoted LDS allocations"),
    llvm::cl::init(true));

namespace {

/// Allocate a workgroup-addressed tensor with the given shape and element type.
static bufferization::AllocTensorOp
allocateWorkgroupTensor(OpBuilder &b, Location loc, ArrayRef<int64_t> shape,
                        Type elementType) {
  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  RankedTensorType tensorType =
      RankedTensorType::get(shape, elementType, sharedMemoryAddrSpace);
  auto allocTensorOp = bufferization::AllocTensorOp::create(
      b, loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);
  return allocTensorOp;
}

/// Allocates a workgroup tensor and writes the vector into it.
static FailureOr<Value> allocateTensorForVector(OpBuilder &b, Location loc,
                                                Value vector) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  auto allocTensorOp = allocateWorkgroupTensor(b, loc, vectorType.getShape(),
                                               vectorType.getElementType());

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied = vector::TransferWriteOp::create(b, loc, vector, allocTensorOp,
                                                 indices, inBounds)
                     .getResult();
  return copied;
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value tensor) {
  Value c0 = arith::ConstantIndexOp::create(b, tensor.getLoc(), 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  return vector::TransferReadOp::create(b, tensor.getLoc(), vectorType, tensor,
                                        indices, /*padding=*/std::nullopt,
                                        inBounds)
      .getResult();
}

/// Promote a vector value through shared memory: write to a workgroup tensor,
/// synchronize with a value barrier, and read back. On success, returns the
/// promoted value and the transfer_write op that consumes the original vector.
static FailureOr<std::pair<Value, Operation *>>
promoteViaSharedMemory(OpBuilder &b, Location loc, Value vector) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  FailureOr<Value> written = allocateTensorForVector(b, loc, vector);
  if (failed(written)) {
    return failure();
  }
  Operation *writeOp = written->getDefiningOp();
  auto synced = IREE::GPU::ValueBarrierOp::create(b, loc, *written);
  Value read = readVectorFromTensor(b, vectorType, synced.getResult(0));
  return std::make_pair(read, writeOp);
}

/// Materialize DMA promotions for transfer_read ops whose results are tagged
/// with UseGlobalLoadDMAAttr by the promotion analysis.
static LogicalResult materializeDMAPromotions(
    FunctionOpInterface funcOp,
    const llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface>
        &layouts) {
  // Run backward promotion type analysis.
  PromotionTypeMap promotionTypes = analyzePromotionTypes(funcOp);

  // Collect transfer_read ops tagged for DMA promotion.
  SmallVector<vector::TransferReadOp> dmaReads;
  funcOp.walk([&](vector::TransferReadOp readOp) {
    auto it = promotionTypes.find(readOp.getResult());
    if (it != promotionTypes.end() &&
        isa<IREE::GPU::UseGlobalLoadDMAAttr>(it->second)) {
      dmaReads.push_back(readOp);
    }
  });

  if (dmaReads.empty()) {
    return success();
  }

  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);

  bool targetSupportsDMA = target && targetSupportsGlobalLoadDMA(target);
  if (!targetSupportsDMA) {
    LDBG() << "Target does not support direct-to-lds, using fallback via "
           << "registers instead";
  }

  OpBuilder builder(funcOp);
  for (vector::TransferReadOp readOp : dmaReads) {
    Location loc = readOp.getLoc();
    VectorType vectorType = readOp.getVectorType();

    // Synchronize before write to shared memory (same hack as the regular LDS
    // path).
    builder.setInsertionPointToStart(readOp->getBlock());
    gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);

    bool eligible = targetSupportsDMA;
    if (eligible) {
      VectorType vectorType = readOp.getVectorType();

      // Total transfer size must be aligned to the minimum DMA transfer size
      // for the full workgroup (all subgroups must participate).
      // TODO: Check if participation of only some subgroups makes sense.
      int64_t totalElements = vectorType.getNumElements();
      int64_t numThreads = llvm::product_of(*getWorkgroupSize(funcOp));
      int64_t elementBits = vectorType.getElementType().getIntOrFloatBitWidth();
      ArrayRef<int64_t> dmaSizes = target.getWgp().getDmaSizes().asArrayRef();
      int64_t minElementsPerWorkgroup = std::numeric_limits<int64_t>::max();
      for (int64_t dmaSize : dmaSizes) {
        if (dmaSize % elementBits != 0) {
          continue;
        }
        int64_t elementsPerLane = dmaSize / elementBits;
        minElementsPerWorkgroup =
            std::min(minElementsPerWorkgroup, numThreads * elementsPerLane);
      }
      if (minElementsPerWorkgroup == std::numeric_limits<int64_t>::max() ||
          totalElements % minElementsPerWorkgroup != 0) {
        LDBG() << "No suitable DMA size available, falling back to shared "
               << "memory promotion";
        eligible = false;
      }
    }
    // Look up the layout for this transfer_read result.
    if (eligible && !layouts.contains(readOp.getResult())) {
      LDBG() << "No layout found for DMA read, falling "
             << "back to shared-memory promotion";
      eligible = false;
    }

    if (!eligible) {
      // Fallback: shared-memory register-roundtrip promotion.
      builder.setInsertionPointAfter(readOp);
      auto promoted = promoteViaSharedMemory(builder, loc, readOp.getResult());
      if (failed(promoted)) {
        return failure();
      }
      auto [replacement, writeOp] = *promoted;
      // Replace all uses of the original read except the transfer_write
      // inside promoteViaSharedMemory (which consumes the read result).
      readOp.getResult().replaceAllUsesExcept(replacement, writeOp);
      continue;
    }

    // DMA path: alloc_tensor -> [swizzle_hint] -> async_dma -> value_barrier
    // -> transfer_read.
    builder.setInsertionPoint(readOp);
    auto layout = layouts.lookup(readOp.getResult());

    // Build permutation_map attribute if the original read has a non-identity
    // permutation map.
    AffineMapAttr permMapAttr = readOp.getPermutationMapAttr();

    auto allocOp = allocateWorkgroupTensor(builder, loc, vectorType.getShape(),
                                           vectorType.getElementType());
    Value dmaDest = allocOp;
    // Try to wrap the alloc in a swizzle_hint for bank conflict avoidance.
    if (clEnableDMASwizzle && target) {
      int64_t totalElements = vectorType.getNumElements();
      int64_t elementBitWidth =
          vectorType.getElementType().getIntOrFloatBitWidth();
      // The innermost element tile dimension determines the contiguous access
      // width per thread during MMA reads from the flat LDS buffer.
      // accessElems must divide this so ResolveSwizzleHints can correctly
      // unroll the distributed loads.
      auto nestedLayout = dyn_cast<IREE::VectorExt::NestedLayoutAttr>(layout);
      int64_t computeAccessWidth =
          nestedLayout ? nestedLayout.getElementTile().back() : 1;
      auto swizzleParams = getXorShuffleParamsForDMA(
          target, elementBitWidth, totalElements, computeAccessWidth);
      if (succeeded(swizzleParams)) {
        LDBG() << "Using swizzled DMA path with rowElems="
               << swizzleParams->rowElems
               << " accessElems=" << swizzleParams->accessElems;
        auto swizzleAttr = IREE::Codegen::XORShuffleAttr::get(
            builder.getContext(), swizzleParams->rowElems,
            swizzleParams->accessElems, /*row_stride=*/0, /*per_phase=*/0);
        dmaDest = IREE::Codegen::SwizzleHintOp::create(builder, loc, allocOp,
                                                       swizzleAttr);
      }
    }

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    SmallVector<Value> zeroIndices(vectorType.getRank(), c0);

    auto dmaOp = IREE::GPU::AsyncDMAOp::create(
        builder, loc, dmaDest.getType(), readOp.getBase(), readOp.getIndices(),
        dmaDest, zeroIndices, TypeAttr::get(vectorType), permMapAttr,
        readOp.getInBoundsAttr());

    auto synced =
        IREE::GPU::ValueBarrierOp::create(builder, loc, dmaOp.getResult());

    SmallVector<bool> readInBounds(vectorType.getRank(), true);
    Value replacement = vector::TransferReadOp::create(
        builder, loc, vectorType, synced.getResult(0), zeroIndices,
        /*padding=*/std::nullopt, readInBounds);
    readOp.getResult().replaceAllUsesWith(replacement);
    readOp->erase();
  }
  return success();
}

/// Materialize shared memory for all to_layout ops marked with
/// shared_memory_conversion. Clears the attribute after materialization.
static LogicalResult
materializeSharedMemoryConversions(FunctionOpInterface funcOp) {
  SmallVector<IREE::VectorExt::ToLayoutOp> opsToPromote;
  funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
    if (op.getSharedMemoryConversion()) {
      opsToPromote.push_back(op);
    }
  });

  OpBuilder builder(funcOp);
  for (IREE::VectorExt::ToLayoutOp op : opsToPromote) {
    // HACK: Until proper barrier placement is handled later we have to
    // synchronize explicitly in this pass.

    // Synchronize before the write to shared memory to avoid stepping over
    // reads in the previous iteration of a loop. We set this barrier
    // at the start of this block.
    builder.setInsertionPointToStart(op->getBlock());
    gpu::BarrierOp::create(builder, op->getLoc(), gpu::AddressSpace::Workgroup);

    builder.setInsertionPoint(op);
    OpOperand &operand = op.getInputMutable();
    // TODO: Since we know the read/write layout for this memory, we can get
    // optimal swizzling here. Figure out how to do that.
    FailureOr<std::pair<Value, Operation *>> promoted =
        promoteViaSharedMemory(builder, op->getLoc(), operand.get());
    if (failed(promoted)) {
      return failure();
    }
    operand.set(promoted->first);

    // Remove the shared_memory_conversion attribute from the to_layout
    // operation.
    op.setSharedMemoryConversion(false);
  }
  return success();
}

struct GPUVectorAllocPass final
    : impl::GPUVectorAllocPassBase<GPUVectorAllocPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Phase 1: Remove stretching broadcasts before layout analysis — the
    // analysis asserts that broadcasts don't stretch.
    {
      RewritePatternSet patterns(funcOp.getContext());
      populateVectorLayoutCanonicalizations(patterns);
      walkAndApplyPatterns(funcOp, std::move(patterns));
    }

    // Phase 2: Run layout analysis to find additional conflict points.
    // The analysis sees the materialized shared memory roundtrips and
    // only detects genuinely new conflicts.
    llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface> layouts;
    propagateVectorLayoutInfo(funcOp, layouts);

    // Mark newly-inserted to_layout ops where input/output layouts don't
    // match — these are genuine conflicts needing shared memory.
    funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
      auto inputLayout = layouts.lookup(op.getInput());
      auto outputLayout = layouts.lookup(op.getResult());
      if (inputLayout && outputLayout &&
          inputLayout.needsSharedMemoryForConversion(outputLayout)) {
        op.setSharedMemoryConversion(true);
      }
    });

    // Phase 3: Materialize DMA promotions (uses layouts map).
    if (failed(materializeDMAPromotions(funcOp, layouts))) {
      return signalPassFailure();
    }

    // Phase 4: Materialize shared memory conversions.
    if (failed(materializeSharedMemoryConversions(funcOp))) {
      return signalPassFailure();
    }

    // Phase 5: Clean up promotion_type attributes from to_layout ops.
    funcOp.walk([](IREE::VectorExt::ToLayoutOp op) {
      op->removeAttr(kPromotionTypeAttr);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
