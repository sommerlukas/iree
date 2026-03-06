// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "iree-codegen-vector-tile-size-analysis"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

using TileSizeSet = llvm::SmallSet<int64_t, 2>;

/// Per-dimension tile size candidates. Each dimension has an independent set
/// of candidate tile sizes. Satisfies the requirements for use as a
/// `Lattice<ValueT>` value type.
class TileSizeCandidates {
public:
  TileSizeCandidates() = default;
  explicit TileSizeCandidates(unsigned rank) : dims(rank) {}

  /// Construct from a single concrete tile size (one value per dimension).
  static TileSizeCandidates fromSizes(ArrayRef<int64_t> sizes) {
    TileSizeCandidates result(sizes.size());
    for (unsigned i = 0; i < sizes.size(); ++i) {
      result.dims[i].insert(sizes[i]);
    }
    return result;
  }

  unsigned rank() const { return dims.size(); }
  bool empty() const { return dims.empty(); }

  const TileSizeSet &operator[](unsigned i) const { return dims[i]; }
  TileSizeSet &operator[](unsigned i) { return dims[i]; }

  /// Merge candidates from `other` into this. Returns true if anything changed.
  bool merge(const TileSizeCandidates &other) {
    assert(rank() == other.rank() && "rank mismatch");
    bool changed = false;
    for (unsigned i = 0; i < rank(); ++i) {
      for (int64_t v : other.dims[i]) {
        changed |= dims[i].insert(v).second;
      }
    }
    return changed;
  }

  /// Returns true if any dimension has more than one candidate.
  bool hasAlternatives() const {
    return llvm::any_of(dims,
                        [](const TileSizeSet &s) { return s.size() > 1; });
  }

  /// Map from operand space to iteration space via an indexing map.
  TileSizeCandidates mapToIterationSpace(AffineMap indexingMap,
                                         unsigned numLoops) const {
    TileSizeCandidates result(numLoops);
    for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
      if (!dimExpr) {
        continue;
      }
      unsigned iterDim = dimExpr.getPosition();
      for (int64_t v : dims[i]) {
        result.dims[iterDim].insert(v);
      }
    }
    return result;
  }

  /// Map from iteration space to operand space via an indexing map.
  /// Returns empty TileSizeCandidates if any operand dim can't be determined.
  TileSizeCandidates mapFromIterationSpace(AffineMap indexingMap) const {
    unsigned numResults = indexingMap.getNumResults();
    TileSizeCandidates result(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
      if (!dimExpr) {
        return {};
      }
      unsigned iterDim = dimExpr.getPosition();
      if (iterDim >= rank() || dims[iterDim].empty()) {
        return {};
      }
      result.dims[i] = dims[iterDim];
    }
    return result;
  }

  /// Lattice join: per-dimension set union. Uninitialized is identity.
  static TileSizeCandidates join(const TileSizeCandidates &lhs,
                                 const TileSizeCandidates &rhs) {
    if (lhs.empty()) {
      return rhs;
    }
    if (rhs.empty()) {
      return lhs;
    }
    TileSizeCandidates result = lhs;
    result.merge(rhs);
    return result;
  }

  /// Lattice meet: same as join (both directions accumulate via set union).
  static TileSizeCandidates meet(const TileSizeCandidates &lhs,
                                 const TileSizeCandidates &rhs) {
    return join(lhs, rhs);
  }

  bool operator==(const TileSizeCandidates &rhs) const {
    return dims == rhs.dims;
  }

  void print(raw_ostream &os) const {
    os << "[";
    llvm::interleaveComma(dims, os, [&](const TileSizeSet &s) {
      os << "{";
      llvm::interleaveComma(s, os);
      os << "}";
    });
    os << "]";
  }

private:
  SmallVector<TileSizeSet> dims;
};

/// Returns true if the operation is trivially duplicatable and should not
/// propagate merged tile sizes across independent consumers.
static bool isDuplicatable(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (isa<tensor::EmptyOp>(defOp)) {
    return true;
  }
  if (defOp->hasTrait<OpTrait::ConstantLike>()) {
    return true;
  }
  // Catches linalg.fill that has been lowered/fused into linalg.generic form
  // (scalar input broadcast into tensor.empty output).
  if (auto genericOp = dyn_cast<linalg::GenericOp>(defOp)) {
    if (genericOp.getNumDpsInputs() == 1 && genericOp.getNumDpsInits() == 1 &&
        !isa<ShapedType>(genericOp.getDpsInputs()[0].getType())) {
      Value init = genericOp.getDpsInits()[0];
      if (init.getDefiningOp<tensor::EmptyOp>()) {
        return true;
      }
    }
  }
  if (auto fillOp = dyn_cast<linalg::FillOp>(defOp)) {
    if (fillOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>()) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Lattice and analysis definitions
//===----------------------------------------------------------------------===//

class TileSizeLattice : public dataflow::Lattice<TileSizeCandidates> {
public:
  using Lattice::Lattice;
};

/// Read the TileSizeCandidates from a lattice, returning empty candidates
/// if the lattice value is duplicatable with alternatives.
static const TileSizeCandidates &
getCandidatesFor(Value val, const TileSizeLattice *lattice) {
  static const TileSizeCandidates empty;
  if (!lattice) {
    return empty;
  }
  auto &candidates = lattice->getValue();
  if (candidates.empty()) {
    return empty;
  }
  if (isDuplicatable(val) && candidates.hasAlternatives()) {
    return empty;
  }
  return candidates;
}

/// Gather tile size candidates into the iteration space of a linalg op by
/// looking up each operand value's candidates via `getCandidates`.
static TileSizeCandidates getIterationSpaceTileSizes(
    linalg::LinalgOp linalgOp,
    function_ref<const TileSizeCandidates &(Value)> getCandidates) {
  unsigned numLoops = linalgOp.getNumLoops();
  TileSizeCandidates iterCandidates(numLoops);
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    auto &candidates = getCandidates(operand.get());
    if (candidates.empty()) {
      continue;
    }
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    auto mapped = candidates.mapToIterationSpace(map, numLoops);
    iterCandidates.merge(mapped);
  }
  return iterCandidates;
}

/// Forward analysis: propagates tile size candidates from operands to results.
/// Control flow through scf.for/scf.if is handled automatically by the
/// framework via RegionBranchOpInterface.
class TileSizeForwardAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TileSizeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    // Seed to_layout anchors before the regular initialization. This ensures
    // seeds are set even for to_layout ops in regions that DeadCodeAnalysis
    // hasn't yet marked as live during init.
    top->walk([&](ToLayoutOp toLayout) {
      LDBG() << "Anchor: " << toLayout;
      auto candidates = TileSizeCandidates::fromSizes(
          toLayout.getLayout().getUndistributedShape());
      auto *lattice = getLatticeElement(toLayout.getResult());
      propagateIfChanged(lattice, lattice->join(candidates));
    });
    return SparseForwardDataFlowAnalysis::initialize(top);
  }

  void setToEntryState(TileSizeLattice *lattice) override {
    // Entry state is uninitialized (identity for join).
    propagateIfChanged(lattice, lattice->join(TileSizeCandidates()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const TileSizeLattice *> operands,
                               ArrayRef<TileSizeLattice *> results) override {
    // to_layout: don't propagate operand forward (anchor boundary).
    // Seeding is done in initialize().
    if (isa<ToLayoutOp>(op)) {
      return success();
    }

    // linalg.generic: propagate through indexing maps.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      unsigned numLoops = genericOp.getNumLoops();
      TileSizeCandidates iterCandidates(numLoops);
      for (OpOperand &operand : genericOp->getOpOperands()) {
        auto &candidates = getCandidatesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        if (candidates.empty()) {
          continue;
        }
        AffineMap map = genericOp.getMatchingIndexingMap(&operand);
        iterCandidates.merge(candidates.mapToIterationSpace(map, numLoops));
      }
      for (unsigned i = 0; i < genericOp.getNumDpsInits(); ++i) {
        OpOperand *init = genericOp.getDpsInitOperand(i);
        AffineMap map = genericOp.getMatchingIndexingMap(init);
        auto resultCandidates = iterCandidates.mapFromIterationSpace(map);
        if (!resultCandidates.empty()) {
          propagateIfChanged(results[i], results[i]->join(resultCandidates));
        }
      }
      return success();
    }

    // Elementwise ops: propagate to all results.
    if (OpTrait::hasElementwiseMappableTraits(op)) {
      for (auto [operandLattice, operandVal] :
           llvm::zip(operands, op->getOperands())) {
        auto &candidates = getCandidatesFor(operandVal, operandLattice);
        if (candidates.empty()) {
          continue;
        }
        for (TileSizeLattice *result : results) {
          propagateIfChanged(result, result->join(candidates));
        }
      }
      return success();
    }

    return success();
  }
};

/// Backward analysis: propagates tile size candidates from results to operands.
/// Control flow through scf.for/scf.if is handled automatically by the
/// framework via RegionBranchOpInterface.
class TileSizeBackwardAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<TileSizeLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void setToExitState(TileSizeLattice *lattice) override {
    // Exit state is uninitialized (identity for meet).
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<TileSizeLattice *> operands,
                 ArrayRef<const TileSizeLattice *> results) override {
    // to_layout: propagate result tile sizes backward to input.
    if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
      auto &candidates = getCandidatesFor(toLayout.getResult(), results[0]);
      if (!candidates.empty()) {
        TileSizeLattice *inputLattice = operands[0];
        propagateIfChanged(inputLattice, inputLattice->meet(candidates));
      }
      return success();
    }

    // linalg.generic: propagate through indexing maps.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      unsigned numLoops = genericOp.getNumLoops();
      TileSizeCandidates iterCandidates(numLoops);
      // Gather result candidates into iteration space via DPS init maps.
      for (auto [result, resultLattice] :
           llvm::zip(genericOp.getResults(), results)) {
        auto &candidates = getCandidatesFor(result, resultLattice);
        if (candidates.empty()) {
          continue;
        }
        unsigned resultIdx = cast<OpResult>(result).getResultNumber();
        OpOperand *init = genericOp.getDpsInitOperand(resultIdx);
        AffineMap map = genericOp.getMatchingIndexingMap(init);
        iterCandidates.merge(candidates.mapToIterationSpace(map, numLoops));
      }
      // Gather operand candidates into iteration space.
      for (OpOperand &operand : genericOp->getOpOperands()) {
        auto &candidates = getCandidatesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        if (candidates.empty()) {
          continue;
        }
        AffineMap map = genericOp.getMatchingIndexingMap(&operand);
        iterCandidates.merge(candidates.mapToIterationSpace(map, numLoops));
      }
      // Map iteration space candidates back to each operand.
      for (OpOperand &operand : genericOp->getOpOperands()) {
        AffineMap map = genericOp.getMatchingIndexingMap(&operand);
        auto operandCandidates = iterCandidates.mapFromIterationSpace(map);
        if (operandCandidates.empty()) {
          continue;
        }
        TileSizeLattice *operandLattice = operands[operand.getOperandNumber()];
        propagateIfChanged(operandLattice,
                           operandLattice->meet(operandCandidates));
      }
      return success();
    }

    // Elementwise ops: propagate to all operands.
    if (OpTrait::hasElementwiseMappableTraits(op)) {
      for (auto [resultVal, resultLattice] :
           llvm::zip(op->getResults(), results)) {
        auto &candidates = getCandidatesFor(resultVal, resultLattice);
        if (candidates.empty()) {
          continue;
        }
        for (auto [operandLattice, operandVal] :
             llvm::zip(operands, op->getOperands())) {
          if (!isa<ShapedType>(operandVal.getType())) {
            continue;
          }
          propagateIfChanged(operandLattice, operandLattice->meet(candidates));
        }
      }
      return success();
    }

    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {}
  void visitCallOperand(OpOperand &operand) override {}
  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}
};

//===----------------------------------------------------------------------===//
// Result querying
//===----------------------------------------------------------------------===//

/// Collect per-dimension tile size candidate sets from a linalg op's operands,
/// querying the solver for lattice state.
static TileSizeCandidates
getIterationSpaceTileSizes(linalg::LinalgOp linalgOp,
                           const DataFlowSolver &solver) {
  return getIterationSpaceTileSizes(
      linalgOp, [&](Value v) -> const TileSizeCandidates & {
        auto *lattice = solver.lookupState<TileSizeLattice>(v);
        return getCandidatesFor(v, lattice);
      });
}

/// Given a linalg op and the solver, compute per-dimension sets of
/// candidate tile sizes. Returns a vector of size numLoops, where each entry
/// is the deduplicated set of tile sizes for that iteration dimension.
/// Returns an empty vector if any dimension has no candidates.
static SmallVector<SmallVector<int64_t>>
getPerDimTileSizes(linalg::LinalgOp linalgOp, const DataFlowSolver &solver) {
  auto perDimSizes = getIterationSpaceTileSizes(linalgOp, solver);

  // Return empty if any dimension has no candidates.
  SmallVector<SmallVector<int64_t>> results;
  for (unsigned i = 0; i < perDimSizes.rank(); ++i) {
    if (perDimSizes[i].empty()) {
      return {};
    }
    results.push_back(
        SmallVector<int64_t>(perDimSizes[i].begin(), perDimSizes[i].end()));
  }
  return results;
}

//===----------------------------------------------------------------------===//
// MaterializeVectorTileSizesPass
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_MATERIALIZEVECTORTILESIZESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class MaterializeVectorTileSizesPass final
    : public impl::MaterializeVectorTileSizesPassBase<
          MaterializeVectorTileSizesPass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<TileSizeForwardAnalysis>();
    SymbolTableCollection symbolTable;
    solver.load<TileSizeBackwardAnalysis>(symbolTable);

    if (failed(solver.initializeAndRun(funcOp))) {
      return signalPassFailure();
    }

    funcOp->walk([&](linalg::LinalgOp linalgOp) {
      auto perDimSizes = getPerDimTileSizes(linalgOp, solver);
      if (perDimSizes.empty()) {
        return;
      }

      LDBG() << "Materializing tile size on " << *linalgOp;

      SmallVector<Attribute> dimAttrs;
      for (const auto &dimSizes : perDimSizes) {
        dimAttrs.push_back(
            DenseI64ArrayAttr::get(linalgOp->getContext(), dimSizes));
      }
      linalgOp->setAttr(kVectorTileSizesAttrName,
                        ArrayAttr::get(linalgOp->getContext(), dimAttrs));
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
