// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" --iree-codegen-enable-dma-swizzle=false | FileCheck %s --check-prefix=NOSWIZZLE

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [0, 0]
>

func.func @test(%vector: vector<16x16xf16>) -> vector<16x16xf16> {
  %out = iree_vector_ext.to_layout %vector to layout(#layout) {shared_memory_conversion} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}


//    CHECK-LABEL: func.func @test
//         CHECK:    gpu.barrier memfence [#gpu.address_space<workgroup>]
//         CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x16xf16, #gpu.address_space<workgroup>>
//         CHECK:    %[[WRITE:.+]] = vector.transfer_write %{{.*}}, %[[ALLOC]]
//         CHECK:    %[[BAR:.+]]   = iree_gpu.value_barrier %[[WRITE]]
//         CHECK:    %[[READ:.+]]  = vector.transfer_read %[[BAR]]
//         CHECK:    %[[OUT:.+]]   = iree_vector_ext.to_layout %[[READ]]

// -----

#gpu_target_elem = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>
#exec_target_elem = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_elem}>
#translation_elem = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#layout_dma_elem = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [4, 1],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 1]
>

func.func @async_dma(
    %src: tensor<4x64xf16>, %other: vector<4x64xf16>, %i: index, %j: index)
    -> vector<4x64xf16>
    attributes {hal.executable.target = #exec_target_elem, translation_info = #translation_elem} {
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%i, %j], %cst {in_bounds = [true, true]}
      : tensor<4x64xf16>, vector<4x64xf16>
  %mul = arith.mulf %read, %other : vector<4x64xf16>
  %out = iree_vector_ext.to_layout %mul to layout(#layout_dma_elem)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<4x64xf16>
  return %out : vector<4x64xf16>
}

// CHECK-LABEL: func.func @async_dma
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<4x64xf16>
// CHECK-SAME:    %[[OTHER:[a-zA-Z0-9]+]]: vector<4x64xf16>
// CHECK-SAME:    %[[I:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[J:[a-zA-Z0-9]+]]: index
//      CHECK:    gpu.barrier
//      CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {{.*}} #gpu.address_space<workgroup>
//      CHECK:    %[[DMA:.+]] = iree_gpu.async_dma %[[SRC]][%[[I]], %[[J]]] to %[[ALLOC]]
//      CHECK:    %[[BARRIER:.+]] = iree_gpu.value_barrier %[[DMA]]
//      CHECK:    %[[READ:.+]] = vector.transfer_read %[[BARRIER]]
//      CHECK:    %[[MUL:.+]] = arith.mulf %[[READ]], %[[OTHER]]
//      CHECK:    %[[OUT:.+]] = iree_vector_ext.to_layout %[[MUL]]
// CHECK-NOT:     iree_gpu.promotion_type
//      CHECK:    return %[[OUT]]

// -----

// Test: Fallback when DMA prerequisites are not met (no GPU target).

#layout_fallback = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 16],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @async_dma_fallback(%src: tensor<16x16xf16>, %i: index, %j: index)
    -> vector<16x16xf16> {
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%i, %j], %cst {in_bounds = [true, true]}
      : tensor<16x16xf16>, vector<16x16xf16>
  %out = iree_vector_ext.to_layout %read to layout(#layout_fallback)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}

// CHECK-LABEL: func.func @async_dma_fallback
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<16x16xf16>
//      CHECK:    gpu.barrier
//  CHECK-NOT:    iree_gpu.async_dma
//      CHECK:    %[[READ:.+]] = vector.transfer_read %[[SRC]]
//      CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {{.*}} #gpu.address_space<workgroup>
//      CHECK:    %[[WRITE:.+]] = vector.transfer_write %[[READ]], %[[ALLOC]]
//      CHECK:    %[[BAR:.+]] = iree_gpu.value_barrier %[[WRITE]]
//      CHECK:    %[[READ2:.+]] = vector.transfer_read %[[BAR]]
//      CHECK:    %[[OUT:.+]] = iree_vector_ext.to_layout %[[READ2]]
// CHECK-NOT:     iree_gpu.promotion_type
//      CHECK:    return %[[OUT]]

// -----

// Test: Fallback when transfer is too small for all subgroups.
// Minimum DMA transfer per workgroup = 256 * (32/16) = 512 elements, but
// tensor has only 256 elements.

#gpu_target_small = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>
#exec_target_small = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_small}>
#translation_small = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64>

#layout_small = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1],
  batch_tile = [1, 1, 1],
  outer_tile = [1, 1, 1],
  thread_tile = [1, 4, 16],
  element_tile = [1, 4, 1],

  subgroup_strides = [0, 0, 0],
  thread_strides   = [0, 16, 1]
>

func.func @async_dma_too_small_for_subgroups(
    %src: tensor<1x16x16xf16>, %i: index, %j: index, %k: index)
    -> vector<1x16x16xf16>
    attributes {hal.executable.target = #exec_target_small, translation_info = #translation_small} {
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%i, %j, %k], %cst {in_bounds = [true, true, true]}
      : tensor<1x16x16xf16>, vector<1x16x16xf16>
  %out = iree_vector_ext.to_layout %read to layout(#layout_small)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<1x16x16xf16>
  return %out : vector<1x16x16xf16>
}

// CHECK-LABEL: func.func @async_dma_too_small_for_subgroups
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<1x16x16xf16>
//      CHECK:    gpu.barrier
//  CHECK-NOT:    iree_gpu.async_dma
//      CHECK:    %[[READ:.+]] = vector.transfer_read %[[SRC]]
//      CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {{.*}} #gpu.address_space<workgroup>
//      CHECK:    %[[WRITE:.+]] = vector.transfer_write %[[READ]], %[[ALLOC]]
//      CHECK:    %[[BAR:.+]] = iree_gpu.value_barrier %[[WRITE]]
//      CHECK:    %[[READ2:.+]] = vector.transfer_read %[[BAR]]
//      CHECK:    %[[OUT:.+]] = iree_vector_ext.to_layout %[[READ2]]
//      CHECK:    return %[[OUT]]

// -----

// Test: async_dma with swizzle.

#gpu_target_swizzle = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128],
  workgroup_memory_bank_count = 32
>>
#exec_target_swizzle = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_swizzle}>
#translation_swizzle = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#layout_dma_swizzle = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [4, 1],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 1]
>

func.func @async_dma_swizzled(
    %src: tensor<4x64xf16>, %i: index, %j: index)
    -> vector<4x64xf16>
    attributes {hal.executable.target = #exec_target_swizzle, translation_info = #translation_swizzle} {
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%i, %j], %cst {in_bounds = [true, true]}
      : tensor<4x64xf16>, vector<4x64xf16>
  %out = iree_vector_ext.to_layout %read to layout(#layout_dma_swizzle)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<4x64xf16>
  return %out : vector<4x64xf16>
}

// The element tile is [4, 1] — the innermost (contiguous) dimension is 1, so
// no valid swizzle can be applied. The DMA should proceed without a hint.
// CHECK-LABEL: func.func @async_dma_swizzled
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<4x64xf16>
// CHECK-SAME:    %[[I:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[J:[a-zA-Z0-9]+]]: index
//      CHECK:    gpu.barrier
//      CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {{.*}} : tensor<4x64xf16, #gpu.address_space<workgroup>>
//  CHECK-NOT:    iree_codegen.swizzle_hint
//      CHECK:    %[[DMA:.+]] = iree_gpu.async_dma %[[SRC]][%[[I]], %[[J]]] to %[[ALLOC]]
//      CHECK:    %[[BARRIER:.+]] = iree_gpu.value_barrier %[[DMA]]
//      CHECK:    %[[READ:.+]] = vector.transfer_read %[[BARRIER]]{{.*}} : {{.*}}, vector<4x64xf16>
//  CHECK-NOT:    vector.shape_cast
//      CHECK:    %[[OUT:.+]] = iree_vector_ext.to_layout %[[READ]]
//      CHECK:    return %[[OUT]]

// When swizzle is disabled, fall back to non-swizzled 2D DMA.
// NOSWIZZLE-LABEL: func.func @async_dma_swizzled
// NOSWIZZLE-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<4x64xf16>
//      NOSWIZZLE:    gpu.barrier
//      NOSWIZZLE:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {{.*}} : tensor<4x64xf16, #gpu.address_space<workgroup>>
//  NOSWIZZLE-NOT:    iree_codegen.swizzle_hint
//      NOSWIZZLE:    %[[DMA:.+]] = iree_gpu.async_dma %[[SRC]]{{.*}} to %[[ALLOC]]
//      NOSWIZZLE:    %[[BARRIER:.+]] = iree_gpu.value_barrier %[[DMA]]
//      NOSWIZZLE:    %[[READ:.+]] = vector.transfer_read %[[BARRIER]]{{.*}} : {{.*}}, vector<4x64xf16>
//  NOSWIZZLE-NOT:    vector.shape_cast
//      NOSWIZZLE:    return

// -----

// Test: async_dma with swizzle — element tile has contiguous innermost dim.

#gpu_target_swizzle2 = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128],
  workgroup_memory_bank_count = 32
>>
#exec_target_swizzle2 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_swizzle2}>
#translation_swizzle2 = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#layout_dma_swizzle2 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [1, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [16, 1]
>

func.func @async_dma_swizzled_contiguous(
    %src: tensor<4x128xf16>, %i: index, %j: index)
    -> vector<4x128xf16>
    attributes {hal.executable.target = #exec_target_swizzle2, translation_info = #translation_swizzle2} {
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%i, %j], %cst {in_bounds = [true, true]}
      : tensor<4x128xf16>, vector<4x128xf16>
  %out = iree_vector_ext.to_layout %read to layout(#layout_dma_swizzle2)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<4x128xf16>
  return %out : vector<4x128xf16>
}

// The element tile is [1, 8] — innermost contiguous dim is 8, so swizzle
// should be applied.
// CHECK-LABEL: func.func @async_dma_swizzled_contiguous
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<4x128xf16>
// CHECK-SAME:    %[[I:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[J:[a-zA-Z0-9]+]]: index
//      CHECK:    gpu.barrier
//      CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {{.*}} : tensor<4x128xf16, #gpu.address_space<workgroup>>
//      CHECK:    %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC]][#iree_codegen.xor_shuffle<64, 8>]
//      CHECK:    %[[DMA:.+]] = iree_gpu.async_dma %[[SRC]][%[[I]], %[[J]]] to %[[HINT]]
//      CHECK:    %[[BARRIER:.+]] = iree_gpu.value_barrier %[[DMA]]
//      CHECK:    %[[READ:.+]] = vector.transfer_read %[[BARRIER]]{{.*}} : {{.*}}, vector<4x128xf16>
//      CHECK:    %[[OUT:.+]] = iree_vector_ext.to_layout %[[READ]]
//      CHECK:    return %[[OUT]]
