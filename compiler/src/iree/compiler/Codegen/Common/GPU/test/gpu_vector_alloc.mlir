// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s

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
#translation_elem = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [64, 1, 1] subgroup_size = 64>

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
