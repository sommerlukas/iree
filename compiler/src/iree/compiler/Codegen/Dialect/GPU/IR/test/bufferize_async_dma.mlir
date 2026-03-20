// RUN: iree-opt %s --one-shot-bufferize="bufferize-function-boundaries" --split-input-file | FileCheck %s

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [1, 1],
  subgroup_strides = [0, 0],
  thread_strides = [0, 1]
>

// Test basic tensor -> memref bufferization.
func.func @bufferize_async_dma(%source: tensor<20x64xf16>,
                                %dest: tensor<1x64xf16>,
                                %i: index, %j: index, %c0: index)
    -> tensor<1x64xf16> {
  %0 = iree_gpu.async_dma %source[%i, %j], %dest[%c0, %c0], #layout
      : tensor<20x64xf16> -> tensor<1x64xf16> -> tensor<1x64xf16>
  return %0 : tensor<1x64xf16>
}

// CHECK-LABEL: func @bufferize_async_dma
//       CHECK:   iree_gpu.async_dma {{.+}} : memref<20x64xf16{{.*}}> -> memref<1x64xf16{{.*}}>
//   CHECK-NOT:   ->

// -----

#layout_ib = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [1, 1],
  subgroup_strides = [0, 0],
  thread_strides = [0, 1]
>

// Test bufferization with in_bounds attribute.
func.func @bufferize_async_dma_in_bounds(%source: tensor<20x64xf16>,
                                          %dest: tensor<1x64xf16>,
                                          %i: index, %j: index, %c0: index)
    -> tensor<1x64xf16> {
  %0 = iree_gpu.async_dma %source[%i, %j], %dest[%c0, %c0], #layout_ib
      in_bounds [true, false]
      : tensor<20x64xf16> -> tensor<1x64xf16> -> tensor<1x64xf16>
  return %0 : tensor<1x64xf16>
}

// CHECK-LABEL: func @bufferize_async_dma_in_bounds
//       CHECK:   iree_gpu.async_dma {{.+}} in_bounds [true, false] : memref<20x64xf16{{.*}}> -> memref<1x64xf16{{.*}}>
//   CHECK-NOT:   ->
