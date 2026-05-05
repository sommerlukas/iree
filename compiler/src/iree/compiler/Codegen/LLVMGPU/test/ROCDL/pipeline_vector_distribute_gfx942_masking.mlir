// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

// Test that the vector distribute pipeline correctly handles attention with
// a K1 dimension (63) that is not aligned to the reduction tile size (64),
// requiring masking.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @attention_20x16x63x4096x64()
// CHECK:   arith.cmpi slt
// CHECK:   scf.for
// CHECK:     arith.select
// CHECK:   scf.yield
hal.executable private @attention_k1_unaligned {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_20x16x63x4096x64 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_20x16x63x4096x64() attributes {translation_info = #translation} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x16x63xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x63xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x16x64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 16, 63], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x16x63xf16>> -> tensor<20x16x63xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 63], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x63xf16>> -> tensor<20x4096x63xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %7 = tensor.empty() : tensor<20x16x64xf32>
        %8 = tensor.empty() : tensor<20x16xf32>
        %9:3 = iree_linalg_ext.online_attention {
          decomposition_config = {
            pv_attrs = {
              lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 1, 1, 64], [1, 0, 4, 3]], subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 4, 3]], thread = [0, 0, 0, 8]}>
            },
            qk_attrs = {
              lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 1, 1, 64], [1, 0, 3, 4]], subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 2, 4]], thread = [0, 0, 8, 0]}>
            }
          },
          indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> ()>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
          ],
          lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 0, 256, 0], workgroup = [1, 1, 0, 0, 16], reduction = [0, 0, 64, 0, 0]}>
        } ins(%4, %5, %6, %cst : tensor<20x16x63xf16>, tensor<20x4096x63xf16>, tensor<20x4096x64xf16>, f16) outs(%7, %8, %8 : tensor<20x16x64xf32>, tensor<20x16xf32>, tensor<20x16xf32>) {
        ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<20x16x64xf32>, tensor<20x16xf32>, tensor<20x16xf32>
        iree_tensor_ext.dispatch.tensor.store %9#0, %3, offsets = [0, 0, 0], sizes = [20, 16, 64], strides = [1, 1, 1] : tensor<20x16x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x16x64xf32>>
        return
      }
    }
  }
}

// -----

// Test that the vector distribute pipeline correctly handles attention with
// a K1 dimension (24) that is padded up to the VMFMA reduction width (32) and
// still lowered through the intrinsic route.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}>

// CHECK-LABEL: func.func @attention_dispatch_0_attention_16x24x16x16()
// CHECK: %alloc_7 = memref.alloc() : memref<16x36xf16, #gpu.address_space<workgroup>>
// CHECK: %subview_8 = memref.subview %alloc_7[0, 0] [16, 32] [1, 1]
// CHECK: arith.cmpi slt
// CHECK-NOT: scf.for
// CHECK: vector.transfer_read %alloc_9[%{{.+}}, %{{.+}}], %0 {in_bounds = [true, true]} : memref<16x36xf16, #gpu.address_space<workgroup>>, vector<1x8xf16>
// CHECK: arith.select
hal.executable private @attention_k1_padded_intrinsic {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_dispatch_0_attention_16x24x16x16 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_dispatch_0_attention_16x24x16x16() attributes {translation_info = #translation} {
        %cst = arith.constant 1.000000e+00 : f32
        %cst_0 = arith.constant -3.40282347E+38 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %cst_2 = arith.constant 2.041020e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x24xf16, #hal.descriptor_type<storage_buffer>>
        %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<16x24xf16, #hal.descriptor_type<storage_buffer>> to memref<16x24xf16, #amdgpu.address_space<fat_raw_buffer>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x24xf16, #hal.descriptor_type<storage_buffer>>
        %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<16x24xf16, #hal.descriptor_type<storage_buffer>> to memref<16x24xf16, #amdgpu.address_space<fat_raw_buffer>>
        %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x16xf16, #hal.descriptor_type<storage_buffer>>
        %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<16x16xf16, #hal.descriptor_type<storage_buffer>> to memref<16x16xf16, #amdgpu.address_space<fat_raw_buffer>>
        %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<16x16xf16, #hal.descriptor_type<storage_buffer>>
        %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<16x16xf16, #hal.descriptor_type<storage_buffer>> to memref<16x16xf16, #amdgpu.address_space<fat_raw_buffer>>
        %8 = iree_codegen.load_from_buffer %1 : memref<16x24xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<16x24xf16>
        %9 = iree_codegen.load_from_buffer %3 : memref<16x24xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<16x24xf16>
        %10 = iree_codegen.load_from_buffer %5 : memref<16x16xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<16x16xf16>
        %11 = tensor.empty() : tensor<16x16xf16>
        %12 = tensor.empty() : tensor<16x16xf32>
        %13 = tensor.empty() : tensor<16xf32>
        %14 = linalg.fill ins(%cst_1 : f32) outs(%12 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %15 = linalg.fill ins(%cst_0 : f32) outs(%13 : tensor<16xf32>) -> tensor<16xf32>
        %16 = linalg.fill ins(%cst_1 : f32) outs(%13 : tensor<16xf32>) -> tensor<16xf32>
        %17:3 = iree_linalg_ext.online_attention {decomposition_config = {pv_attrs = {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>, promote_operands = [1], subgroup_basis = [[1, 1, 1, 1], [0, 2, 3]]}>}, qk_attrs = {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16, col_major = true>, promote_operands = [0, 1], subgroup_basis = [[1, 1, 1, 1], [0, 1, 2]]}>}}, indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>], lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1, 2], promoted_tile_shapes = [array<i64: 16, 32>, array<i64: 16, 32>, array<i64: 16, 16>], reduction = [0, 0, 16, 0], workgroup = [16, 0, 0, 16]}>} ins(%8, %9, %10, %cst_2 : tensor<16x24xf16>, tensor<16x24xf16>, tensor<16x16xf16>, f16) outs(%14, %15, %16 : tensor<16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
        ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<16x16xf32>, tensor<16xf32>, tensor<16xf32>
        %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17#2, %17#0 : tensor<16xf32>, tensor<16x16xf32>) outs(%11 : tensor<16x16xf16>) {
        ^bb0(%in: f32, %in_3: f32, %out: f16):
          %19 = arith.divf %cst, %in : f32
          %20 = arith.mulf %19, %in_3 : f32
          %21 = arith.truncf %20 : f32 to f16
          linalg.yield %21 : f16
        } -> tensor<16x16xf16>
        iree_codegen.store_to_buffer %18, %7 : tensor<16x16xf16> into memref<16x16xf16, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
    }
  }
}

// -----

// Test that the vector distribute pipeline correctly handles attention with
// a K2 dimension (4080) that is not aligned to the tile size (64), requiring
// masking.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @attention_20x16x64x4080x64()
// CHECK:   scf.for
// CHECK:     arith.cmpi slt
// CHECK:     arith.select
// CHECK:   scf.yield
hal.executable private @attention_k2_unaligned {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_20x16x64x4080x64 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_20x16x64x4080x64() attributes {translation_info = #translation} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x16x64xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4080x64xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4080x64xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x16x64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 16, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x16x64xf16>> -> tensor<20x16x64xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4080, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4080x64xf16>> -> tensor<20x4080x64xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4080, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4080x64xf16>> -> tensor<20x4080x64xf16>
        %7 = tensor.empty() : tensor<20x16x64xf32>
        %8 = tensor.empty() : tensor<20x16xf32>
        %9:3 = iree_linalg_ext.online_attention {
          decomposition_config = {
            pv_attrs = {
              attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [1], subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 3, 4]]}>},
            qk_attrs = {
              attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16>, promote_operands = [0, 1], subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 2, 3]]}>
            }
          },
          indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> ()>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
          ],
          lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1, 2], reduction = [0, 0, 0, 64, 0], workgroup = [1, 16, 0, 0, 64]}>
        } ins(%4, %5, %6, %cst : tensor<20x16x64xf16>, tensor<20x4080x64xf16>, tensor<20x4080x64xf16>, f16) outs(%7, %8, %8 : tensor<20x16x64xf32>, tensor<20x16xf32>, tensor<20x16xf32>) {
        ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<20x16x64xf32>, tensor<20x16xf32>, tensor<20x16xf32>
        iree_tensor_ext.dispatch.tensor.store %9#0, %3, offsets = [0, 0, 0], sizes = [20, 16, 64], strides = [1, 1, 1] : tensor<20x16x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x16x64xf32>>
        return
      }
    }
  }
}

// -----

// Test that the vector distribute pipeline correctly handles layernorm with
// a reduction dimension (508) that is not aligned to the reduction tile size
// (512), requiring masking.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: func.func @layernorm_reduction_2x508_f16()
// CHECK:   arith.cmpi slt
// CHECK:   arith.select
hal.executable private @layernorm_reduction_unaligned {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @layernorm_reduction_2x508_f16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @layernorm_reduction_2x508_f16() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f16
        %cst_0 = arith.constant 5.080000e+02 : f16
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x508xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<508xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<508xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x508xf16>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 508], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x508xf16>> -> tensor<2x508xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [508], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<508xf16>> -> tensor<508xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [508], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<508xf16>> -> tensor<508xf16>
        %7 = tensor.empty() : tensor<2x508xf16>
        %8 = tensor.empty() : tensor<2xf16>
        %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<2xf16>) -> tensor<2xf16>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<2x508xf16>) outs(%9 : tensor<2xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], reduction = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %out: f16):
          %14 = arith.addf %in, %out : f16
          linalg.yield %14 : f16
        } -> tensor<2xf16>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %10 : tensor<2x508xf16>, tensor<2xf16>) outs(%7 : tensor<2x508xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], serial = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %14 = arith.divf %in_2, %cst_0 : f16
          %15 = arith.subf %in, %14 : f16
          linalg.yield %15 : f16
        } -> tensor<2x508xf16>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%11 : tensor<2x508xf16>) outs(%9 : tensor<2xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], reduction = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %out: f16):
          %14 = arith.mulf %in, %in : f16
          %15 = arith.addf %14, %out : f16
          linalg.yield %15 : f16
        } -> tensor<2xf16>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %12, %5, %6 : tensor<2x508xf16>, tensor<2xf16>, tensor<508xf16>, tensor<508xf16>) outs(%7 : tensor<2x508xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 64], [0, 1]], serial = [0, 512], subgroup_basis = [[1, 1], [0, 1]], workgroup = [1, 0]}>} {
        ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
          %14 = arith.divf %in_2, %cst_0 : f16
          %15 = arith.truncf %cst_1 : f32 to f16
          %16 = arith.addf %14, %15 : f16
          %17 = math.rsqrt %16 : f16
          %18 = arith.mulf %in, %17 : f16
          %19 = arith.mulf %18, %in_3 : f16
          %20 = arith.addf %19, %in_4 : f16
          linalg.yield %20 : f16
        } -> tensor<2x508xf16>
        iree_tensor_ext.dispatch.tensor.store %13, %3, offsets = [0, 0], sizes = [2, 508], strides = [1, 1] : tensor<2x508xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x508xf16>>
        return
      }
    }
  }
}
