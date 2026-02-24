// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:     %s | FileCheck %s

// Test that the vector distribute pipeline correctly handles attention with
// a K2 dimension (4080) that is not aligned to the tile size (64), requiring
// masking.

// CHECK-LABEL: func.func @attention_dispatch_0_attention_20x16x64x4080x64()
// CHECK: scf.for
// CHECK:   vector.create_mask
// CHECK:   arith.select
// CHECK: scf.yield
hal.executable private @attention_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32], workgroup_memory_bank_count = 32>>, ukernels = "none"}>) {
    hal.executable.export public @attention_dispatch_0_attention_20x16x64x4080x64 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_dispatch_0_attention_20x16x64x4080x64() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}>} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>>
        %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>>
        %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>>
        %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<20x4080x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>>
        %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<20x16x64xf16, #hal.descriptor_type<storage_buffer>> to memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        %8 = iree_codegen.load_from_buffer %1 : memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x16x64xf16>
        %9 = iree_codegen.load_from_buffer %3 : memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x4080x64xf16>
        %10 = iree_codegen.load_from_buffer %5 : memref<20x4080x64xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<20x4080x64xf16>
        %11 = tensor.empty() : tensor<20x16x64xf16>
        %12 = iree_linalg_ext.attention {decomposition_config = {pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [1], subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 3, 4]]}>}, qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16>, promote_operands = [0, 1], subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 2, 3]]}>}}, indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>], lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1, 2], reduction = [0, 0, 0, 64, 0], workgroup = [1, 16, 0, 0, 64]}>} ins(%8, %9, %10, %cst : tensor<20x16x64xf16>, tensor<20x4080x64xf16>, tensor<20x4080x64xf16>, f16) outs(%11 : tensor<20x16x64xf16>) {
        ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<20x16x64xf16>
        iree_codegen.store_to_buffer %12, %7 : tensor<20x16x64xf16> into memref<20x16x64xf16, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
    }
  }
}
