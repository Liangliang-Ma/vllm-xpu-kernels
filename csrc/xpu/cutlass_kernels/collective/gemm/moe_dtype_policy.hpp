#pragma once
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include <cfloat>

#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "moe_array_mma.hpp"
#include "moe_array_epilogue.hpp"
#include "moe_callbacks.hpp"
#include "moe_dtype_policy.hpp"
#include "moe_gemm_array_cooperative.hpp"
#include "moe_tile_scheduler.hpp"

using namespace cute;
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

class moe_policy_base {
 public:
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = float;
  using ElementB = float;
  using ElementOutput = float;
  using ElementScale = float;
  
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
 
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int PipelineStages = 2;
  using EpilogueDispatchPolicy = cutlass::epilogue::MoE16Group;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      float_t,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
};

class moe_bf16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
  
  using TileShape = Shape<_256, _256, _32>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::detail::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::detail::TagToStrideC_t<LayoutD*>,
      FusionCallbacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U16x8x16_ST_N,
      void,
      void>;
  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,  // A
      GmemTiledCopyB,
      void,
      void,
      cute::identity  // B
      >;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

class moe_fp16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementScale = cutlass::half_t;
  
  using TileShape = Shape<_256, _256, _32>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::detail::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::detail::TagToStrideC_t<LayoutD*>,
      FusionCallbacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U16x8x16_ST_N,
      void,
      void>;
  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,  // A
      GmemTiledCopyB,
      void,
      void,
      cute::identity  // B
      >;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

};

class moe_mxfp8_policy : public moe_policy_base {
 public:
  // using ElementType = cutlass::mx_float8_t<float_e4m3_t>;
  // using ElementA = typename ElementType::DataType;
  // using ElementB = typename ElementType::DataType;
  // using ElementOutput = cutlass::bfloat16_t;
  // using ElementScale = typename ElementType::ScaleFactorType;
  // using StrideScale = cute::Stride<_1, int64_t, int64_t>;
  // using GmemTiledCopyScaleA = void;
  // using GmemTiledCopyScaleB = void;
  
  // using TileShape = Shape<_512, _512, _32>;
  // using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>, Layout<TileShape>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
  // using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFP8Group<PipelineStages>;
  // using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
  //         decltype(tile_shape(TiledMma()))>;
  // using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
  //         EpilogueDispatchPolicy,
  //         TileShape,
  //         ElementAccumulator,
  //         cutlass::gemm::TagToStrideC_t<LayoutC*>,
  //         ElementOutput,
  //         cutlass::gemm::TagToStrideC_t<LayoutD*>,
  //         FusionCallBacks,
  //         XE_2D_U32x8x16_LD_N,
  //         void, void,
  //         XE_2D_U32x8x16_ST_N,
  //         void, void>;

  // using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
  //         GEMMDispatchPolicy,
  //         TileShape,
  //         cute::tuple<ElementInputA, ElementScale>,
  //         cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA*>, StrideScale*>,
  //         cute::tuple<ElementInputB, ElementScale>,
  //         cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB*>, StrideScale*>,
  //         TiledMma,
  //         cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>, void, void, cute::identity,  // A
  //         cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>, void, void, cute::identity   // B
  // >;

  // using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  //   ProblemShape,
  //   CollectiveMainloop,
  //   CollectiveEpilogue,
  //   GroupScheduler
  // >;

  // using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementScale = cutlass::half_t;
  
  using TileShape = Shape<_256, _256, _32>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopMoE16Group<PipelineStages>;
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::detail::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::detail::TagToStrideC_t<LayoutD*>,
      FusionCallbacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U16x8x16_ST_N,
      void,
      void>;
  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,  // A
      GmemTiledCopyB,
      void,
      void,
      cute::identity  // B
      >;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


};

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
