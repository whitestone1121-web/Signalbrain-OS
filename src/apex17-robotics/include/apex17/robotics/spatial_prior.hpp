#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Apex17::Robotics {

// =============================================================================
// SpatialPrior
// -----------------------------------------------------------------------------
// GPU-resident front-end perception primitive for signal-native robotics.
//
// Purpose:
//   Convert raw spatial sensor streams (LiDAR / RGB-D / radar / fused point sets)
//   into a deterministic, planner-agnostic structural prior:
//
//     - reachability ordering
//     - cluster spans and centroids
//     - outlier mask / anomaly scores
//     - solidity index
//     - occupancy gradient
//     - traversability valleys
//     - scene regime classification
//     - confidence metrics
//
// Design goals:
//   - planner-agnostic
//   - SLAM-agnostic
//   - deterministic outputs
//   - GPU-first / zero-copy capable
//   - ROS2 / Isaac / custom runtime friendly
// =============================================================================

// ---------------------------------------------------------------------------
// Core enums
// ---------------------------------------------------------------------------

enum class DeviceType : uint8_t {
    CPU  = 0,
    CUDA = 1
};

enum class ScalarType : uint8_t {
    Float32  = 0,
    Float16  = 1,
    BFloat16 = 2,
    UInt16   = 3,
    UInt32   = 4,
    Int32    = 5,
    UInt8    = 6
};

enum class SensorModality : uint8_t {
    Unknown       = 0,
    LiDAR         = 1,
    RGBD          = 2,
    Radar         = 3,
    EventCloud    = 4,
    FusedPointSet = 5
};

enum class SceneRegime : uint8_t {
    Unknown              = 0,
    OpenSpace            = 1,
    StructuredIndoor     = 2,
    DenseClutter         = 3,
    DynamicTraffic       = 4,
    LowSolidityOcclusion = 5,
    SparseReturns        = 6,
    SensorDegraded       = 7
};

enum class TraversabilityClass : uint8_t {
    Unknown        = 0,
    Traversable    = 1,
    Caution        = 2,
    NonTraversable = 3
};

enum class RationaleCode : uint16_t {
    None                         = 0,
    HighSolidityObstacle         = 100,
    LowSolidityParticulate       = 101,
    DynamicMotionPressure        = 102,
    ReachabilityValleyConfirmed  = 103,
    ReachabilityDiscontinuity    = 104,
    OutlierBurstDetected         = 105,
    SensorDegradationSuspected   = 106,
    SceneTransitionDetected      = 107,
    ConfidenceLow                = 108
};

enum class StatusCode : uint16_t {
    Ok                = 0,
    InvalidArgument   = 1,
    NotInitialized    = 2,
    DeviceError       = 3,
    UnsupportedFormat = 4,
    BufferTooSmall    = 5,
    InternalError     = 255
};

enum class ExecutionMode : uint8_t {
    Unknown         = 0,
    Proceed         = 1,
    ProceedCautious = 2,
    Stop            = 3,
    Reroute         = 4
};

enum class BackendKind : uint8_t {
    Auto         = 0,
    CPUReference = 1,
    CUDAOptics   = 2
};

// ---------------------------------------------------------------------------
// Basic math structs
// ---------------------------------------------------------------------------

struct Vec3f {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
};

struct Bounds3f {
    Vec3f min{};
    Vec3f max{};
};

struct Pose3f {
    // Row-major 4×4 homogeneous transform: world ← sensor
    std::array<float, 16> T_world_sensor{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
};

// ---------------------------------------------------------------------------
// Generic tensor / buffer view for zero-copy integration
// ---------------------------------------------------------------------------

struct BufferView {
    const void* data{nullptr};
    size_t      bytes{0};
    ScalarType  dtype{ScalarType::Float32};
    DeviceType  device{DeviceType::CPU};

    // Shape + stride in elements, not bytes
    std::vector<int64_t> shape{};
    std::vector<int64_t> stride{};

    // Optional CUDA stream / backend handle (opaque)
    void* stream{nullptr};

    [[nodiscard]] bool valid() const noexcept {
        return data != nullptr && bytes > 0;
    }
};

struct MutableBufferView {
    void*      data{nullptr};
    size_t     bytes{0};
    ScalarType dtype{ScalarType::Float32};
    DeviceType device{DeviceType::CPU};
    std::vector<int64_t> shape{};
    std::vector<int64_t> stride{};
    void* stream{nullptr};

    [[nodiscard]] bool valid() const noexcept {
        return data != nullptr && bytes > 0;
    }
};

// ---------------------------------------------------------------------------
// Input point cloud contract
// ---------------------------------------------------------------------------

struct PointFieldLayout {
    // Offsets in bytes within each point record
    uint32_t x_offset{0};
    uint32_t y_offset{4};
    uint32_t z_offset{8};

    // Optional attributes
    std::optional<uint32_t> intensity_offset{};
    std::optional<uint32_t> reflectivity_offset{};
    std::optional<uint32_t> velocity_x_offset{};
    std::optional<uint32_t> velocity_y_offset{};
    std::optional<uint32_t> velocity_z_offset{};
    std::optional<uint32_t> timestamp_offset{};
    std::optional<uint32_t> ring_offset{};
    std::optional<uint32_t> semantic_offset{};
};

struct PointCloudView {
    BufferView points{};                  // [N, point_stride_bytes] raw interleaved
    uint32_t   point_count{0};
    uint32_t   point_stride_bytes{16};
    PointFieldLayout layout{};
    SensorModality   modality{SensorModality::Unknown};

    // Sensor pose and frame metadata
    Pose3f   sensor_pose{};
    uint64_t timestamp_ns{0};

    // Coordinate frame names (for ROS 2 / TF2 integration)
    std::string frame_name{};        // e.g. "lidar_front", "base_link"
    std::string world_frame_name{};  // e.g. "odom", "map"

    // Optional frame and sequence bookkeeping
    uint64_t frame_id{0};
    uint32_t source_id{0};

    // Physical bounding volume if already known
    std::optional<Bounds3f> sensor_bounds{};
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct ClusteringConfig {
    uint32_t min_samples{10};
    uint32_t min_cluster_size{20};

    // Reachability extraction controls
    float max_reachability_cut{0.0f};   // 0 ⇒ adaptive
    bool  adaptive_cut{true};

    // Internal scale handling
    bool enable_variable_density{true};
    bool enable_outlier_scoring{true};

    // Spatial limits
    float max_range_m{200.0f};
    float min_range_m{0.05f};

    // Optional downsampling / voxelisation
    bool  enable_voxel_prepass{false};
    float voxel_size_m{0.05f};
};

struct TraversabilityConfig {
    bool enable_traversability{true};

    // Density / structure thresholds
    float solidity_obstacle_threshold{0.80f};
    float solidity_particulate_threshold{0.35f};

    // Valley selection
    uint32_t max_valleys{32};
    float    min_valley_width_m{0.40f};
    float    min_clearance_m{0.25f};

    // Local planning prior
    float occupancy_gradient_radius_m{1.0f};
    float traversability_horizon_m{15.0f};
};

struct AnomalyConfig {
    bool  enable_anomaly_detection{true};
    float outlier_score_threshold{0.85f};
    float scene_transition_threshold{0.60f};
    float sensor_degradation_threshold{0.70f};
};

struct PerformanceConfig {
    DeviceType  device{DeviceType::CUDA};
    uint32_t    device_index{0};
    BackendKind backend{BackendKind::Auto};

    // GPU execution policy
    bool use_cuda_graphs{true};
    bool use_persistent_kernel{false};
    bool deterministic{true};

    // Precision
    ScalarType compute_dtype{ScalarType::Float16};
    ScalarType accumulation_dtype{ScalarType::Float32};
    ScalarType score_dtype{ScalarType::Float32};

    // Stream handling
    bool allow_external_stream{true};

    // Workspace caps
    size_t   max_workspace_bytes{1ull << 30};   // 1 GiB
    uint32_t max_points_per_frame{4'000'000};
};

struct SpatialPriorConfig {
    ClusteringConfig     clustering{};
    TraversabilityConfig traversability{};
    AnomalyConfig        anomaly{};
    PerformanceConfig    performance{};

    // Output control
    bool emit_ordering{true};
    bool emit_cluster_labels{true};
    bool emit_outlier_mask{true};
    bool emit_cluster_centroids{true};
    bool emit_rationale_codes{true};

    // RMQ + Persistence feature gates
    bool enable_rmq_valley_queries{false};  // sparse table valley queries (opt-in, Phase G1)
    bool enable_persistence_h0{false};       // H₀ persistence digest (off by default, serial)
    bool emit_persistence_pairs{false};     // full pair list (expensive, for diagnostics)
};

// ---------------------------------------------------------------------------
// Derived outputs
// ---------------------------------------------------------------------------

struct ClusterSpan {
    uint32_t begin_index{0};        // in reachability ordering space
    uint32_t end_index{0};          // exclusive
    uint32_t point_count{0};
    float    reachability_min{0.0f};
    float    reachability_mean{0.0f};
    float    reachability_max{0.0f};
};

struct ClusterDescriptor {
    uint32_t            cluster_id{0};
    ClusterSpan         span{};
    Vec3f               centroid{};
    Bounds3f            bounds{};
    float               solidity_index{0.0f};       // [0,1]
    float               density_score{0.0f};         // [0,1]
    float               dynamic_score{0.0f};         // [0,1]
    TraversabilityClass traversability{TraversabilityClass::Unknown};

    // Cluster motion estimate hooks (for tracker / FlowAgent)
    Vec3f  estimated_velocity{};
    float  velocity_confidence{0.0f};  // [0,1]
};

struct TraversabilityValley {
    uint32_t valley_id{0};
    uint32_t ordering_begin{0};
    uint32_t ordering_end{0};

    Vec3f    centroid{};
    Bounds3f bounds{};

    float depth_score{0.0f};           // stronger valley ⇒ higher
    float width_m{0.0f};
    float clearance_m{0.0f};
    float traversability_score{0.0f};  // [0,1]
    float confidence{0.0f};            // [0,1]

    // Local risk signals around this valley
    float local_solidity_index{0.0f};   // [0,1] — solidity in valley region
    float local_outlier_fraction{0.0f}; // [0,1] — outlier density in valley
    float uncertainty{0.0f};            // [0,1] — independent uncertainty budget
};

// Persistent Homology (H₀) types
struct PersistencePair {
    uint32_t birth_idx{0};       // ordering index where component appears
    uint32_t death_idx{0};       // ordering index where component merges
    float    birth_value{0};     // reachability at birth
    float    death_value{0};     // reachability at death
    float    persistence{0};     // death - birth (structural significance)
};

struct PersistenceDigest {
    float    max_persistence{0};        // longest-lived topological feature
    float    total_persistence{0};      // sum of all feature lifetimes
    float    persistence_entropy{0};    // Shannon entropy of normalized lifetimes
    float    persistence_stability{0};  // high = few dominant structures, low = noisy
    uint32_t num_significant{0};        // features with persistence > 5% of max
    uint32_t num_components{0};         // total H₀ features
};

/// Compact topological scene fingerprint for O(1) WorldIndex matching.
/// Built host-side from persistence pairs — no GPU cost.
struct TopologicalFingerprint {
    static constexpr int kResolution = 16;
    static constexpr int kDims = kResolution * kResolution;  // 256 floats

    float data[kDims] = {};
    size_t hash{0};
    bool   valid{false};

    /// Build persistence image from pairs.
    /// Maps (birth, persistence) into a 16×16 grid, weighted by persistence.
    void Build(const std::vector<PersistencePair>& pairs, float max_filtration) {
        std::memset(data, 0, sizeof(data));
        hash = 0;
        valid = false;

        if (pairs.empty() || max_filtration <= 0.0f) return;

        for (const auto& p : pairs) {
            float x = p.birth_value;
            float y = p.persistence;
            int gx = static_cast<int>((x / max_filtration) * (kResolution - 1));
            int gy = static_cast<int>((y / max_filtration) * (kResolution - 1));
            gx = std::max(0, std::min(kResolution - 1, gx));
            gy = std::max(0, std::min(kResolution - 1, gy));
            data[gy * kResolution + gx] += y;  // weight by persistence
        }

        // Normalize to [0,1]
        float mx = 0;
        for (float v : data) { if (v > mx) mx = v; }
        if (mx > 0) { for (float& v : data) v /= mx; }

        // LSH: quantize to 4-bit and hash
        size_t h = 0;
        for (int i = 0; i < kDims; ++i) {
            int q = static_cast<int>(data[i] * 15.0f);
            h ^= static_cast<size_t>(q * 2654435761u) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        hash = h;
        valid = true;
    }

    /// Similarity score against another fingerprint (cosine-like, [0,1]).
    [[nodiscard]] float Similarity(const TopologicalFingerprint& other) const {
        if (!valid || !other.valid) return 0.0f;
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < kDims; ++i) {
            dot += data[i] * other.data[i];
            na  += data[i] * data[i];
            nb  += other.data[i] * other.data[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return (denom > 1e-10f) ? dot / denom : 0.0f;
    }
};

struct SceneMetrics {
    float        global_solidity_index{0.0f};     // [0,1]
    float        occupancy_gradient{0.0f};        // normalised, see motion_pressure
    float        motion_pressure{0.0f};           // derived dynamic occupancy pressure
    float        outlier_fraction{0.0f};          // [0,1]
    float        scene_change_score{0.0f};        // [0,1]
    float        sensor_degradation_score{0.0f};  // [0,1]
    float        confidence{0.0f};                // [0,1]
    float        uncertainty{0.0f};               // [0,1] — independent of confidence
    SceneRegime  regime{SceneRegime::Unknown};

    // Persistence-derived metrics
    float        persistence_entropy{0.0f};       // topological complexity
    float        max_persistence{0.0f};           // strongest structural feature
    float        persistence_stability{0.0f};     // structural dominance (0=noisy, 1=clean)
};

// ---------------------------------------------------------------------------
// NOTE: Compute() stages outputs to host-owned vectors.
// ComputeInto() is the low-latency path for downstream zero-copy consumers.
// ---------------------------------------------------------------------------

struct SpatialPriorResult {
    uint64_t   frame_id{0};
    uint64_t   timestamp_ns{0};
    StatusCode status{StatusCode::Ok};

    // Core scene interpretation
    SceneMetrics metrics{};

    // Primary planner-facing output
    std::optional<TraversabilityValley> best_valley{};

    // Optional structured outputs (populated per config flags)
    std::vector<uint32_t> reachability_order{};   // point indices in OPTICS order [N]
    std::vector<float>    reachability{};          // reachability distance per ordered point [N]
    std::vector<int32_t>  cluster_labels{};        // per-point cluster id, −1 for outlier [N]
    std::vector<uint8_t>  outlier_mask{};          // per-point mask [N]
    std::vector<float>    outlier_score{};          // per-point score [N], optional
    std::vector<ClusterDescriptor>   clusters{};
    std::vector<TraversabilityValley> valleys{};
    std::vector<RationaleCode>       rationale_codes{};

    // Persistent Homology outputs (gated by config.enable_persistence_h0)
    std::vector<PersistencePair> persistence_pairs{};  // only if emit_persistence_pairs
    PersistenceDigest            persistence_digest{};
    TopologicalFingerprint       topological_fingerprint{};  // scene signature

    // Diagnostic / performance
    uint32_t point_count{0};
    float    compute_ms{0.0f};
    float    reachability_ms{0.0f};
    float    clustering_ms{0.0f};
    float    traversability_ms{0.0f};
    float    rmq_ms{0.0f};
    float    persistence_ms{0.0f};
    float    anomaly_ms{0.0f};

    [[nodiscard]] bool ok() const noexcept {
        return status == StatusCode::Ok;
    }
};

// ---------------------------------------------------------------------------
// Council-facing reduced signal contract
// ---------------------------------------------------------------------------

struct CouncilSignal {
    uint64_t frame_id{0};
    uint64_t timestamp_ns{0};

    float       solidity_index{0.0f};          // scene / global
    float       occupancy_gradient{0.0f};      // normalised gradient
    float       motion_pressure{0.0f};         // derived dynamic occupancy pressure
    bool        has_motion_pressure{false};     // true when motion_pressure is populated
    float       topology_confidence{0.0f};     // overall topology confidence
    float       best_valley_score{0.0f};       // traversability score
    float       best_valley_clearance_m{0.0f};
    float       scene_change_score{0.0f};
    float       degradation_score{0.0f};
    SceneRegime regime{SceneRegime::Unknown};

    // Persistence-derived (populated when enable_persistence_h0 = true)
    float       persistence_stability{0.0f};   // 0=noisy, 1=clean dominant structure
    float       persistence_entropy{0.0f};     // topological complexity

    std::vector<RationaleCode> rationale_codes{};
};

struct AgentReport {
    std::string   agent_name{};
    float         score{0.0f};                 // 0..1
    float         confidence{0.0f};            // 0..1
    float         risk_delta{0.0f};            // signed
    float         recommended_velocity{1.0f};  // multiplier
    ExecutionMode recommended_mode{ExecutionMode::Unknown};
    RationaleCode rationale{RationaleCode::None};
};

struct DirectorDecision {
    uint64_t      frame_id{0};
    uint64_t      timestamp_ns{0};

    float         consensus_delta{0.0f};       // agent disagreement spread
    float         aggregate_risk{0.0f};         // weighted mean risk across agents
    float         velocity_multiplier{1.0f};
    float         path_confidence{0.0f};
    ExecutionMode execution_mode{ExecutionMode::Unknown};
    std::vector<RationaleCode> rationale_codes{};
};

// ---------------------------------------------------------------------------
// Optional external output buffers (zero-copy for downstream consumers)
// ---------------------------------------------------------------------------

struct SpatialPriorOutputBuffers {
    MutableBufferView reachability_order{};
    MutableBufferView reachability{};
    MutableBufferView cluster_labels{};
    MutableBufferView outlier_mask{};
    MutableBufferView outlier_score{};
};

// ---------------------------------------------------------------------------
// Main interface
// ---------------------------------------------------------------------------

class SpatialPrior {
public:
    virtual ~SpatialPrior() = default;

    SpatialPrior(const SpatialPrior&)            = delete;
    SpatialPrior& operator=(const SpatialPrior&) = delete;

    // Factory
    [[nodiscard]] static std::unique_ptr<SpatialPrior>
    Create(const SpatialPriorConfig& config);

    // Lifecycle
    virtual StatusCode Initialize() = 0;
    virtual void       Shutdown()   = 0;
    [[nodiscard]] virtual bool IsInitialized() const noexcept = 0;

    // Configuration
    [[nodiscard]] virtual const SpatialPriorConfig& GetConfig() const noexcept = 0;
    virtual StatusCode Reconfigure(const SpatialPriorConfig& config) = 0;

    // Core compute — owning result
    virtual SpatialPriorResult Compute(const PointCloudView& input) = 0;

    // Core compute — zero-copy / external buffer variant
    virtual StatusCode ComputeInto(
        const PointCloudView& input,
        const SpatialPriorOutputBuffers& buffers,
        SpatialPriorResult* result_out) = 0;

    // Reduced council-facing signal projection
    [[nodiscard]] virtual CouncilSignal
    ToCouncilSignal(const SpatialPriorResult& result) const = 0;

    // Warmup / graph capture
    virtual StatusCode Warmup(uint32_t expected_point_count) = 0;

    // Device synchronisation hook for integrators
    virtual StatusCode Synchronize() = 0;

    // Diagnostics
    [[nodiscard]] virtual std::string DescribeBackend() const = 0;
    [[nodiscard]] virtual size_t      WorkspaceBytes() const noexcept = 0;

protected:
    SpatialPrior() = default;
};

// ---------------------------------------------------------------------------
// Helper free functions
// ---------------------------------------------------------------------------

[[nodiscard]] const char* ToString(DeviceType value) noexcept;
[[nodiscard]] const char* ToString(ScalarType value) noexcept;
[[nodiscard]] const char* ToString(SensorModality value) noexcept;
[[nodiscard]] const char* ToString(SceneRegime value) noexcept;
[[nodiscard]] const char* ToString(TraversabilityClass value) noexcept;
[[nodiscard]] const char* ToString(RationaleCode value) noexcept;
[[nodiscard]] const char* ToString(StatusCode value) noexcept;
[[nodiscard]] const char* ToString(ExecutionMode value) noexcept;
[[nodiscard]] const char* ToString(BackendKind value) noexcept;

// Validation
[[nodiscard]] StatusCode Validate(const SpatialPriorConfig& config) noexcept;
[[nodiscard]] StatusCode Validate(const PointCloudView& input) noexcept;

// ---------------------------------------------------------------------------
// Dashboard event projection
// ---------------------------------------------------------------------------

struct DashboardEvent {
    uint64_t    ts_ns{0};
    StatusCode  status{StatusCode::Ok};             // perception status
    SceneRegime scene_regime{SceneRegime::Unknown};

    float solidity{0.0f};
    float flow_pressure{0.0f};
    float topology_confidence{0.0f};
    float best_valley_depth_score{0.0f};
    float best_valley_clearance_m{0.0f};

    float         consensus_delta{0.0f};       // filled later by Director
    float         velocity_multiplier{1.0f};   // filled later by Director
    ExecutionMode execution_mode{ExecutionMode::Unknown};  // from Director

    std::vector<RationaleCode> rationale_codes{};
};

[[nodiscard]] DashboardEvent ToDashboardEvent(const SpatialPriorResult& result);

// ---------------------------------------------------------------------------
// Implementation notes
// ---------------------------------------------------------------------------
//
// 1. Input contract:
//    - PointCloudView should accept raw DMA / driver-owned buffers.
//    - CUDA path should avoid host-side reshaping / copies.
//    - Structured PointCloud2 decoding occurs in the C++ adapter layer.
//
// 2. Determinism:
//    - For investor / demo mode, enable deterministic path.
//    - For max-throughput, allow non-deterministic but bounded kernels.
//
// 3. Output semantics:
//    - "best_valley" is a traversability prior, not a final trajectory.
//    - "solidity_index" distinguishes particulate clutter vs solid obstacles.
//    - "occupancy_gradient" approximates local motion pressure / flow.
//
// 4. Intended downstream consumers:
//    - SLAM front-end
//    - Local planner
//    - Risk agent / council
//    - Dashboard / explainability stream
//
// ---------------------------------------------------------------------------

} // namespace Apex17::Robotics
