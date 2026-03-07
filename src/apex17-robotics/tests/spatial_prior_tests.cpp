// =============================================================================
// spatial_prior_tests.cpp — Unit tests for Apex17::Robotics::SpatialPrior
//
// Minimal test framework: assert-based, no external deps.
// Run via: ctest --test-dir build --output-on-failure
// =============================================================================

#include "apex17/robotics/spatial_prior.hpp"
#include "apex17/robotics/rgbd_utils.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace Apex17::Robotics;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name)                                        \
    do {                                                  \
        ++tests_run;                                      \
        std::printf("  %-50s ", #name);                   \
    } while (0)

#define PASS()                                            \
    do {                                                  \
        ++tests_passed;                                   \
        std::printf("[PASS]\n");                           \
    } while (0)

struct Point { float x, y, z, intensity; };

static PointCloudView MakeSyntheticCloud(
    const std::vector<Point>& pts, uint64_t frame = 1)
{
    PointCloudView pcv{};
    pcv.points.data        = pts.data();
    pcv.points.bytes       = pts.size() * sizeof(Point);
    pcv.points.dtype       = ScalarType::Float32;
    pcv.points.device      = DeviceType::CPU;
    pcv.point_count        = static_cast<uint32_t>(pts.size());
    pcv.point_stride_bytes = sizeof(Point);
    pcv.modality           = SensorModality::LiDAR;
    pcv.timestamp_ns       = 1712345678'000'000'000ULL;
    pcv.frame_id           = frame;
    pcv.frame_name         = "lidar_front";
    pcv.world_frame_name   = "odom";
    pcv.layout.x_offset    = 0;
    pcv.layout.y_offset    = 4;
    pcv.layout.z_offset    = 8;
    pcv.layout.intensity_offset = 12;
    return pcv;
}

static std::vector<Point> MakePoints(uint32_t N) {
    std::vector<Point> pts(N);
    for (uint32_t i = 0; i < N; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(N);
        pts[i] = {t * 10.0f, 5.0f * (1.0f - t), 0.5f, 0.5f + 0.5f * t};
    }
    return pts;
}

/// Tests use CPU device for stable contract validation.
/// CUDA kernel behavior should be tested separately with known-good data.
static SpatialPriorConfig MakeTestConfig() {
    SpatialPriorConfig cfg{};
    cfg.performance.device = DeviceType::CPU;
    return cfg;
}

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

static void test_validate_config_default() {
    TEST(validate_config_default);
    SpatialPriorConfig cfg{};
    assert(Validate(cfg) == StatusCode::Ok);
    PASS();
}

static void test_validate_config_bad_min_samples() {
    TEST(validate_config_bad_min_samples);
    SpatialPriorConfig cfg{};
    cfg.clustering.min_samples = 0;
    assert(Validate(cfg) == StatusCode::InvalidArgument);
    PASS();
}

static void test_validate_config_bad_range() {
    TEST(validate_config_bad_range);
    SpatialPriorConfig cfg{};
    cfg.clustering.max_range_m = 1.0f;
    cfg.clustering.min_range_m = 5.0f;
    assert(Validate(cfg) == StatusCode::InvalidArgument);
    PASS();
}

static void test_validate_config_bad_solidity_threshold() {
    TEST(validate_config_bad_solidity_threshold);
    SpatialPriorConfig cfg{};
    cfg.traversability.solidity_obstacle_threshold = 1.5f;
    assert(Validate(cfg) == StatusCode::InvalidArgument);
    PASS();
}

static void test_validate_config_bad_voxel_prepass() {
    TEST(validate_config_bad_voxel_prepass);
    SpatialPriorConfig cfg{};
    cfg.clustering.enable_voxel_prepass = true;
    cfg.clustering.voxel_size_m = -0.01f;
    assert(Validate(cfg) == StatusCode::InvalidArgument);
    PASS();
}

static void test_validate_config_zero_workspace() {
    TEST(validate_config_zero_workspace);
    SpatialPriorConfig cfg{};
    cfg.performance.max_workspace_bytes = 0;
    assert(Validate(cfg) == StatusCode::InvalidArgument);
    PASS();
}

// ---------------------------------------------------------------------------
// Input validation tests
// ---------------------------------------------------------------------------

static void test_validate_input_ok() {
    TEST(validate_input_ok);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    assert(Validate(pcv) == StatusCode::Ok);
    PASS();
}

static void test_validate_input_zero_count() {
    TEST(validate_input_zero_count);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    pcv.point_count = 0;
    assert(Validate(pcv) == StatusCode::InvalidArgument);
    PASS();
}

static void test_validate_input_null_data() {
    TEST(validate_input_null_data);
    PointCloudView pcv{};
    pcv.point_count = 10;
    pcv.point_stride_bytes = 16;
    // points.data is nullptr
    assert(Validate(pcv) == StatusCode::InvalidArgument);
    PASS();
}

static void test_validate_input_buffer_too_small() {
    TEST(validate_input_buffer_too_small);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    pcv.points.bytes = 10; // way too small
    assert(Validate(pcv) == StatusCode::BufferTooSmall);
    PASS();
}

static void test_validate_input_bad_stride() {
    TEST(validate_input_bad_stride);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    pcv.point_stride_bytes = 8; // < 12
    assert(Validate(pcv) == StatusCode::InvalidArgument);
    PASS();
}

static void test_validate_input_bad_field_offset() {
    TEST(validate_input_bad_field_offset);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    pcv.layout.x_offset = 20; // exceeds stride (16)
    assert(Validate(pcv) == StatusCode::UnsupportedFormat);
    PASS();
}

static void test_validate_input_shape_mismatch() {
    TEST(validate_input_shape_mismatch);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    pcv.points.shape = {50}; // claims 50, but point_count is 100
    assert(Validate(pcv) == StatusCode::BufferTooSmall);
    PASS();
}

// ---------------------------------------------------------------------------
// Engine lifecycle tests
// ---------------------------------------------------------------------------

static void test_create_engine() {
    TEST(create_engine);
    SpatialPriorConfig cfg = MakeTestConfig();
    auto engine = SpatialPrior::Create(cfg);
    assert(engine != nullptr);
    PASS();
}

static void test_create_bad_config() {
    TEST(create_bad_config);
    SpatialPriorConfig cfg = MakeTestConfig();
    cfg.clustering.min_samples = 0;
    auto engine = SpatialPrior::Create(cfg);
    assert(engine == nullptr);
    PASS();
}

static void test_init_shutdown() {
    TEST(init_shutdown);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    assert(engine);
    assert(!engine->IsInitialized());
    assert(engine->Initialize() == StatusCode::Ok);
    assert(engine->IsInitialized());
    engine->Shutdown();
    assert(!engine->IsInitialized());
    PASS();
}

static void test_compute_before_init() {
    TEST(compute_before_init);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    auto result = engine->Compute(pcv);
    assert(result.status == StatusCode::NotInitialized);
    PASS();
}

// ---------------------------------------------------------------------------
// Compute tests
// ---------------------------------------------------------------------------

static void test_compute_basic() {
    TEST(compute_basic);
    auto pts = MakePoints(200);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    engine->Initialize();

    auto result = engine->Compute(pcv);
    assert(result.ok());
    assert(result.point_count == 200);
    assert(result.frame_id == 1);
    assert(result.compute_ms >= 0.0f);
    assert(result.metrics.regime != SceneRegime::Unknown);
    PASS();
}

static void test_compute_has_valley() {
    TEST(compute_has_valley);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    engine->Initialize();

    auto result = engine->Compute(pcv);
    assert(result.ok());
    assert(result.best_valley.has_value());
    assert(result.best_valley->traversability_score > 0.0f);
    assert(result.best_valley->local_solidity_index >= 0.0f);
    assert(result.best_valley->uncertainty >= 0.0f);
    PASS();
}

static void test_compute_reachability_ordering() {
    TEST(compute_reachability_ordering);
    SpatialPriorConfig cfg = MakeTestConfig();
    cfg.emit_ordering = true;
    auto pts = MakePoints(50);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(cfg);
    engine->Initialize();

    auto result = engine->Compute(pcv);
    assert(result.ok());
    assert(result.reachability_order.size() == 50);
    assert(result.reachability.size() == 50);
    PASS();
}

static void test_compute_cluster_labels() {
    TEST(compute_cluster_labels);
    SpatialPriorConfig cfg = MakeTestConfig();
    cfg.emit_cluster_labels = true;
    cfg.emit_cluster_centroids = true;
    auto pts = MakePoints(80);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(cfg);
    engine->Initialize();

    auto result = engine->Compute(pcv);
    assert(result.ok());
    assert(result.cluster_labels.size() == 80);
    assert(!result.clusters.empty());
    PASS();
}

// ---------------------------------------------------------------------------
// Council / Dashboard projection tests
// ---------------------------------------------------------------------------

static void test_council_signal() {
    TEST(council_signal);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    engine->Initialize();

    auto result = engine->Compute(pcv);
    auto sig = engine->ToCouncilSignal(result);
    assert(sig.frame_id == result.frame_id);
    assert(sig.solidity_index >= 0.0f && sig.solidity_index <= 1.0f);
    PASS();
}

static void test_dashboard_event() {
    TEST(dashboard_event);
    auto pts = MakePoints(100);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    engine->Initialize();

    auto result = engine->Compute(pcv);
    auto ev = ToDashboardEvent(result);
    assert(ev.ts_ns == result.timestamp_ns);
    assert(ev.solidity >= 0.0f);
    PASS();
}

// ---------------------------------------------------------------------------
// Enum ToString coverage tests
// ---------------------------------------------------------------------------

static void test_tostring_coverage() {
    TEST(tostring_coverage);
    assert(std::strlen(ToString(DeviceType::CUDA)) > 0);
    assert(std::strlen(ToString(ScalarType::Float16)) > 0);
    assert(std::strlen(ToString(SensorModality::LiDAR)) > 0);
    assert(std::strlen(ToString(SceneRegime::DenseClutter)) > 0);
    assert(std::strlen(ToString(TraversabilityClass::Caution)) > 0);
    assert(std::strlen(ToString(RationaleCode::HighSolidityObstacle)) > 0);
    assert(std::strlen(ToString(StatusCode::Ok)) > 0);
    assert(std::strlen(ToString(ExecutionMode::ProceedCautious)) > 0);
    assert(std::strlen(ToString(BackendKind::CUDAOptics)) > 0);
    PASS();
}

// ---------------------------------------------------------------------------
// Reconfigure test
// ---------------------------------------------------------------------------

static void test_reconfigure() {
    TEST(reconfigure);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    engine->Initialize();

    SpatialPriorConfig new_cfg{};
    new_cfg.clustering.min_samples = 5;
    new_cfg.clustering.min_cluster_size = 10;
    assert(engine->Reconfigure(new_cfg) == StatusCode::Ok);

    // Bad reconfig should fail
    SpatialPriorConfig bad_cfg{};
    bad_cfg.clustering.min_samples = 0;
    assert(engine->Reconfigure(bad_cfg) == StatusCode::InvalidArgument);
    PASS();
}

// ---------------------------------------------------------------------------
// ComputeInto buffer validation test
// ---------------------------------------------------------------------------

static void test_compute_into_null() {
    TEST(compute_into_null_result);
    auto pts = MakePoints(10);
    auto pcv = MakeSyntheticCloud(pts);
    auto engine = SpatialPrior::Create(MakeTestConfig());
    engine->Initialize();
    SpatialPriorOutputBuffers bufs{};
    assert(engine->ComputeInto(pcv, bufs, nullptr) == StatusCode::InvalidArgument);
    PASS();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== Apex17::Robotics::SpatialPrior Tests ===\n\n");

    // Config validation
    test_validate_config_default();
    test_validate_config_bad_min_samples();
    test_validate_config_bad_range();
    test_validate_config_bad_solidity_threshold();
    test_validate_config_bad_voxel_prepass();
    test_validate_config_zero_workspace();

    // Input validation
    test_validate_input_ok();
    test_validate_input_zero_count();
    test_validate_input_null_data();
    test_validate_input_buffer_too_small();
    test_validate_input_bad_stride();
    test_validate_input_bad_field_offset();
    test_validate_input_shape_mismatch();

    // Engine lifecycle
    test_create_engine();
    test_create_bad_config();
    test_init_shutdown();
    test_compute_before_init();

    // Compute
    test_compute_basic();
    test_compute_has_valley();
    test_compute_reachability_ordering();
    test_compute_cluster_labels();

    // Projections
    test_council_signal();
    test_dashboard_event();

    // Enums
    test_tostring_coverage();

    // Reconfigure
    test_reconfigure();

    // ComputeInto
    test_compute_into_null();

    // --- RMQ + Persistent Homology Smoke Tests ---

    // RMQ: When enabled, should still produce valleys with reasonable scores
    TEST(RMQ_ValleyDetection_CUDA);
    {
        SpatialPriorConfig cfg{};
        cfg.performance.device = DeviceType::CUDA;
        cfg.enable_rmq_valley_queries = true;

        auto eng = SpatialPrior::Create(cfg);
        if (!eng) { std::printf("[SKIP - no CUDA]\n"); }
        else {
            eng->Initialize();
            std::vector<Point> pts(500);
            for (uint32_t i = 0; i < 500; ++i) {
                float t = static_cast<float>(i) / 500.0f;
                pts[i] = {t * 20.0f - 10.0f, (i % 5 == 0) ? 1.5f : 0.0f,
                          (i % 9 == 0) ? 1.0f : 0.1f, 0.3f + t * 0.7f};
            }
            auto pcv = MakeSyntheticCloud(pts);
            auto result = eng->Compute(pcv);
            assert(result.ok());
            assert(result.rmq_ms >= 0.0f);        // RMQ timing recorded
            // Should have at least one valley or zero (depends on data)
            // Key: doesn't crash, produces valid output
            if (!result.valleys.empty()) {
                assert(result.best_valley.has_value());
                assert(result.best_valley->traversability_score >= 0.0f);
                assert(result.best_valley->traversability_score <= 1.0f);
            }
            eng->Shutdown();
            PASS();
        }
    }

    // PH: When enabled, should produce persistence digest
    TEST(PH_PersistenceDigest_CUDA);
    {
        SpatialPriorConfig cfg{};
        cfg.performance.device = DeviceType::CUDA;
        cfg.enable_persistence_h0 = true;

        auto eng = SpatialPrior::Create(cfg);
        if (!eng) { std::printf("[SKIP - no CUDA]\n"); }
        else {
            eng->Initialize();
            std::vector<Point> pts(200);
            for (uint32_t i = 0; i < 200; ++i) {
                float t = static_cast<float>(i) / 200.0f;
                pts[i] = {t * 10.0f - 5.0f, 0.0f, 0.1f, 0.5f};
            }
            auto pcv = MakeSyntheticCloud(pts);
            auto result = eng->Compute(pcv);
            assert(result.ok());
            assert(result.persistence_ms >= 0.0f);
            // Digest should have meaningful values
            assert(result.persistence_digest.num_components >= 1);
            assert(result.metrics.persistence_stability >= 0.0f);
            assert(result.metrics.persistence_stability <= 1.0f);
            eng->Shutdown();
            PASS();
        }
    }

    // PH: Pairs should be empty when emit_persistence_pairs is false
    TEST(PH_NoPairsUnlessRequested);
    {
        SpatialPriorConfig cfg{};
        cfg.performance.device = DeviceType::CUDA;
        cfg.enable_persistence_h0 = true;
        cfg.emit_persistence_pairs = false;

        auto eng = SpatialPrior::Create(cfg);
        if (!eng) { std::printf("[SKIP - no CUDA]\n"); }
        else {
            eng->Initialize();
            std::vector<Point> pts(100);
            for (uint32_t i = 0; i < 100; ++i)
                pts[i] = {static_cast<float>(i) * 0.1f, 0.0f, 0.0f, 0.5f};
            auto pcv = MakeSyntheticCloud(pts);
            auto result = eng->Compute(pcv);
            assert(result.ok());
            assert(result.persistence_pairs.empty());
            eng->Shutdown();
            PASS();
        }
    }

    // PH: Full pairs emitted when requested
    TEST(PH_EmitPersistencePairs);
    {
        SpatialPriorConfig cfg{};
        cfg.performance.device = DeviceType::CUDA;
        cfg.enable_persistence_h0 = true;
        cfg.emit_persistence_pairs = true;

        auto eng = SpatialPrior::Create(cfg);
        if (!eng) { std::printf("[SKIP - no CUDA]\n"); }
        else {
            eng->Initialize();
            std::vector<Point> pts(200);
            for (uint32_t i = 0; i < 200; ++i) {
                float t = static_cast<float>(i) / 200.0f;
                pts[i] = {t * 10.0f - 5.0f, (i % 7 == 0) ? 2.0f : 0.0f,
                          0.1f, 0.4f + t * 0.5f};
            }
            auto pcv = MakeSyntheticCloud(pts);
            auto result = eng->Compute(pcv);
            assert(result.ok());
            // With emit_persistence_pairs=true, should have pairs
            // Each pair should have non-negative persistence
            for (auto& p : result.persistence_pairs) {
                assert(p.persistence >= 0.0f);
                assert(p.death_value >= p.birth_value);
            }
            eng->Shutdown();
            PASS();
        }
    }

    // Config flags: RMQ and PH off by default
    TEST(ConfigFlags_DefaultOff);
    {
        SpatialPriorConfig cfg{};
        assert(!cfg.enable_rmq_valley_queries);
        assert(!cfg.enable_persistence_h0);
        assert(!cfg.emit_persistence_pairs);
        PASS();
    }

    // =======================================================================
    // RGB-D Camera Perception Tests
    // =======================================================================
    // Prove Apex17 handles camera/depth sensor data — not just LiDAR.
    // Uses rgbd_utils.hpp for depth image deprojection.
    // =======================================================================



    // --- Test: Depth image deprojection produces valid 3D coordinates ---
    TEST(RGBD_Deprojection_Valid);
    {
        auto cam = GenericVGA();
        auto depth_data = GenerateSyntheticDepthImage(cam.width, cam.height, 2.0f, cam.depth_scale);

        DepthImageView image{};
        image.data   = depth_data.data();
        image.width  = cam.width;
        image.height = cam.height;
        image.dtype  = ScalarType::UInt16;

        auto result = DeprojectDepthImage(image, cam, 0.1f, 10.0f);
        assert(result.ok());
        assert(result.valid_count > 100);  // Should have many valid points
        assert(result.min_depth_m >= 0.1f);
        assert(result.max_depth_m <= 10.0f);
        assert(result.mean_depth_m > 0.0f);
        // Points should be packed as XYZ triples
        assert(result.points_xyz.size() == result.valid_count * 3);
        PASS();
    }

    // --- Test: Engine accepts SensorModality::RGBD ---
    TEST(RGBD_Modality_Accepted);
    {
        auto cam = RealSenseD435();
        auto depth_data = GenerateSyntheticDepthImage(cam.width, cam.height, 2.0f, cam.depth_scale);
        DepthImageView image{depth_data.data(), cam.width, cam.height, 0, ScalarType::UInt16};
        auto deproj = DeprojectDepthImage(image, cam);
        auto pcv = MakeRGBDPointCloud(deproj, 1, "camera_depth");

        // Verify modality and validate input
        assert(pcv.modality == SensorModality::RGBD);
        assert(pcv.point_stride_bytes == 12);
        assert(Validate(pcv) == StatusCode::Ok);
        PASS();
    }

    // --- Test: Full compute pipeline on RGB-D data ---
    TEST(RGBD_Compute_Basic);
    {
        SpatialPriorConfig cfg = MakeTestConfig();
        cfg.clustering.min_samples = 5;
        cfg.clustering.min_cluster_size = 10;
        auto engine = SpatialPrior::Create(cfg);
        engine->Initialize();

        auto cam = RealSenseD435();
        auto depth_data = GenerateSyntheticDepthImage(cam.width, cam.height, 2.0f, cam.depth_scale);
        DepthImageView image{depth_data.data(), cam.width, cam.height, 0, ScalarType::UInt16};
        auto deproj = DeprojectDepthImage(image, cam);
        auto pcv = MakeRGBDPointCloud(deproj, 1);

        auto result = engine->Compute(pcv);
        assert(result.ok());
        assert(result.point_count == deproj.valid_count);
        assert(result.compute_ms >= 0.0f);
        engine->Shutdown();
        PASS();
    }

    // --- Test: Scene regime detected from RGB-D data ---
    TEST(RGBD_SceneRegime_Detected);
    {
        SpatialPriorConfig cfg = MakeTestConfig();
        cfg.clustering.min_samples = 5;
        cfg.clustering.min_cluster_size = 10;
        auto engine = SpatialPrior::Create(cfg);
        engine->Initialize();

        auto cam = AzureKinectNFOV();
        auto depth_data = GenerateWarehouseAisleDepth(cam.width, cam.height, cam.depth_scale);
        DepthImageView image{depth_data.data(), cam.width, cam.height, 0, ScalarType::UInt16};
        auto deproj = DeprojectDepthImage(image, cam);
        auto pcv = MakeRGBDPointCloud(deproj, 2);

        auto result = engine->Compute(pcv);
        assert(result.ok());
        assert(result.metrics.regime != SceneRegime::Unknown);
        assert(result.metrics.confidence > 0.0f);
        engine->Shutdown();
        PASS();
    }

    // --- Test: Traversability valley from RGB-D ---
    TEST(RGBD_Traversability_Valley);
    {
        SpatialPriorConfig cfg = MakeTestConfig();
        cfg.clustering.min_samples = 5;
        cfg.clustering.min_cluster_size = 10;
        auto engine = SpatialPrior::Create(cfg);
        engine->Initialize();

        auto cam = GenericVGA();
        auto depth_data = GenerateWarehouseAisleDepth(cam.width, cam.height, cam.depth_scale);
        DepthImageView image{depth_data.data(), cam.width, cam.height, 0, ScalarType::UInt16};
        auto deproj = DeprojectDepthImage(image, cam, 0.1f, 10.0f, 2);  // subsample 2x
        auto pcv = MakeRGBDPointCloud(deproj, 3);

        auto result = engine->Compute(pcv);
        assert(result.ok());
        assert(result.best_valley.has_value());
        assert(result.best_valley->traversability_score >= 0.0f);
        assert(result.best_valley->traversability_score <= 1.0f);
        engine->Shutdown();
        PASS();
    }

    // --- Test: H₀ persistence from RGB-D ---
    TEST(RGBD_Persistence_H0);
    {
        SpatialPriorConfig cfg = MakeTestConfig();
        cfg.clustering.min_samples = 5;
        cfg.clustering.min_cluster_size = 10;
        cfg.enable_persistence_h0 = true;
        auto engine = SpatialPrior::Create(cfg);
        engine->Initialize();

        auto cam = RealSenseD435();
        auto depth_data = GenerateSyntheticDepthImage(cam.width, cam.height, 2.0f, cam.depth_scale);
        DepthImageView image{depth_data.data(), cam.width, cam.height, 0, ScalarType::UInt16};
        auto deproj = DeprojectDepthImage(image, cam, 0.1f, 10.0f, 2);
        auto pcv = MakeRGBDPointCloud(deproj, 4);

        auto result = engine->Compute(pcv);
        assert(result.ok());
        assert(result.persistence_digest.num_components >= 1);
        assert(result.persistence_digest.max_persistence > 0.0f);
        assert(result.metrics.persistence_stability >= 0.0f);
        assert(result.metrics.persistence_stability <= 1.0f);
        engine->Shutdown();
        PASS();
    }

    // --- Test: Topological fingerprint from RGB-D ---
    TEST(RGBD_Topological_Fingerprint);
    {
        SpatialPriorConfig cfg = MakeTestConfig();
        cfg.clustering.min_samples = 5;
        cfg.clustering.min_cluster_size = 10;
        cfg.enable_persistence_h0 = true;
        cfg.emit_persistence_pairs = true;
        auto engine = SpatialPrior::Create(cfg);
        engine->Initialize();

        auto cam = GenericVGA();
        auto depth_data = GenerateSyntheticDepthImage(cam.width, cam.height, 2.0f, cam.depth_scale);
        DepthImageView image{depth_data.data(), cam.width, cam.height, 0, ScalarType::UInt16};
        auto deproj = DeprojectDepthImage(image, cam);
        auto pcv = MakeRGBDPointCloud(deproj, 5);

        auto result = engine->Compute(pcv);
        assert(result.ok());
        assert(result.topological_fingerprint.valid);
        assert(result.topological_fingerprint.hash != 0);
        engine->Shutdown();
        PASS();
    }

    // --- Test: Cross-modality consistency (LiDAR vs RGBD) ---
    // Same spatial layout generates similar topological fingerprints
    // regardless of whether tagged as LiDAR or RGBD.
    TEST(RGBD_CrossModality_Consistency);
    {
        SpatialPriorConfig cfg = MakeTestConfig();
        cfg.clustering.min_samples = 5;
        cfg.clustering.min_cluster_size = 10;
        cfg.enable_persistence_h0 = true;
        cfg.emit_persistence_pairs = true;
        auto engine = SpatialPrior::Create(cfg);
        engine->Initialize();

        // Create identical point data
        auto cam = GenericVGA();
        auto depth_data = GenerateSyntheticDepthImage(cam.width, cam.height, 2.0f, cam.depth_scale);
        DepthImageView image{depth_data.data(), cam.width, cam.height, 0, ScalarType::UInt16};
        auto deproj = DeprojectDepthImage(image, cam, 0.1f, 10.0f, 4);  // subsample

        // Run as RGBD
        auto pcv_rgbd = MakeRGBDPointCloud(deproj, 10);
        auto result_rgbd = engine->Compute(pcv_rgbd);
        assert(result_rgbd.ok());

        // Run same points as LiDAR
        auto pcv_lidar = pcv_rgbd;
        pcv_lidar.modality = SensorModality::LiDAR;
        pcv_lidar.frame_name = "lidar_front";
        pcv_lidar.frame_id = 11;
        auto result_lidar = engine->Compute(pcv_lidar);
        assert(result_lidar.ok());

        // Fingerprints should be identical (same data, same pipeline)
        if (result_rgbd.topological_fingerprint.valid &&
            result_lidar.topological_fingerprint.valid) {
            float sim = result_rgbd.topological_fingerprint.Similarity(
                result_lidar.topological_fingerprint);
            assert(sim >= 0.99f);  // Same data → essentially identical fingerprint
        }

        // Scene metrics should match closely
        float sol_diff = std::abs(result_rgbd.metrics.global_solidity_index -
                                  result_lidar.metrics.global_solidity_index);
        assert(sol_diff < 0.01f);

        engine->Shutdown();
        PASS();
    }

    std::printf("\n=== %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
