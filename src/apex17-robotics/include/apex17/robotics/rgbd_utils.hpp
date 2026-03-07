#pragma once

#include "apex17/robotics/spatial_prior.hpp"

#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace Apex17::Robotics {

// =============================================================================
// RGB-D Depth Image Utilities
// -----------------------------------------------------------------------------
// Converts raw depth images from RGB-D cameras (Intel RealSense, Azure Kinect,
// ZED 2, etc.) into PointCloudView for consumption by SpatialPrior.
//
// This proves that Apex17's perception pipeline is sensor-agnostic:
// the same OPTICS → H₀ PH → TopologicalFingerprint path processes
// both LiDAR point clouds and camera-derived depth data.
// =============================================================================

// ---------------------------------------------------------------------------
// Camera Intrinsics
// ---------------------------------------------------------------------------

struct CameraIntrinsics {
    float fx{0.0f};          // Focal length X (pixels)
    float fy{0.0f};          // Focal length Y (pixels)
    float cx{0.0f};          // Principal point X (pixels)
    float cy{0.0f};          // Principal point Y (pixels)
    uint32_t width{0};       // Image width (pixels)
    uint32_t height{0};      // Image height (pixels)
    float depth_scale{1.0f}; // Multiplier: raw_value * depth_scale = meters

    // Radial distortion (Brown-Conrady model)
    float k1{0.0f};
    float k2{0.0f};
    float k3{0.0f};
    // Tangential distortion
    float p1{0.0f};
    float p2{0.0f};

    [[nodiscard]] bool valid() const noexcept {
        return fx > 0.0f && fy > 0.0f && width > 0 && height > 0;
    }
};

// ---------------------------------------------------------------------------
// Camera Presets — production-validated intrinsics
// ---------------------------------------------------------------------------

/// Intel RealSense D435 (640×480 depth mode)
[[nodiscard]] inline CameraIntrinsics RealSenseD435() noexcept {
    CameraIntrinsics cam{};
    cam.fx = 382.613f;
    cam.fy = 382.613f;
    cam.cx = 318.693f;
    cam.cy = 236.770f;
    cam.width  = 640;
    cam.height = 480;
    cam.depth_scale = 0.001f;  // mm → meters
    return cam;
}

/// Microsoft Azure Kinect (640×576 NFOV depth mode)
[[nodiscard]] inline CameraIntrinsics AzureKinectNFOV() noexcept {
    CameraIntrinsics cam{};
    cam.fx = 504.206f;
    cam.fy = 504.206f;
    cam.cx = 321.938f;
    cam.cy = 330.782f;
    cam.width  = 640;
    cam.height = 576;
    cam.depth_scale = 0.001f;
    return cam;
}

/// Stereolabs ZED 2 (672×376 depth mode)
[[nodiscard]] inline CameraIntrinsics ZED2() noexcept {
    CameraIntrinsics cam{};
    cam.fx = 350.572f;
    cam.fy = 350.572f;
    cam.cx = 336.843f;
    cam.cy = 189.591f;
    cam.width  = 672;
    cam.height = 376;
    cam.depth_scale = 0.001f;
    return cam;
}

/// Generic 640×480 pinhole (unit tests / synthetic data)
[[nodiscard]] inline CameraIntrinsics GenericVGA() noexcept {
    CameraIntrinsics cam{};
    cam.fx = 525.0f;
    cam.fy = 525.0f;
    cam.cx = 319.5f;
    cam.cy = 239.5f;
    cam.width  = 640;
    cam.height = 480;
    cam.depth_scale = 0.001f;
    return cam;
}

// ---------------------------------------------------------------------------
// Depth Image View
// ---------------------------------------------------------------------------

struct DepthImageView {
    const void* data{nullptr};    // Raw depth buffer
    uint32_t    width{0};
    uint32_t    height{0};
    uint32_t    stride_bytes{0};  // Row stride in bytes (0 = width × element_size)
    ScalarType  dtype{ScalarType::UInt16};  // UInt16 (typical) or Float32

    [[nodiscard]] bool valid() const noexcept {
        return data != nullptr && width > 0 && height > 0;
    }
};

// ---------------------------------------------------------------------------
// Deprojection result
// ---------------------------------------------------------------------------

struct DeprojectionResult {
    std::vector<float> points_xyz;    // Packed [x,y,z, x,y,z, ...] in meters
    uint32_t valid_count{0};          // Points with valid depth
    uint32_t total_pixels{0};         // Total depth pixels
    float    min_depth_m{0.0f};       // Nearest valid point
    float    max_depth_m{0.0f};       // Farthest valid point
    float    mean_depth_m{0.0f};      // Average depth

    [[nodiscard]] bool ok() const noexcept {
        return valid_count > 0 && !points_xyz.empty();
    }
};

// ---------------------------------------------------------------------------
// Core deprojection: depth image → 3D point cloud
// ---------------------------------------------------------------------------

/// Deproject a depth image into a 3D point cloud using pinhole camera model.
/// Filters invalid depths (0, NaN, out of range).
///
/// @param image      Depth image buffer (UInt16 or Float32)
/// @param intrinsics Camera intrinsics (focal length, principal point, scale)
/// @param min_depth  Minimum valid depth in meters (default 0.1m)
/// @param max_depth  Maximum valid depth in meters (default 10.0m)
/// @param subsample  Subsample factor (1 = every pixel, 2 = every other, etc.)
/// @return           DeprojectionResult with packed XYZ points
[[nodiscard]] inline DeprojectionResult DeprojectDepthImage(
    const DepthImageView& image,
    const CameraIntrinsics& intrinsics,
    float min_depth = 0.1f,
    float max_depth = 10.0f,
    uint32_t subsample = 1)
{
    DeprojectionResult result{};
    if (!image.valid() || !intrinsics.valid() || subsample == 0) return result;

    result.total_pixels = image.width * image.height;
    result.points_xyz.reserve(result.total_pixels * 3 / (subsample * subsample));
    result.min_depth_m = max_depth;
    result.max_depth_m = 0.0f;
    double depth_sum = 0.0;

    const float inv_fx = 1.0f / intrinsics.fx;
    const float inv_fy = 1.0f / intrinsics.fy;

    const uint32_t row_stride = (image.stride_bytes > 0)
        ? image.stride_bytes
        : image.width * ((image.dtype == ScalarType::UInt16) ? 2u : 4u);

    for (uint32_t v = 0; v < image.height; v += subsample) {
        const auto* row = reinterpret_cast<const uint8_t*>(image.data) + v * row_stride;

        for (uint32_t u = 0; u < image.width; u += subsample) {
            float depth_m = 0.0f;

            if (image.dtype == ScalarType::UInt16) {
                auto raw = reinterpret_cast<const uint16_t*>(row)[u];
                if (raw == 0) continue;  // Invalid depth
                depth_m = static_cast<float>(raw) * intrinsics.depth_scale;
            } else if (image.dtype == ScalarType::Float32) {
                depth_m = reinterpret_cast<const float*>(row)[u];
            } else {
                continue;  // Unsupported dtype
            }

            // Range filter
            if (depth_m < min_depth || depth_m > max_depth) continue;
            if (std::isnan(depth_m) || std::isinf(depth_m)) continue;

            // Pinhole deprojection: pixel (u,v,d) → 3D (X,Y,Z)
            float x = (static_cast<float>(u) - intrinsics.cx) * depth_m * inv_fx;
            float y = (static_cast<float>(v) - intrinsics.cy) * depth_m * inv_fy;
            float z = depth_m;

            result.points_xyz.push_back(x);
            result.points_xyz.push_back(y);
            result.points_xyz.push_back(z);
            result.valid_count++;

            if (depth_m < result.min_depth_m) result.min_depth_m = depth_m;
            if (depth_m > result.max_depth_m) result.max_depth_m = depth_m;
            depth_sum += depth_m;
        }
    }

    if (result.valid_count > 0) {
        result.mean_depth_m = static_cast<float>(depth_sum / result.valid_count);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Convenience: DeprojectionResult → PointCloudView
// ---------------------------------------------------------------------------

/// Build a PointCloudView from a deprojection result for SpatialPrior::Compute().
/// The returned view borrows from the DeprojectionResult — do not destroy it
/// before passing to the engine.
///
/// Points are packed as [x,y,z] with stride = 12 bytes.
[[nodiscard]] inline PointCloudView MakeRGBDPointCloud(
    const DeprojectionResult& deproj,
    uint64_t frame_id = 0,
    const std::string& camera_frame = "camera_depth_optical_frame")
{
    PointCloudView pcv{};
    pcv.points.data        = deproj.points_xyz.data();
    pcv.points.bytes       = deproj.points_xyz.size() * sizeof(float);
    pcv.points.dtype       = ScalarType::Float32;
    pcv.points.device      = DeviceType::CPU;
    pcv.point_count        = deproj.valid_count;
    pcv.point_stride_bytes = 12;  // 3 × float32 = 12 bytes (XYZ only, no intensity)
    pcv.modality           = SensorModality::RGBD;
    pcv.frame_id           = frame_id;
    pcv.frame_name         = camera_frame;
    pcv.world_frame_name   = "odom";
    pcv.layout.x_offset    = 0;
    pcv.layout.y_offset    = 4;
    pcv.layout.z_offset    = 8;
    return pcv;
}

// ---------------------------------------------------------------------------
// Synthetic depth image generators (for testing)
// ---------------------------------------------------------------------------

/// Generate a synthetic depth image simulating a flat floor at `floor_depth` meters
/// with random objects placed as rectangular depth patches.
[[nodiscard]] inline std::vector<uint16_t> GenerateSyntheticDepthImage(
    uint32_t width, uint32_t height,
    float floor_depth_m = 2.0f,
    float depth_scale = 0.001f,
    uint32_t num_objects = 5)
{
    std::vector<uint16_t> depth(width * height);
    uint16_t floor_raw = static_cast<uint16_t>(floor_depth_m / depth_scale);

    // Fill with floor
    for (auto& d : depth) d = floor_raw;

    // Place synthetic objects at various depths
    struct SynObj { uint32_t x, y, w, h; float depth_m; };
    SynObj objects[] = {
        {100, 100, 60, 80, 1.2f},   // near object
        {300, 200, 40, 50, 0.8f},   // very near (occluding)
        {450, 350, 70, 60, 1.5f},   // mid-range
        {200, 300, 50, 40, 3.0f},   // behind floor (should clip)
        {500, 100, 80, 90, 0.5f},   // close obstacle
    };

    for (uint32_t i = 0; i < num_objects && i < 5; ++i) {
        auto& obj = objects[i];
        uint16_t raw = static_cast<uint16_t>(obj.depth_m / depth_scale);
        for (uint32_t r = obj.y; r < obj.y + obj.h && r < height; ++r) {
            for (uint32_t c = obj.x; c < obj.x + obj.w && c < width; ++c) {
                depth[r * width + c] = raw;
            }
        }
    }

    // Add some invalid pixels (0 = no depth)
    for (uint32_t i = 0; i < width * 2; ++i) {
        depth[i] = 0;  // Top rows invalid (sky/ceiling)
    }

    return depth;
}

/// Generate a structured indoor scene depth image (warehouse aisle).
/// Two side walls converge to a back wall — typical AMR perception scenario.
[[nodiscard]] inline std::vector<uint16_t> GenerateWarehouseAisleDepth(
    uint32_t width, uint32_t height,
    float depth_scale = 0.001f)
{
    std::vector<uint16_t> depth(width * height);
    float aisle_length_m = 8.0f;
    float aisle_width_m  = 3.0f;

    for (uint32_t v = 0; v < height; ++v) {
        for (uint32_t u = 0; u < width; ++u) {
            float nv = static_cast<float>(v) / static_cast<float>(height);
            float nu = static_cast<float>(u) / static_cast<float>(width);

            float d_m;
            // Floor region (bottom of image)
            if (nv > 0.65f) {
                d_m = 1.5f + (1.0f - nv) * 3.0f;  // floor receding
            }
            // Left wall
            else if (nu < 0.15f) {
                d_m = 1.0f + nu * 10.0f;
            }
            // Right wall
            else if (nu > 0.85f) {
                d_m = 1.0f + (1.0f - nu) * 10.0f;
            }
            // Back wall (center, distance)
            else if (nv < 0.3f) {
                d_m = aisle_length_m;
            }
            // Open aisle
            else {
                d_m = aisle_length_m * (1.0f - nv * 0.5f);
            }

            d_m = std::max(0.1f, std::min(10.0f, d_m));
            depth[v * width + u] = static_cast<uint16_t>(d_m / depth_scale);
        }
    }

    return depth;
}

// ---------------------------------------------------------------------------
// Camera name helper
// ---------------------------------------------------------------------------

[[nodiscard]] inline const char* CameraPresetName(const CameraIntrinsics& cam) noexcept {
    if (cam.width == 640 && cam.height == 480 && std::abs(cam.fx - 382.613f) < 1.0f)
        return "Intel RealSense D435";
    if (cam.width == 640 && cam.height == 576 && std::abs(cam.fx - 504.206f) < 1.0f)
        return "Microsoft Azure Kinect";
    if (cam.width == 672 && cam.height == 376 && std::abs(cam.fx - 350.572f) < 1.0f)
        return "Stereolabs ZED 2";
    if (cam.width == 640 && cam.height == 480 && std::abs(cam.fx - 525.0f) < 1.0f)
        return "Generic VGA Pinhole";
    return "Custom";
}

} // namespace Apex17::Robotics
