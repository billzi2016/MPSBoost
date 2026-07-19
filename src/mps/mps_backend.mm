// MPSBoost Metal backend public lifetime glue.
//
// Intent: keep MpsBackend construction, move, destruction, and diagnostics in a
// small translation unit while compute responsibilities live in focused files.

#include "mps_backend_internal.hpp"

namespace mpsboost {

MpsBackend::MpsBackend(std::string metallib_path)
    : impl_(std::make_unique<Impl>(metallib_path)) {}
MpsBackend::~MpsBackend() = default;
MpsBackend::MpsBackend(MpsBackend&&) noexcept = default;
MpsBackend& MpsBackend::operator=(MpsBackend&&) noexcept = default;

BackendTiming MpsBackend::last_timing() const noexcept { return impl_->timing_; }

}  // namespace mpsboost
