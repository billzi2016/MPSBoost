// Atomic filesystem persistence for serialized native model bytes.
//
// Saving uses a same-directory temporary file, fsync, and rename. Loading reads
// a bounded byte vector before handing all format validation to the codec.

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "mpsboost/trainer.hpp"

namespace mpsboost {
namespace {

std::string ErrnoMessage(const char* stage) {
  return std::string(stage) + ": " + std::strerror(errno);
}

void WriteModelBytes(const std::vector<std::uint8_t>& bytes,
                     const std::string& path) {
  if (path.empty()) {
    throw TrainingError("模型保存路径不能为空");
  }
  const std::filesystem::path target(path);
  const std::filesystem::path directory =
      target.parent_path().empty() ? std::filesystem::path(".")
                                   : target.parent_path();
  if (!std::filesystem::is_directory(directory)) {
    throw TrainingError("模型保存目录不存在");
  }
  std::string temporary_pattern =
      (directory / ("." + target.filename().string() + ".mpsboost-XXXXXX"))
          .string();
  std::vector<char> mutable_pattern(temporary_pattern.begin(),
                                    temporary_pattern.end());
  mutable_pattern.push_back('\0');
  const int descriptor = ::mkstemp(mutable_pattern.data());
  if (descriptor < 0) {
    throw TrainingError(ErrnoMessage("创建模型临时文件失败"));
  }
  const std::filesystem::path temporary(mutable_pattern.data());
  try {
    std::size_t written = 0;
    while (written < bytes.size()) {
      const ssize_t count =
          ::write(descriptor, bytes.data() + written, bytes.size() - written);
      if (count <= 0) {
        throw TrainingError(ErrnoMessage("写入模型临时文件失败"));
      }
      written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0) {
      throw TrainingError(ErrnoMessage("同步模型临时文件失败"));
    }
    if (::rename(temporary.c_str(), target.c_str()) != 0) {
      throw TrainingError(ErrnoMessage("原子替换模型文件失败"));
    }
  } catch (...) {
    ::close(descriptor);
    ::unlink(temporary.c_str());
    throw;
  }
}

std::vector<std::uint8_t> ReadModelBytes(const std::string& path) {
  if (path.empty()) {
    throw TrainingError("模型加载路径不能为空");
  }
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > std::numeric_limits<std::size_t>::max()) {
    throw TrainingError("无法读取模型文件大小");
  }
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    throw TrainingError(ErrnoMessage("打开模型文件失败"));
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  std::size_t position = 0;
  while (position < bytes.size()) {
    const ssize_t count =
        ::read(descriptor, bytes.data() + position, bytes.size() - position);
    if (count <= 0) {
      ::close(descriptor);
      throw TrainingError(count == 0 ? "模型文件读取提前结束"
                                     : ErrnoMessage("读取模型文件失败"));
    }
    position += static_cast<std::size_t>(count);
  }
  ::close(descriptor);
  return bytes;
}

}  // namespace

void SaveModelFile(const RegressionModel& model, const std::string& path) {
  WriteModelBytes(SerializeModel(model), path);
}

void SaveModelFile(const MulticlassModel& model, const std::string& path) {
  WriteModelBytes(SerializeModel(model), path);
}

RegressionModel LoadModelFile(const std::string& path) {
  return DeserializeModel(ReadModelBytes(path));
}

MulticlassModel LoadMulticlassModelFile(const std::string& path) {
  return DeserializeMulticlassModel(ReadModelBytes(path));
}

}  // namespace mpsboost
