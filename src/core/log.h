#pragma once

#include <iostream>
#include <string>

namespace rr {

enum class LogLevel {
  Info,
  Warn,
  Error
};

inline void log(LogLevel level, const std::string& message) {
  const char* prefix = "[INFO]";
  std::ostream* out = &std::cout;
  if (level == LogLevel::Warn) {
    prefix = "[WARN]";
  } else if (level == LogLevel::Error) {
    prefix = "[ERROR]";
    out = &std::cerr;
  }
  (*out) << prefix << " " << message << "\n";
}

}  // namespace rr

