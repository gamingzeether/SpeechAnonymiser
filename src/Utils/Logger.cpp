#include "Logger.hpp"

#include <chrono>
#include <time.h>
#include <mutex>
#include <filesystem>

void Logger::Stream::init() {
  if (fstreamPath != "") {
    std::string logDir = fileName(STRINGIFY(LOG_DIR));
    stream = new std::ofstream(logDir, std::ios::app);
    if (!((std::ofstream*)stream)->is_open()) {
      std::printf("Failed to open file for %s\n", logDir.c_str());
    }
  }
  outputVerbose = std::vector<bool>(Logger::verbosityLevels, false);
  for (int i : outputVerboseIndex) {
    outputVerbose[i] = true;
  }
}

void Logger::Stream::operator<<(const std::string& str) {
  std::unique_lock<std::mutex> lock = (mutex) ?
    std::unique_lock<std::mutex>(*mutex)
    : std::unique_lock<std::mutex>();

  *stream << str;
  std::flush(*stream);
}

Logger::Stream& Logger::Stream::operator=(const Logger::Stream& other) {
  fstreamPath = other.fstreamPath;
  stream = other.stream;
  outputVerboseIndex = other.outputVerboseIndex;
  outputVerbose = other.outputVerbose;
  mutex = other.mutex;
  return *this;
}

void Logger::log(const std::string& message, int verbosity, int color) {
  assert(verbosity <= verbosityLevels);
  assert(streams.size() > 0);

  if (color == Color::DEFAULT) {
    switch (verbosity) {
      case Verbosity::WARN:
        color = Color::YELLOW;
        break;
      case Verbosity::ERRO:
        color = Color::RED;
        break;
      case Verbosity::DEAD:
        color = Color::RED;
        break;
    }
  }

  std::string time = timeString();
  for (size_t s = 0; s < streams.size(); s++) {
    Stream& stream = streams[s];
    if (stream.outputsTo(verbosity)) {
      std::string& verbosityName = verbosityNames[verbosity];
      bool useColor = stream.supportsColor() && color != Color::DEFAULT;
      std::string& colorStartStr = colors[useColor ? color : Color::NONE];
      std::string& colorEndStr = colors[useColor ? Color::DEFAULT : Color::NONE];
      std::string logLine = Util::format("%s%s %s:  %s%s\n", colorStartStr.c_str(), time.c_str(), verbosityName.c_str(), message.c_str(), colorEndStr.c_str());
      stream << logLine;
    }
  }
}

std::string Logger::fileName(const std::string& base) {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);

  tm lt = *std::localtime(&time);
  return Util::format("%s%04d_%02d_%02d-%02d_%02d-%s", STRINGIFY(LOG_DIR), lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday, lt.tm_hour, lt.tm_min, base.c_str());
}

std::string Logger::timeString() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  
  tm lt = *std::localtime(&time);
  return Util::format("[%02d:%02d:%02d]", lt.tm_hour, lt.tm_min, lt.tm_sec);
}
