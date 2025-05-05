#include "Util.hpp"

#include <stdexcept>
#include <vector>

std::wstring Util::str2wstr(const std::string& str) {
  return cvt.from_bytes(str);
}

std::string Util::wstr2str(const std::wstring& wstr) {
  return cvt.to_bytes(wstr);
}

size_t Util::customHasher(const std::wstring& str) {
  size_t v = 0;
  for (size_t i = 0; i < str.length(); i++) {
    v = (v << sizeof(wchar_t) * 8) ^ str[i];
  }
  return v;
}

void Util::removeTrailingSlash(std::string& path) {
  char back = path.back();
  size_t length = path.length();
  if (length > 1 && (back == '/' || back == '\\'))
    path.pop_back();
}

std::string Util::leftPad(std::string message, int width, const char padChar) {
  if (width > message.size()) {
    size_t nPad = width - message.size();
    std::string padding = "";
    for (size_t i = 0; i < nPad; i++)
      padding += padChar;
    message = padding + message;
  }
  return message;
}

std::vector<std::string> Util::split(const std::string& input, const char delimiter) {
  std::vector<std::string> parts;
  size_t start = 0;
  size_t end = 0;
  while (end != input.npos) {
    end = input.find(delimiter, start);
    parts.push_back(input.substr(start, end - start));
    start = end + 1;
  }
  return parts;
}

void Util::stripPadding(std::string& input, const char padChar) {
  size_t start = input.find_first_not_of(padChar);
  if (start != input.npos)
    input = input.substr(start);
}

bool Util::contains(const std::string& a, const std::string& b) {
  bool terminate = false;
  // Outer loop: starting position
  for (size_t start = 0; start < a.size() && !terminate; start++) {
    bool match = true;
    // Inner loop: compare substrings
    for (size_t nMatch = 0; nMatch < b.size(); nMatch++) {
      size_t aPos = start + nMatch;
      size_t bPos = nMatch;
      // Out of bounds; impossible to contain at this point
      if (aPos >= a.size()) {
        terminate = true;
        match = false;
        break;
      }
      char aChar = a[start + nMatch];
      char bChar = b[nMatch];
      if (aChar != bChar) {
        match = false;
        break;
      }
    }
    if (match)
      return true;
  }
  return false;
}
