#pragma once

#include "../common_inc.hpp"

#include "JSONHelper.hpp"

class Config {
public:
  Config() {};
  Config(const std::string& path, int version);

  template <typename T>
  Config& setDefault(const char* key, T val) {
    defaultCfg[key] = val;
    return *this;
  };
  bool matchesDefault();

  JSONHelper& defaultObject() { return defaultCfg; };
  JSONHelper& object() { return json; };

  void save();
  void load();
  bool loadDefault(const std::string& defaultPath);
  void saveDefault(const std::string& defaultPath);
  void useDefault() {
    defaultCfg.filepath() = json.filepath();
    json.close();
    json = defaultCfg;
  };
  void close() { json.close(); };
private:
  bool _matches(const JSONHelper::JSONObj& o1, const JSONHelper::JSONObj& o2, bool loose = false);

  std::string path;
  int version;
  JSONHelper json;
  JSONHelper defaultCfg;
};
