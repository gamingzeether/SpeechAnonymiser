#pragma once

#include "common_inc.h"

#include "JSONHelper.h"

class Config {
public:
	Config() {};
	Config(const std::string& path, int version);

	template <typename T>
	Config& setDefault(const char* key, T val) {
		defaultCfg[key] = val;
		defaultKeys.push_back(key);
		return *this;
	};
	bool matchesDefault();

	JSONHelper& object() { return json; };

	void save();
	void load();
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
	std::vector<const char*> defaultKeys;
};
