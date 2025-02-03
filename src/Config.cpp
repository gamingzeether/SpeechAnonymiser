#include "Config.h"

#include <filesystem>

Config::Config(const std::string& path, int version) {
	this->path = path;
	this->version = version;
	defaultCfg.open("", version); // Open new blank json
}

bool Config::matchesDefault() {
	return Config::_matches(defaultCfg.getRoot(), json.getRoot());
}

bool Config::_matches(const JSONHelper::JSONObj& o1, const JSONHelper::JSONObj& o2, bool loose) {
	JSONHelper::Type type = o1.get_type();
	if (type != o2.get_type())
		return false;
	
	bool match = true;
	switch (type) {
	case JSONHelper::INVALID:
		match = false;
		break;
	case JSONHelper::INT:
		match = o1.get_int() == o2.get_int();
		break;
	case JSONHelper::DOUBLE:
		match = o1.get_real() == o2.get_real();
		break;
	case JSONHelper::STRING:
		match = o1.get_string() == o2.get_string();
		break;
	case JSONHelper::BOOL:
		match = o1.get_bool() == o2.get_bool();
		break;
	case JSONHelper::ARRAY:
	{
		int c1 = o1.get_array_size();
		int c2 = o2.get_array_size();
		if (c1 == c2) {
			for (int i = 0; i < c1; i++) {
				if (!Config::_matches(o1[i], o2[i])) {
					match = false;
					break;
				}
			}
		} else {
			match = false;
		}
	}
		break;
	case JSONHelper::OBJECT:
	{
		std::vector<std::string> keys = o1.get_keys();
		for (int i = 0; i < keys.size(); i++) {
			const char* key = keys[i].c_str();
			if (!Config::_matches(o1[key], o2[key])) {
				match = false;
				break;
			}
		}
	}
		break;
	}
	return match;
}

void Config::save() {
	json.save();
}

void Config::load() {
	json.open(path, version);
}

bool Config::loadDefault(const std::string& defaultPath) {
	if (std::filesystem::exists(defaultPath)) {
		return defaultCfg.open(defaultPath, version);
	}
	return false;
}

void Config::saveDefault(const std::string& defaultPath) {
	defaultCfg.filepath() = defaultPath;
	defaultCfg.save();
}
