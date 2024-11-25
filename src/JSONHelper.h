#pragma once

#include "common_inc.h"

#include <string>
#include <yyjson.h>

class JSONHelper
{
public:
	// Wrapper for yyjson_mut_val
	class JSONObj {
	public:
		JSONObj operator[](const char* key);
		JSONObj operator[](const std::string& key) { return operator[](key.c_str()); };
		JSONObj operator[](const int index) const { return JSONObj(_doc, yyjson_mut_arr_get(val, index)); };

		// Getters
		int get_int() const { return yyjson_mut_get_int(val); };
		double get_real() const { return yyjson_mut_get_real(val); };
		std::string get_string() const { return yyjson_mut_get_str(val); };
		int get_array_size() const { return yyjson_mut_arr_size(val); };
		bool get_bool() const { return yyjson_mut_get_bool(val); };

		// Setters
		void operator=(int value) { yyjson_mut_set_int(val, value); };
		void operator=(double value) { yyjson_mut_set_real(val, value); };
		void operator=(const std::string& value) { yyjson_mut_set_str(val, value.c_str()); };
		void operator=(bool value) { yyjson_mut_set_bool(val, value); };
		JSONObj add_arr(const char* key) {
			yyjson_mut_val* item = yyjson_mut_obj_add_arr(_doc, val, key);
			return JSONObj(_doc, item);
		};
		JSONObj add_arr(const std::string& key) { return add_arr(key.c_str()); };
		JSONObj append() {
			yyjson_mut_val* item = yyjson_mut_arr_add_obj(_doc, val);
			return JSONObj(_doc, item);
		};

		JSONObj(yyjson_mut_doc* doc, yyjson_mut_val* value);
		JSONObj() {};
	private:
		yyjson_mut_val* val;
		yyjson_mut_doc* _doc;
	};

	// Returns true if successfully opened, false if not or wrong version
	bool open(std::string openPath, int version = -1);
	void close();
	void save();
	JSONObj getRoot() { return rootObj; };

	JSONObj operator[](const char* key);
	JSONObj operator[](const std::string& key) { return operator[](key.c_str()); };

	JSONHelper() {};
private:
	std::string path;
	yyjson_mut_doc* doc;
	yyjson_mut_val* root;
	JSONObj rootObj;
};

