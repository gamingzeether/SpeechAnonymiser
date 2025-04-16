#pragma once

#include "../common_inc.hpp"

#include <string>
#include <vector>
#include <yyjson.h>

class JSONHelper
{
public:
	enum Type {
		INVALID,
		INT,
		DOUBLE,
		STRING,
		BOOL,
		ARRAY,
		OBJECT,
	};
	// Wrapper for yyjson_mut_val
	class JSONObj {
	public:
		JSONObj operator[](const char* key) const;
		JSONObj operator[](const std::string& key) const;
		JSONObj operator[](const int index) const { return JSONObj(_doc, yyjson_mut_arr_get(val, index)); };

		// Getters
		int get_int() const { return yyjson_mut_get_int(val); };
		double get_real() const { return yyjson_mut_get_real(val); };
		std::string get_string() const { return std::string(yyjson_mut_get_str(val)); };
		size_t get_array_size() const { return yyjson_mut_arr_size(val); };
		bool get_bool() const { return yyjson_mut_get_bool(val); };
		std::vector<std::string> get_keys() const {
			std::vector<std::string> keys;
			auto iterator = yyjson_mut_obj_iter_with(val);
			yyjson_mut_val* key;
			while (key = yyjson_mut_obj_iter_next(&iterator)) {
				keys.push_back(yyjson_mut_get_str(key));
			}
			return keys;
		};

		Type get_type() const;
		bool exists(const std::string& key) const { return (NULL != yyjson_mut_obj_get(val, key.c_str())); };

		// Setters
		void operator=(int value) { yyjson_mut_set_int(val, value); };
		void operator=(double value) { yyjson_mut_set_real(val, value); };
		void operator=(const std::string& value) {
			// Results in unfreed memory but allows writing even after value is freed
			// There is a better way to do this but I would have to rewrite a lot of things
			char* copy = (char*)malloc(sizeof(char) * value.length());
			strcpy(copy, value.c_str());
			yyjson_mut_set_str(val, copy);
		};
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
		void clear_arr() {
			yyjson_mut_arr_clear(val);
		};

		JSONObj(yyjson_mut_doc* doc, yyjson_mut_val* value);
		JSONObj() {};
	private:
		yyjson_mut_val* val;
		yyjson_mut_doc* _doc;
	};

	// Returns true if successfully opened, false if not or wrong version
	bool open(std::string openPath, int version = -1, bool create = true);
	void close();
	void save();
	JSONObj getRoot() const { return rootObj; };
	std::string& filepath() { return path; };

	JSONObj operator[](const char* key);

	JSONHelper() {};
private:
	std::string path;
	yyjson_mut_doc* doc;
	yyjson_mut_val* root;
	JSONObj rootObj;
};

