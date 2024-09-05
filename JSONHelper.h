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

		// Getters
		int get_int() { return yyjson_mut_get_int(val); };
		int get_real() { return yyjson_mut_get_real(val); };

		// Setters
		void operator=(int value) { yyjson_mut_set_int(val, value); };
		void operator=(double value) { yyjson_mut_set_real(val, value); };

		JSONObj(yyjson_mut_doc* doc, yyjson_mut_val* value);
		JSONObj() {};
	private:
		yyjson_mut_val* val;
		yyjson_mut_doc* _doc;
	};

	// Returns true if successfully opened, false if not or wrong version
	bool open(const char* openPath, int version = -1);
	void close();
	void save();

	JSONObj operator[](const char* key);

	JSONHelper() {};
private:
	const char* path;
	yyjson_mut_doc* doc;
	yyjson_mut_val* root;
	JSONObj rootObj;
};

