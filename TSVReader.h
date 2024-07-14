#pragma once
#include "define.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

class TSVReader {
private:
	std::ifstream reader;
	std::vector<std::string> columns;
	int column_count;
public:
    enum Indices {
        CLIENT_ID,
        PATH,
        SENTENCE_ID,
        SENTENCE,
        SENTENCE_DOMAIN,
        UP_VOTES,
        DOWN_VOTES,
        AGE,
        GENDER,
        ACCENTS,
        VARIANT,
        LOCALE,
        SEGMENT,
    };

	const std::string get_column_name(const int index) { return columns[index]; };
	const int get_column_count() { return column_count; };

	void open(const char* filepath);
	bool good();
	std::string* read_line();
};
