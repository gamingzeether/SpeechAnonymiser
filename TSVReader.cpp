#include "TSVReader.h"

#include <iostream>
#include <sstream>
#include "structs.h"

void TSVReader::open(const std::string& filepath) {
	if (reader.is_open()) {
		reader.close();
	}
	reader.open(filepath);
	if (reader.is_open()) {
		char c;
		std::string column = "";
		column_count = 0;
		while (reader) {
			c = reader.get();
			switch (c) {
			case '\t':
				columns.push_back(column);
				column = "";
				column_count++;
				break;
			case '\n':
				columns.push_back(column);
				column_count++;
				return;
			default:
				column += c;
				break;
			}
		}
	} else {
		std::cout << "Failed to open file at " << filepath << std::endl;
		throw("Failed to open file");
	}
}

bool TSVReader::good() {
	char c = reader.peek();
	return c != '\n' && c != EOF;
};

std::string* TSVReader::read_line() {
	std::string* elements = new std::string[column_count];
	std::string line;
	std::getline(reader, line);
	size_t last = 0, next;
	for (int i = 0; i < column_count; i++) {
		next = line.find('\t', last);
		elements[i] = line.substr(last, next - last);
		last = next + 1;
	}
	return elements;
}
