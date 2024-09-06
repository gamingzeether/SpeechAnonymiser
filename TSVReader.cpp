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
		bool done = false;
		while (reader && !done) {
			c = reader.get();
			switch (c) {
			case '\t':
				columns.push_back(column);
				column = "";
				column_count++;
				break;
			case '\n':
				columns.push_back(column);
				done = true;
				column_count++;
				break;
			default:
				column += c;
				break;
			}
		}
	} else {
		printf("Failed to open file at %s\n", filepath.c_str());
		throw("Failed to open file");
	}

	printf("Loading TSV: %s\n", filepath.c_str());
	std::string line;
	while (std::getline(reader, line)) {
		std::string* elements = new std::string[3];
		size_t last = 0, next;
		size_t col = 0;
		for (int i = 0; i < column_count; i++) {
			next = line.find('\t', last);
			if (i == CLIENT_ID || 
				i == PATH) {
				elements[col] = line.substr(last, next - last);
				col++;
			}
			last = next + 1;
		}
		lines.push_back(elements);
	}
	printf("Loaded %zu lines\n", lines.size());
	reader.close();
}

std::string* TSVReader::read_line() {
	return lines[rand() % lines.size()];
}

std::string* TSVReader::read_line_ordered() {
	if (readLine <= lines.size()) {
		return lines[readLine];
	}
	return NULL;
}
