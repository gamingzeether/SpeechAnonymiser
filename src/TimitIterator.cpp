#include "TimitIterator.hpp"

void TimitIterator::open(const std::string& path) {
    std::filesystem::recursive_directory_iterator iter(path);
    for (const auto& item : iter) {
        if (item.is_regular_file()) {
            auto path = item.path();
            if (path.extension() == ".PHN") {
                paths.push_back(path);
            }
        }
    }
}

std::filesystem::path TimitIterator::next() {
    return paths[pointer++];
}

bool TimitIterator::good() {
    return pointer < paths.size();
}
