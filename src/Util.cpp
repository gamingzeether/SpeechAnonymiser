#include "Util.hpp"

#include <stdexcept>
#include <vector>

// https://stackoverflow.com/a/7154226
std::wstring Util::utf8_to_utf16(const std::string& utf8) {
    std::vector<unsigned long> unicode;
    size_t i = 0;
    while (i < utf8.size())
    {
        unsigned long uni;
        size_t todo;
        bool error = false;
        unsigned char ch = utf8[i++];
        if (ch <= 0x7F)
        {
            uni = ch;
            todo = 0;
        } else if (ch <= 0xBF)
        {
            throw std::logic_error("not a UTF-8 string");
        } else if (ch <= 0xDF)
        {
            uni = ch & 0x1F;
            todo = 1;
        } else if (ch <= 0xEF)
        {
            uni = ch & 0x0F;
            todo = 2;
        } else if (ch <= 0xF7)
        {
            uni = ch & 0x07;
            todo = 3;
        } else
        {
            throw std::logic_error("not a UTF-8 string");
        }
        for (size_t j = 0; j < todo; ++j)
        {
            if (i == utf8.size())
                throw std::logic_error("not a UTF-8 string");
            unsigned char ch = utf8[i++];
            if (ch < 0x80 || ch > 0xBF)
                throw std::logic_error("not a UTF-8 string");
            uni <<= 6;
            uni += ch & 0x3F;
        }
        if (uni >= 0xD800 && uni <= 0xDFFF)
            throw std::logic_error("not a UTF-8 string");
        if (uni > 0x10FFFF)
            throw std::logic_error("not a UTF-8 string");
        unicode.push_back(uni);
    }
    std::wstring utf16;
    for (size_t i = 0; i < unicode.size(); ++i)
    {
        unsigned long uni = unicode[i];
        if (uni <= 0xFFFF)
        {
            utf16 += (wchar_t)uni;
        } else
        {
            uni -= 0x10000;
            utf16 += (wchar_t)((uni >> 10) + 0xD800);
            utf16 += (wchar_t)((uni & 0x3FF) + 0xDC00);
        }
    }
    return utf16;
}

size_t Util::customHasher(const std::wstring& str) {
    size_t v = 0;
    for (size_t i = 0; i < str.length(); i++) {
        v = (v << sizeof(wchar_t) * 8) ^ str[i];
    }
    return v;
}

void Util::removeTrailingSlash(std::string& path) {
    char back = path.back();
    size_t length = path.length();
    if (length > 1 && (back == '/' || back == '\\'))
        path.pop_back();
}
