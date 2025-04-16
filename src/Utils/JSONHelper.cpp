#include "JSONHelper.hpp"

#include <iostream>

JSONHelper::Type JSONHelper::JSONObj::get_type() const {
    if (yyjson_mut_is_int(val)) {
        return Type::INT;
    } else if (yyjson_mut_is_real(val)) {
        return Type::DOUBLE;
    } else if (yyjson_mut_is_str(val)) {
        return Type::STRING;
    } else if (yyjson_mut_is_bool(val)) {
        return Type::BOOL;
    } else if (yyjson_mut_is_arr(val)) {
        return Type::ARRAY;
    } else if (yyjson_mut_is_obj(val)) {
        return Type::OBJECT;
    }
    return Type::INVALID;
}

JSONHelper::JSONObj::JSONObj(yyjson_mut_doc* doc, yyjson_mut_val* value) {
    _doc = doc;
    val = value;
}

JSONHelper::JSONObj JSONHelper::JSONObj::operator[](const char* key) const {
    yyjson_mut_val* obj = yyjson_mut_obj_get(val, key);

    if (obj == NULL) {
        obj = yyjson_mut_obj_add_obj(_doc, val, key);
    }

    return JSONObj(_doc, obj);
}

JSONHelper::JSONObj JSONHelper::JSONObj::operator[](const std::string& key) const {
    return operator[](key.c_str());
}

bool JSONHelper::open(std::string openPath, int version, bool create) {
	path = openPath;

    int jsonVersion = -1;

    bool opened = false;
    if (openPath != "") {
        yyjson_doc* iDoc;
        yyjson_read_err err;
        iDoc = yyjson_read_file(path.c_str(), YYJSON_READ_NOFLAG, NULL, &err);
        if (iDoc == NULL) {
            if (!create)
                std::cout << ("JSON read error (%u): %s at position: %ld\n", err.code, err.msg, err.pos);
        } else {
            yyjson_val* iRoot = yyjson_doc_get_root(iDoc);
            yyjson_val* iVersion = yyjson_obj_get(iRoot, "_version");
            jsonVersion = yyjson_get_int(iVersion);

            if (jsonVersion == version) {
                doc = yyjson_doc_mut_copy(iDoc, NULL);
                root = yyjson_mut_doc_get_root(doc);
                opened = true;
            }
            yyjson_doc_free(iDoc);
        }
    }
    if (!opened && create) {
        doc = yyjson_mut_doc_new(NULL);
        root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_int(doc, root, "_version", version);
    }

    rootObj = JSONObj(doc, root);

    return opened || create;
}

void JSONHelper::close() {
    yyjson_mut_doc_free(doc);
}

void JSONHelper::save() {
    yyjson_mut_write_file(path.c_str(), doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, NULL);
}

JSONHelper::JSONObj JSONHelper::operator[](const char* key) {
    return rootObj[key];
}
