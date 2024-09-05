#include "JSONHelper.h"

JSONHelper::JSONObj::JSONObj(yyjson_mut_doc* doc, yyjson_mut_val* value) {
    _doc = doc;
    val = value;
}

JSONHelper::JSONObj JSONHelper::JSONObj::operator[](const char* key) {
    yyjson_mut_val* obj = yyjson_mut_obj_get(val, key);

    if (obj == NULL) {
        obj = yyjson_mut_obj_add_obj(_doc, val, key);
    }

    return JSONObj(_doc, obj);
}

bool JSONHelper::open(const char* openPath, int version) {
	path = openPath;

    yyjson_doc* iDoc;
    int jsonVersion = -1;

    iDoc = yyjson_read_file(path, YYJSON_READ_NOFLAG, NULL, NULL);
    if (iDoc != NULL) {
        yyjson_val* root = yyjson_doc_get_root(iDoc);
        yyjson_val* version = yyjson_obj_get(root, "_version");
        jsonVersion = yyjson_get_int(version);
    }
    if (jsonVersion == version) {
        doc = yyjson_doc_mut_copy(iDoc, NULL);
        root = yyjson_mut_doc_get_root(doc);
    } else {
        doc = yyjson_mut_doc_new(NULL);
        root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_int(doc, root, "_version", version);
    }
    yyjson_doc_free(iDoc);

    rootObj = JSONObj(doc, root);

    return jsonVersion == version;
}

void JSONHelper::close() {
    yyjson_mut_doc_free(doc);
}

void JSONHelper::save() {
    yyjson_mut_write_file(path, doc, YYJSON_WRITE_NOFLAG, NULL, NULL);
}

JSONHelper::JSONObj JSONHelper::operator[](const char* key) {
    return rootObj[key];
}
