vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_sourceforge(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO coot
    FILENAME "bandicoot-${VERSION}.tar.xz"
    SHA512 4c6b8a47a854f21050b2ba7630d00f6318733498369b54e9eb8b49533714d78b0cab19e70b9cfa8e7f806949a6fedffc8d6c4227cff6c9fd27e615f053223cb7
    PATCHES
        pkgconfig.patch
)

set(REQUIRES_PRIVATE "")
foreach(module IN ITEMS blas)
    if(EXISTS "${CURRENT_INSTALLED_DIR}/lib/pkgconfig/${module}.pc")
        string(APPEND REQUIRES_PRIVATE " ${module}")
    endif()
endforeach()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DFIND_CUDA=false
        -DDEFAULT_BACKEND=CL_BACKEND
        "-DREQUIRES_PRIVATE=${REQUIRES_PRIVATE}"
)

vcpkg_cmake_install()

#vcpkg_cmake_config_fixup(PACKAGE_NAME Bandicoot CONFIG_PATH share/Bandicoot/CMake)
#vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/Bandicoot/BandicootConfig.cmake"
#                    [[include("${CMAKE_CURRENT_LIST_DIR}/BandicootLibraryDepends.cmake")]]
#                    "include(CMakeFindDependencyMacro)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/BandicootLibraryDepends.cmake\")"
#                    )
#vcpkg_fixup_pkgconfig()
vcpkg_copy_pdbs()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

file(GLOB SHARE_ARMADILLO_FILES "${CURRENT_PACKAGES_DIR}/share/Bandicoot/*")
if(SHARE_BANDICOOT_FILES STREQUAL "")
    # On case sensitive file system there is an extra empty directory created that should be removed
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/Bandicoot")
endif()

vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/include/bandicoot_bits/config.hpp" "#define COOT_AUX_LIBS " "#define COOT_AUX_LIBS //")

file(COPY "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
#file(COPY "${CMAKE_CURRENT_LIST_DIR}/vcpkg-cmake-wrapper.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
