#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ax::ax_asr_api" for configuration "Release"
set_property(TARGET ax::ax_asr_api APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ax::ax_asr_api PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libax_asr_api.so"
  IMPORTED_SONAME_RELEASE "libax_asr_api.so"
  )

list(APPEND _cmake_import_check_targets ax::ax_asr_api )
list(APPEND _cmake_import_check_files_for_ax::ax_asr_api "${_IMPORT_PREFIX}/lib/libax_asr_api.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
