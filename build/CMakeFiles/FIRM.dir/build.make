# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/luigi/cmake/bin/cmake

# The command to remove a file.
RM = /home/luigi/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luigi/projects/FIRM/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luigi/projects/FIRM/build

# Include any dependencies generated for this target.
include CMakeFiles/FIRM.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FIRM.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FIRM.dir/flags.make

CMakeFiles/FIRM.dir/main.cpp.o: CMakeFiles/FIRM.dir/flags.make
CMakeFiles/FIRM.dir/main.cpp.o: /home/luigi/projects/FIRM/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luigi/projects/FIRM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FIRM.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FIRM.dir/main.cpp.o -c /home/luigi/projects/FIRM/src/main.cpp

CMakeFiles/FIRM.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FIRM.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luigi/projects/FIRM/src/main.cpp > CMakeFiles/FIRM.dir/main.cpp.i

CMakeFiles/FIRM.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FIRM.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luigi/projects/FIRM/src/main.cpp -o CMakeFiles/FIRM.dir/main.cpp.s

# Object files for target FIRM
FIRM_OBJECTS = \
"CMakeFiles/FIRM.dir/main.cpp.o"

# External object files for target FIRM
FIRM_EXTERNAL_OBJECTS =

FIRM: CMakeFiles/FIRM.dir/main.cpp.o
FIRM: CMakeFiles/FIRM.dir/build.make
FIRM: CMakeFiles/FIRM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luigi/projects/FIRM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FIRM"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FIRM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FIRM.dir/build: FIRM

.PHONY : CMakeFiles/FIRM.dir/build

CMakeFiles/FIRM.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FIRM.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FIRM.dir/clean

CMakeFiles/FIRM.dir/depend:
	cd /home/luigi/projects/FIRM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luigi/projects/FIRM/src /home/luigi/projects/FIRM/src /home/luigi/projects/FIRM/build /home/luigi/projects/FIRM/build /home/luigi/projects/FIRM/build/CMakeFiles/FIRM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FIRM.dir/depend

