DEBUG := 1

# Compiler
CXX := g++

# Number of processes for MPI
NP := 2

# Compiler flags
CXXFLAGS := -std=c++17 -Wall -Wextra -I/usr/include/eigen3
LDLIBS := -pthread -fopenmp
LDTESTS := -lgtest -lgtest_main -pthread

# Debug mode
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -Og -DDEBUG -DPPGRAD_DEBUG
else
	CXXFLAGS += -O3 -DNDEBUG
endif

# Run command (empty if not MPI)
RUN := 
ifeq ($(CXX), mpic++)
	RUN := mpirun -np $(NP)
	CXXFLAGS += -DUSE_MPI
endif

# Directories
BUILD_DIR = build
SRC_DIR = src
TESTS_DIR = tests
INCLUDE_DIR = include
EXAMPLES_DIR = examples

# Find all source and test files
SRC_FILES = $(wildcard $(SRC_DIR)/**/*.cpp)
TEST_FILES = $(wildcard $(TESTS_DIR)/*.cpp)
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJ_FILES = $(TEST_FILES:$(TESTS_DIR)/%.cpp=$(BUILD_DIR)/%.o)
EXAMPLE_FILES = $(wildcard $(EXAMPLES_DIR)/*.cpp)
EXAMPLE_OBJ_FILES = $(EXAMPLE_FILES:$(EXAMPLES_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Debug print out the files
ifeq ($(DEBUG), 1)
$(info $$SRC_FILES is [${SRC_FILES}])
$(info $$TEST_FILES is [${TEST_FILES}])
$(info $$OBJ_FILES is [${OBJ_FILES}])
$(info $$TEST_OBJ_FILES is [${TEST_OBJ_FILES}])
$(info $$EXAMPLE_FILES is [${EXAMPLE_FILES}])
$(info $$EXAMPLE_OBJ_FILES is [${EXAMPLE_OBJ_FILES}])
endif


# Default target
all: tests

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I $(INCLUDE_DIR) -c $< -o $@ $(LDLIBS) 

# Compile test files
$(BUILD_DIR)/%.o: $(TESTS_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I $(INCLUDE_DIR) -c $< -o $@ $(LDLIBS) 

# Compile example files
$(BUILD_DIR)/%.o: $(EXAMPLES_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I $(INCLUDE_DIR) -c $< -o $@ $(LDLIBS)

# Link tests and run
tests: $(OBJ_FILES) $(TEST_OBJ_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $(BUILD_DIR)/test_suite.out $(LDLIBS) $(LDTESTS)
	./$(BUILD_DIR)/test_suite.out

# Run Doxygen
doxygen:
	doxygen Doxyfile

# Allow running particular example
example_%: $(OBJ_FILES) $(BUILD_DIR)/%.o
	@echo "Running example $@"
	$(CXX) $(CXXFLAGS) $(LDLIBS) $^ -o $(BUILD_DIR)/$@
	$(RUN) ./$(BUILD_DIR)/$@

# Clean up
clean:
	rm -rf $(BUILD_DIR)/*
	rm -rf docs/*