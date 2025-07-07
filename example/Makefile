#
# Copyright 2025 HUSTJunshuang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# specify SM version(Compute capability)
SM_VER ?= 89
GENCODE_FLAGS := -gencode arch=compute_$(SM_VER),code=sm_$(SM_VER)

# paths and files
NVPERF_DIR = ..
BUILD_DIR = build

INC_PATH := $(NVPERF_DIR)/include
SRC_PATH := src

SRCS := $(wildcard $(SRC_PATH)/*.cu)
TARGETS := $(patsubst $(SRC_PATH)/%.cu, $(BUILD_DIR)/%, $(SRCS))
BINARYS := $(notdir $(TARGETS))

# compilation flags
INCLUDES = $(addprefix -I, $(INC_PATH))
CFLAGS := $(INCLUDES) $(GENCODE_FLAGS) -lineinfo

# targets
all: $(TARGETS)

$(BUILD_DIR)/%: $(SRC_PATH)/%.cu
	@mkdir -p $(@D)
	nvcc $(CFLAGS) $< -o $@

$(BINARYS): %: $(BUILD_DIR)/%

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean $(BINARYS)