cmake_minimum_required(VERSION 3.12)

project(PINNTurbulentChannelFlow VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFOAM
find_path(OPENFOAM_ROOT_DIR NAMES etc/bashrc
  HINTS
  $ENV{WM_PROJECT_DIR}
  $ENV{FOAM_INST_DIR}/OpenFOAM-9
  $ENV{FOAM_INST_DIR}/OpenFOAM-8
  $ENV{FOAM_INST_DIR}/OpenFOAM-7
  /opt/openfoam9
  /opt/openfoam8
  /opt/openfoam7
  DOC "OpenFOAM root directory"
)

if(OPENFOAM_ROOT_DIR)
  message(STATUS "Found OpenFOAM: ${OPENFOAM_ROOT_DIR}")
else()
  message(WARNING "OpenFOAM not found. OpenFOAM integration will be disabled.")
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find PyTorch
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
  find_package(Torch REQUIRED)
  message(STATUS "Found PyTorch: ${TORCH_CMAKE_PREFIX_PATH}")
endif()

# Create custom targets for different operations

# Target for setting up the environment
add_custom_target(setup
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/output
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/logs
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
  COMMENT "Setting up project directories"
)

# Target for running OpenFOAM simulation
if(OPENFOAM_ROOT_DIR)
  add_custom_target(run-openfoam
    COMMAND bash -c "source ${OPENFOAM_ROOT_DIR}/etc/bashrc && cd ${CMAKE_SOURCE_DIR}/openfoam/channelKEpsilon && ./run_simulation.sh"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/openfoam/channelKEpsilon
    COMMENT "Running OpenFOAM simulation"
    DEPENDS setup
  )
endif()

# Target for running PINN training
add_custom_target(train-pinn
  COMMAND ${Python3_EXECUTABLE} -c "
from src import SimulationConfig, PINNRANSModel
config = SimulationConfig(base_dir='${CMAKE_SOURCE_DIR}')
config.create_directories()
model = PINNRANSModel(config)
model.compile_model()
model.train_model()
"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Training PINN model"
  DEPENDS setup
)

# Target for generating plots
add_custom_target(generate-plots
  COMMAND ${Python3_EXECUTABLE} -c "
from src import SimulationConfig, PINNRANSModel, PINNPlotter
import os
config = SimulationConfig(base_dir='${CMAKE_SOURCE_DIR}')
# Assuming model is already trained
try:
    model = PINNRANSModel(config)
    # Load trained model if available
    plotter = PINNPlotter(config)
    plotter.create_all_plots(model.model, '${CMAKE_SOURCE_DIR}/output/plots')
except Exception as e:
    print(f'Error generating plots: {e}')
    print('Make sure the model is trained first using: make train-pinn')
"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Generating visualization plots"
  DEPENDS setup
)

# Target for cleaning outputs
add_custom_target(clean-outputs
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/output
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/logs
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/output
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/logs
  COMMENT "Cleaning output directories"
)

# Target for running tests
add_custom_target(test
  COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests -v
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running tests"
)

# Installation targets
install(DIRECTORY src/
  DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/pinn_channel_flow/
  FILES_MATCHING PATTERN "*.py"
)

install(FILES requirements.txt setup.py pyproject.toml
  DESTINATION share/pinn_channel_flow/
)

# Package configuration
set(CPACK_PACKAGE_NAME "PINNTurbulentChannelFlow")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PINN for turbulent channel flow simulation")
set(CPACK_PACKAGE_VENDOR "PINN Research Team")

include(CPack)
