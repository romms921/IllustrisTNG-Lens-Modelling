#!/bin/zsh

# Check for Homebrew and install if not available
if ! command -v brew &> /dev/null; then
  echo "Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "Homebrew is already installed."
fi

# Install required dependencies using Homebrew
echo "Installing dependencies..."
brew install gcc libomp cfitsio fftw gsl nlopt

# Install SCons via pip
echo "Installing SCons..."
pip install scons

# Clone the Imfit repository or download the source tarball
echo "Downloading Imfit source code..."
if [ ! -d "imfit" ]; then
  git clone https://github.com/perwin/imfit.git
  cd imfit || exit
else
  echo "Imfit source already exists. Skipping download."
  cd imfit || exit
fi

# Compile Imfit using SCons and Clang
echo "Compiling Imfit using Clang and libomp..."
scons --clang-openmp imfit
scons --clang-openmp imfit-mcmc
scons --clang-openmp makeimage

# Move binaries to /usr/local/bin
echo "Moving binaries to /usr/local/bin..."
sudo cp imfit imfit-mcmc makeimage /usr/local/bin/

# Run tests to verify installation
echo "Running tests..."
./do_imfit_tests
./do_mcmc_tests
./do_makeimage_tests

echo "Imfit installation and verification complete!"
