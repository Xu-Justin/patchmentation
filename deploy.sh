# read version
VERSION=$(python3 -c "from setup import version; print(version)")
echo "version:" $VERSION

# remove caches
echo "Removing caches..."
sudo rm -rf build/
sudo rm -rf dist/
sudo rm -rf *.egg-info/
sudo rm -rf patchmentation/*.egg-info/

# build
echo "Building wheel..."
python3 setup.py bdist_wheel

# upload
if [[ $1 == "PyPI" ]]; then
    echo "Uploading to PyPI..."
    twine upload dist/*
elif [[ $1 == "Test PyPI" ]] || [[ $1 == "TestPyPI" ]]; then
    echo "Uploading to Test PyPI..."
    twine upload --repository testpypi dist/*
else
    echo "No arguments are given. [Provide \"PyPI\" to upload to PyPI or \"Test PyPI\" to upload to Test PyPI]"
    echo "Upload failed."
fi
