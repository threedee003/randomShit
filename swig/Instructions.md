Step 1 : Create the header `example.h` and declare function definitions.
Step 2 : Write the functions in `example.cpp` and also import `example.h` in the `.cpp` file.
Step 3 : Write the interface file `example.i`.
Step 4 : Use the command `swig -c++ -python example.i` to create binaries.
Step 5 : Write the `setup.py` file and use the cmd `python3 setup.py build_ext --inplace` to install the library in your python3 site-packages.
