all: format build

format:
	clang-format -style=file -i genetic_sram.cpp

build:
	clang++ -O2 -std=c++20 -fopenmp -lfmt -larmadillo genetic_sram.cpp -o genetic_sram

.phony: format
