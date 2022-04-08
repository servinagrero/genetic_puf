/**
 * @mainpage Genetic PUF Maximization Documentation
 * @author Sergio Vinagrero Gutierrez
 *
 * @section intro_sec Introduction
 * Genetic algorithm to reliable filter PUF CRP tables.
 * This algorithm tries to select a random number of challenges
 * from the full CRP table in order to maximize PUF metrics.
 *
 * @subsection install_dependencies Installing Dependencies
 *
 * This algorithm depends on armadillo, fmt, and OpenMP.
 * OpenMD is not mandatory but highly recomended to reduce the time.
 */

#include <assert.h>
#include <getopt.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <armadillo>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <omp.h>

#include "solver.hpp"

using namespace arma;

struct options_t
{
  std::string in_file;
  std::string out_file;
  float bitalias_thresh;
  int population_size;
  int num_generations;
};

options_t
parse_args(int argc, char** argv)
{
  int opt;
  options_t options;

  while ((opt = getopt(argc, argv, "f:p:t:g:o")) != -1) {
    switch (opt) {
      case 'f':
        if (!strcmp(optarg, "")) {
          throw std::invalid_argument("Must supply PUF CRP table.");
        }
        options.in_file = optarg;
        break;
      case 'o':
        if (!strcmp(optarg, "")) {
          throw std::invalid_argument("Must supply output table.");
        } else {
          options.out_file = optarg;
        }
        break;
      case 't':
        if (strcmp(optarg, "")) {
          options.bitalias_thresh = std::stof(optarg);
        }
        break;
      case 'p':
        if (strcmp(optarg, "")) {
          options.population_size = std::stof(optarg);
        }
        break;
      case 'g':
        if (strcmp(optarg, "")) {
          options.num_generations = std::stof(optarg);
        }
        break;
    }
  }

  return options;
}

void
usage()
{
  std::cout << "usage genetic_puf -f <crp_table>.csv \n"
            << "                  -o: path to output file\n"
            << "                  -g: number of generations\n"
            << "                  -t: threshold for bitaliasing\n"
            << "                  -p: population size\n"
            << "                  -v: verbose\n";
}

int
main(int argc, char** argv)
{
  // options_t options = parse_args(argc, argv);
        
  mat crp_table;
  crp_table.load("/home/vinagres/Programming/dumps_thesis/32/M_bits.csv", csv_ascii);

  cube crps(crp_table.n_rows, crp_table.n_cols, 10, fill::ones);
  crps.slice(0) = crp_table;
  
  fmt::print("crp_table: [{}, {}]\n", crp_table.n_rows, crp_table.n_cols);

  Solver solver = Solver(crps, 5, 0.1);
  solver.optimize(50);
  auto solution = solver.best_genome();

  fmt::print("Printing solution to file: {}.\n", "./genome.csv");
  solution.save("./genome.csv", csv_ascii);

  return 0;
}
