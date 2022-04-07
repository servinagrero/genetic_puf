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
#include <tuple>
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

/**
 * @brief Parse the option from stdin.
 *
 * @param argc Number of arguments passed
 * @param argv The argumntes passed as strings
 *
 * @returns Struct containing the parsed options.
 */
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

/**
 * @brief Compute the number of combinations with repetition of n elements
 *
 * param n Maximum number to create combinations.
 *
 * @return number of pairs.
 */
// void compute_pairs(int n)
// {
// 	for (int i = 0; i < n; ++i) {
// 		for (int j = i + 1; j < n; ++j) {
// 			pairs_lut.push_back(std::make_tuple(i, j));
// 		}
// 	}
// }

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
  crp_table.load("/home/vinagrero/M_bits_station.csv", csv_ascii);

  fmt::print("crp_table: [{}, {}]\n", crp_table.n_rows, crp_table.n_cols);

  Solver solver = Solver(crp_table, 30, 0.05);

  solver.optimize(50);
  auto solution = solver.best_genome();

  // uvec complete(crp_table.n_cols, fill::ones);
  // for (size_t i = 0; i < complete.size(); ++i)
  //         complete.at(i) = i;

  //   metrics_t raw_metrics = solver.calculate_metrics(complete);
  //   fmt::print("Original:\n");
  //   fmt::print("{}\n", raw_metrics);
  //   fmt::print("Fitness {}\n\n", solver.calculate_fitness(raw_metrics));

  // fmt::print("Solution achieved:\n");
  // fmt::print("{}\n", highest_metrics);
  // fmt::print("Fitness {}\n\n", highest_fitness);

  // fmt::print("Dummy bitaliasing solution:\n");
  // uvec sel_dummy = find(BITALIAS_LUT <= CONFIG.bitalias_thresh);
  // metrics_t metrics_dummy = calculate_metrics(solver.crp_table, sel_dummy);
  // fmt::print("{}\n", metrics_dummy);
  // fmt::print("Fitness {}\n\n", solver.calculate_fitness(metrics_dummy));

  fmt::print("Printing solution to file: {}.\n", "./genome.csv");
  solution.save("./genome.csv", csv_ascii);

  return 0;
}
