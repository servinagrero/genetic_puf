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

using namespace arma;

/// PUF CRP table where each row is a device and each column a CRP
mat PUF_TABLE;

/// Vector to precompute the bitaliasing deviation for every CRP
fvec BITALIAS_LUT;

// std::vector<std::tuple<uint8_t, uint8_t> > pairs_lut;

/// RNG used to allow mutations in the vectors
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> rng_bin(0.0, 1.0);

struct options_t {
  /// Change of mutation of a CRP.
  float mutation_rate;
  /// How many vectors should be calculated in every generation.
  int population_size;
  /// Number of generations to run the genetic algorithm.
  int num_generations;
  /// Threshold for the deviation of bitaliasing used to filter challenges.
  float bitalias_thresh;
  /// Filepath to the CSV containing the full CRP table
  std::string in_file;
  /// Filepath to store the CRPs selected.
  std::string out_file;
  /// Whether to print output of each generation.
  bool verbose;
};

/**
 * @brief Parse the option from stdin.
 *
 * @param argc Number of arguments passed
 * @param argv The argumntes passed as strings
 *
 * @returns Struct containing the parsed options.
 */
options_t parse_args(int argc, char **argv) {
  int opt;
  options_t config;

  while ((opt = getopt(argc, argv, "f:p:t:g:o:v")) != -1) {
    switch (opt) {
    case 'f':
      if (!strcmp(optarg, "")) {
        throw std::invalid_argument("Must supply PUF CRP table.");
      }
      config.in_file = optarg;
    case 'o':
      if (!strcmp(optarg, "")) {
        throw std::invalid_argument("Must supply output table.");
      } else {
        config.out_file = optarg;
      }
      break;
    case 't':
      if (strcmp(optarg, "")) {
        config.bitalias_thresh = std::stof(optarg);
      }
      break;
    case 'p':
      if (strcmp(optarg, "")) {
        config.population_size = std::stof(optarg);
      }
      break;
    case 'g':
      if (strcmp(optarg, "")) {
        config.num_generations = std::stof(optarg);
      }
      break;
    case 'v':
      config.verbose = true;
      break;
    }
  }

  return config;
}

options_t CONFIG;

struct metrics_t {
  /// Uniformity mean
  float uniformity_mu;
  /// Uniformity standard deviation
  float uniformity_sd;
  /// Bitaliasing mean
  float bitaliasing_mu;
  /// Bitaliasing standard deviation
  float bitaliasing_sd;
  /// Number of valid CRPs
  int nCRP;
};

template <> struct fmt::formatter<metrics_t> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const metrics_t &m, FormatContext &ctx) {
    return format_to(ctx.out(),
                     "Uniformity({:.4f}, {:.4f}) Bitaliasing({:.4f}, {:.4f}) "
                     "nCRP({}, {:02.2f} %)",
                     m.uniformity_mu, m.uniformity_sd, m.bitaliasing_mu,
                     m.bitaliasing_sd, m.nCRP,
                     ((float)m.nCRP / PUF_TABLE.n_cols) * 100);
  }
};

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

/**
 * @brief Compute the hamming distance between two vectors
 *
 * @param first first vector
 * @param second second vector
 *
 * @return Distance between the vectors
 */
int hamming_dist(uvec first, uvec second) {
  assert(first.size() == second.size());
  int dist = 0;

  for (size_t b = 0; b < first.size(); ++b) {
    if (first.at(b) != second.at(b))
      ++dist;
  }
  return dist;
}

/**
 * @brief Toggle random values in a binary encoded vector
 *
 * @param vec Binary encoded vector
 * @param mutation_rate Probability of mutation.
 *
 * @return Nothing.
 */
void mutate_vector(uvec &vec, float &mutation_rate, float &bitalias_thresh) {
  for (size_t i = 0; i < vec.size(); ++i) {
    if (BITALIAS_LUT.at(i) >= bitalias_thresh) {
      if (vec.at(i) == 1 && rng_bin(gen) <= mutation_rate) {
        vec.at(i) = 0;
      }
    } else {
      if (rng_bin(gen) <= mutation_rate) {
        vec.at(i) = (vec.at(i) + 1) % 2;
      }
    }
  }
}

/**
 * @brief Calculate the metrics from a vec
 *
 * @param vec Binary encoded vector to select the CPRs
 *
 * @return PUF metrics for the given CRPS
 */
metrics_t calculate_metrics(mat &puf_table, uvec &vec) {
  metrics_t metrics;

  auto mat_sel = conv_to<mat>::from(puf_table.cols(vec));
  auto sum_cols = mean(mat_sel, 0);
  auto sum_rows = mean(mat_sel, 1);

  metrics.uniformity_mu = mean(sum_rows);
  metrics.uniformity_sd = stddev(sum_rows);

  metrics.bitaliasing_mu = mean(sum_cols);
  metrics.bitaliasing_sd = stddev(sum_cols);

  metrics.nCRP = vec.size();

  // uvec interHD_values(pairs_lut.size(), fill::zeros);
  // int pair = 0;

  // for (auto &[f, s] : pairs_lut) {
  //   auto first = conv_to<uvec>::from(mat_sel.row(f));
  //   auto second = conv_to<uvec>::from(mat_sel.row(f));
  //   interHD_values.at(pair++) =
  //       hamming_dist(first, second) / PUF_TABLE.n_cols * 100;
  // }

  // metrics.interHD = (2 * sum(interHD_values)) /
  //                   (PUF_TABLE.n_rows * (PUF_TABLE.n_rows - 1)) * 0.01;

  // metrics.interHD = 0;

  return metrics;
}

/**
 * @brief Calculate the fitness of a set of metrics
 *
 * The closser the fitness is to 0 the better the metrics are.
 *
 * @param metrics set of metrics to evaluate.
 * @return the fitness of the metrics.
 */
float calculate_fitness(metrics_t metrics) {
  /// maximum possible fitness achieved by a set of metrics.
  float max_fitness = 3;

  float fitness = 0;

  fitness += abs(metrics.uniformity_mu - 0.5) + metrics.uniformity_sd;
  fitness += abs(metrics.bitaliasing_mu - 0.5) + metrics.bitaliasing_sd;
  fitness += (PUF_TABLE.n_cols - metrics.nCRP) / PUF_TABLE.n_cols;

  return pow((max_fitness - fitness) / max_fitness, 4);
}

void usage() {
        std::cout << "usage genetic_puf -f <crp_table>.csv \n"
                  << "                  -o: path to output file\n"
                  << "                  -g: number of generations\n"
                  << "                  -t: threshold for bitaliasing\n"
                  << "                  -p: population size\n"
                  << "                  -v: verbose\n";
}

int main(int argc, char **argv) {
  CONFIG = parse_args(argc, argv);

  if (CONFIG.in_file == "" || CONFIG.out_file == "") {
          usage();
          return(1);
  }
  
  PUF_TABLE.load(CONFIG.in_file, csv_ascii);

  BITALIAS_LUT = conv_to<fvec>::from(mean(PUF_TABLE, 0));
  BITALIAS_LUT = arma::abs(BITALIAS_LUT - 0.5);

  fmt::print("[{}, {}]\n", PUF_TABLE.n_rows, PUF_TABLE.n_cols);
  fmt::print("POPULATION SIZE {}\n", CONFIG.population_size);
  fmt::print("BITALIASING THRESHOLD {}\n", CONFIG.bitalias_thresh);
  fmt::print("GENERATIONS {}\n", CONFIG.num_generations);

  mat population(CONFIG.population_size, PUF_TABLE.n_cols, fill::ones);
  // population.imbue([&]() { return rng_bin(gen) <= 0.1 ? 0 : 1; });

  float highest_fitness = 0.0;
  metrics_t highest_metrics;

  uvec solution(PUF_TABLE.n_cols, fill::ones);
  field<metrics_t> metrics_table(CONFIG.population_size);
  fvec fitness_table(CONFIG.population_size, fill::zeros);

  for (int generation = 1; generation <= CONFIG.num_generations; ++generation) {
#pragma omp parallel for shared(metrics_table, fitness_table, PUF_TABLE)       \
    schedule(runtime)
    for (int iter = 0; iter < CONFIG.population_size; ++iter) {
      uvec crps = find(population.row(iter) == 1);

      metrics_t m = calculate_metrics(PUF_TABLE, crps);

#pragma omp critical
      {
        metrics_table.at(iter) = m;
        fitness_table.at(iter) = calculate_fitness(m);
      }
    }

    uvec indices = sort_index(fitness_table, "descend");

    if (fitness_table(indices(0)) > highest_fitness) {
      highest_fitness = fitness_table(indices(0));
      highest_metrics = metrics_table(indices(0));
      solution = conv_to<uvec>::from(population.row(indices.at(0)));
    }

    auto best_genome = conv_to<rowvec>::from(population.row(indices.at(0)));

    for (int iter = 0; iter < CONFIG.population_size; ++iter) {
      population.row(iter) = best_genome;
    }

    float mutation_rate = 5 / exp(5 * highest_fitness);

#pragma omp parallel for shared(metrics_table, fitness_table, PUF_TABLE)       \
    schedule(runtime)
    for (int iter = 0; iter < CONFIG.population_size; ++iter) {
      auto genome = conv_to<uvec>::from(population.row(iter));
      mutate_vector(genome, mutation_rate, CONFIG.bitalias_thresh);
      population.row(iter) = conv_to<rowvec>::from(genome);
    }

    fmt::print("{:*>80}\n", "");
    fmt::print("Generation {}\n", generation);
    fmt::print("Mutation rate {}\n", mutation_rate);
    fmt::print("Overall fitness: {:.4f}, {:.4f}\n", mean(fitness_table),
               stddev(fitness_table));
    fmt::print("Max fitness in generation: {:.4f}\n",
               fitness_table(indices(0)));
    fmt::print("Highest world fitness: {:.4f}\n", highest_fitness);
    fmt::print("{}\n\n", metrics_table(indices(0)));
  }

  uvec complete(PUF_TABLE.n_cols, fill::ones);
  for (size_t i = 0; i < complete.size(); ++i)
    complete.at(i) = i;

  metrics_t raw_metrics = calculate_metrics(PUF_TABLE, complete);
  fmt::print("Original:\n");
  fmt::print("{}\n", raw_metrics);
  fmt::print("Fitness {}\n", calculate_fitness(raw_metrics));

  fmt::print("Solution achieved:\n");
  fmt::print("{}\n", highest_metrics);
  fmt::print("Fitness {}\n", highest_fitness);

  fmt::print("Printing solution to file: {}.\n", CONFIG.out_file);
  solution.save(CONFIG.out_file, csv_ascii);

  return 0;
}
