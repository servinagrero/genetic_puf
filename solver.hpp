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

typedef std::pair<float, float> param;

struct metrics_t
{
  param uniformity;
  param bitaliasing;
  param intraHD;
  int nCRP;
  /// Ratio of valid CRPs
  float ratio_valid;
};

template<>
struct fmt::formatter<metrics_t>
{
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template<typename FormatContext>
  auto format(const metrics_t& m, FormatContext& ctx)
  {
    return format_to(ctx.out(),
                     "U({:.4f}, {:.4f}) B({:.4f}, {:.4f}) "
                     "nCRP({}, {:02.2f}%)",
                     m.uniformity.first,
                     m.uniformity.second,
                     m.bitaliasing.first,
                     m.bitaliasing.second,
                     m.nCRP,
                     m.ratio_valid * 100);
  }
};

/// RNG used to allow mutations in the vectors
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> rng_bin(0.0, 1.0);

class Solver
{
public:
  Solver(mat& crp_table, int pop_size, float bitalias_thresh);

  /**
   *
   */
  void optimize(int num_generations);

  uvec best_genome();

  /**
   * @brief Calculate the metrics from a vec
   *
   * @param vec Binary encoded vector to select the CPRs
   *
   * @return PUF metrics for the given CRPS
   */
  metrics_t calculate_metrics(uvec& genome);

  float calculate_fitness(metrics_t metrics);

private:
  /**
   * @brief Toggle random values in a binary encoded vector
   *
   * @param vec Binary encoded vector
   * @param mutation_rate Probability of mutation.
   *
   * @return Nothing.
   */
  void mutate_genome(uvec& genome, float mutation_rate);

  // CRP table
  mat& crp_table;

  field<metrics_t> metrics_table;
  fvec fitness_table;

  // Group of vectors to optimize the metrics
  mat population;

  /// Threshold for the deviation of bitaliasing used to filter challenges.
  float bitalias_thresh;

  /// Vector to precompute the bitaliasing deviation for every CRP
  fvec bitalias_lut;

  /// Filepath to store the CRPs selected.
  std::string out_file;
};

Solver::Solver(mat& crp_table, int pop_size, float bitalias_thresh)
  : crp_table(crp_table)
  , metrics_table(pop_size)
  , fitness_table(pop_size, fill::zeros)
  , population(pop_size, crp_table.n_cols, fill::ones)
  , bitalias_thresh(bitalias_thresh)
{
  this->population.imbue([&]() { return rng_bin(gen) <= 0.2 ? 0 : 1; });

  this->bitalias_lut = conv_to<fvec>::from(mean(this->crp_table, 0));
  this->bitalias_lut = arma::abs(this->bitalias_lut - 0.5);
}

metrics_t
Solver::calculate_metrics(uvec& genome)
{
  metrics_t metrics;

  auto mat_sel = conv_to<mat>::from(this->crp_table.cols(genome));
  auto sum_cols = mean(mat_sel, 0);
  auto sum_rows = mean(mat_sel, 1);

  metrics.uniformity = std::make_pair(mean(sum_rows), stddev(sum_rows));
  metrics.bitaliasing = std::make_pair(mean(sum_cols), stddev(sum_cols));
  metrics.intraHD = std::make_pair(0.0, 0.0);

  metrics.nCRP = genome.size();
  metrics.ratio_valid = ((float)metrics.nCRP / this->crp_table.n_cols);
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
float
Solver::calculate_fitness(metrics_t metrics)
{
  /// maximum possible fitness achieved by a set of metrics.
  float max_fitness = 3;

  float fitness = 0;

  fitness += abs(metrics.uniformity.first - 0.5) + metrics.uniformity.second;
  fitness += abs(metrics.bitaliasing.first - 0.5) + metrics.bitaliasing.second;
  // fitness += metrics.intraHD.first + metrics.intraHD.second;
  fitness += metrics.ratio_valid;

  return pow((max_fitness - fitness) / max_fitness, 4);
}

void
Solver::mutate_genome(uvec& genome, float mutation_rate)
{
  for (size_t i = 0; i < genome.size(); ++i) {
    if (this->bitalias_lut.at(i) >= this->bitalias_thresh) {
      if (genome.at(i) == 1 && rng_bin(gen) <= mutation_rate) {
        genome.at(i) = 0;
      }
    } else {
      if (rng_bin(gen) <= mutation_rate) {
        genome.at(i) = (genome.at(i) + 1) % 2;
      }
    }
  }
}

void
Solver::optimize(int num_generations)
{

  float highest_fitness = max(this->fitness_table);
  metrics_t highest_metrics;

  fmt::print(
    "Population [{}, {}]\n", this->population.n_rows, this->population.n_cols);

  for (int g = 1; g <= num_generations; ++g) {
    fmt::print("[INFO] Generation {}", g);

#pragma omp parallel for shared(metrics_table, fitness_table, crp_table)       \
  schedule(runtime)
    for (size_t iter = 0; iter < this->population.n_rows; ++iter) {
      uvec crps = find(this->population.row(iter) == 1);
      metrics_t m = this->calculate_metrics(crps);

#pragma omp critical
      {
        this->metrics_table.at(iter) = m;
        this->fitness_table.at(iter) = calculate_fitness(m);
      }
    }

    uvec indices = sort_index(fitness_table, "descend");

    if (fitness_table(indices(0)) > highest_fitness) {
      highest_fitness = fitness_table(indices(0));
      highest_metrics = metrics_table(indices(0));
    }

    auto best_genome = conv_to<rowvec>::from(population.row(indices.at(0)));
    for (size_t iter = 0; iter < this->population.n_rows; ++iter) {
      population.row(iter) = best_genome;
    }

    const float mutation_rate = 5 / exp(5 * highest_fitness);

#pragma omp parallel for shared(metrics_table, fitness_table, crp_table)       \
  schedule(runtime)
    for (size_t i = 0; i < this->population.n_rows; ++i) {
      auto genome = conv_to<uvec>::from(population.row(i));
      this->mutate_genome(genome, mutation_rate);
      population.row(i) = conv_to<rowvec>::from(genome);
    }

    fmt::print(" {:*>20}\n", "");
    fmt::print("Mutation rate {}\n", mutation_rate);
    fmt::print("Overall fitness: {:.4f}, {:.4f}\n",
               mean(fitness_table),
               stddev(fitness_table));
    fmt::print("Max fitness in generation: {:.4f}\n",
               fitness_table(indices(0)));
    fmt::print("Highest world fitness: {:.4f}\n", highest_fitness);
    fmt::print("{}\n\n", metrics_table(indices(0)));
  }


  uvec complete(crp_table.n_cols, fill::ones);
  for (size_t i = 0; i < complete.size(); ++i)
          complete.at(i) = i;

  metrics_t raw_metrics = this->calculate_metrics(complete);
  fmt::print("Original:\n");
  fmt::print("{}\n", raw_metrics);
  fmt::print("Fitness {}\n\n", this->calculate_fitness(raw_metrics));

  fmt::print("Solution achieved:\n");
  fmt::print("{}\n", highest_metrics);
  fmt::print("Fitness {}\n\n", highest_fitness);

  fmt::print("Dummy bitaliasing solution:\n");
  uvec sel_dummy = find(this->bitalias_lut <= this->bitalias_thresh);
  metrics_t metrics_dummy = calculate_metrics(sel_dummy);
  fmt::print("{}\n", metrics_dummy);
  fmt::print("Fitness {}\n\n", this->calculate_fitness(metrics_dummy));

}

uvec
Solver::best_genome()
{
  uvec indices = sort_index(this->fitness_table, "descend");
  return conv_to<uvec>::from(this->population.row(indices.at(0)));
}

/**
 * @brief Compute the hamming distance between two vectors
 *
 * @param first first vector
 * @param second second vector
 *
 * @return Distance between the vectors
 */
int
hamming_dist(uvec first, uvec second)
{
  assert(first.size() == second.size());
  int dist = 0;

  for (size_t b = 0; b < first.size(); ++b) {
    if (first.at(b) != second.at(b))
      ++dist;
  }
  return dist;
}
