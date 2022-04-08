#include <assert.h>
#include <fstream>
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

template<typename T>
int
hamming_dist(T& first, T& second);
        
struct metrics_t
{
  float uniformity_mu;
  float uniformity_sd;
  float bitaliasing_mu;
  float bitaliasing_sd;
  float intraHD_mu;
  float intraHD_sd;
  int nCRP;
  float ratio_valid;

   // Calculate the fitness of a set of metrics
   // The closser the fitness is to 0 the better the metrics are.
  float fitness()
  {
    /// maximum possible fitness achieved by a set of metrics.
    float max_fitness = 4.5;

    float fitness = 0;

    fitness += abs(this->uniformity_mu - 0.5) + this->uniformity_sd;
    fitness += abs(this->bitaliasing_mu - 0.5) + this->bitaliasing_sd;
    fitness += (1 - this->intraHD_mu) + this->intraHD_sd;
    fitness += this->ratio_valid;

    return pow((max_fitness - fitness) / max_fitness, 4);
  }
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
                     "U({:.4f}, {:.4f}) B({:.4f}, {:.4f}) R({:.4f}, {:.4f}) "
                     "nCRP({}, {:02.2f}%)",
                     m.uniformity_mu,
                     m.uniformity_sd,
                     m.bitaliasing_mu,
                     m.bitaliasing_sd,
                     m.intraHD_mu,
                     m.intraHD_sd,
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
  Solver(cube& crp_table, int pop_size, float bitalias_thresh);

  void optimize(int num_generations);
  uvec best_genome();
  float best_fitness();
  metrics_t best_metrics();

  // Calculate the metrics from a vec
  // 
  // vec is a binary encoded vector to select the CPRs
  metrics_t calculate_metrics(uvec& genome);

private:
        
   // Toggle random values in a binary encoded vector
   //
   // vec is a binary encoded vector
   // mutation_rate is the probability of mutation.
  rowvec mutate_genome(const rowvec& genome, const float& mutation_rate);

  cube& crp_table;
  field<metrics_t> metrics_table;
  fvec fitness_table;

  // Group of vectors to optimize the metrics
  mat population;

  // Threshold for the deviation of bitaliasing used to filter challenges.
  float bitalias_thresh;

  // Vector to precompute the bitaliasing deviation for every CRP
  fvec bitalias_lut;

  // Filepath to store the CRPs selected.
  std::string out_file;

  std::ofstream log_file;
  int generations;
};

Solver::Solver(cube& crp_table, int pop_size, float bitalias_thresh)
  : crp_table(crp_table)
  , metrics_table(pop_size)
  , fitness_table(pop_size, fill::zeros)
  , population(pop_size, crp_table.n_cols, fill::ones)
  , bitalias_thresh(bitalias_thresh)
  , generations(0)
{
  this->log_file.open("./log_file.csv", std::ios::trunc);
  this->log_file << "generation,fitness,mutation_rate\n";
  this->log_file.close();
  this->population.imbue([&]() { return rng_bin(gen) <= 0.1 ? 0 : 1; });

  this->bitalias_lut = conv_to<fvec>::from(mean(this->crp_table.slice(0), 0));
  this->bitalias_lut = arma::abs(this->bitalias_lut - 0.5);
}

metrics_t
Solver::calculate_metrics(uvec& genome)
{
  metrics_t metrics;

  auto mat_ref = conv_to<mat>::from(this->crp_table.slice(0).cols(genome));
  auto sum_cols = mean(mat_ref, 0);
  auto sum_rows = mean(mat_ref, 1);

  metrics.uniformity_mu = mean(sum_rows);
  metrics.uniformity_sd = stddev(sum_rows);
  metrics.bitaliasing_mu = mean(sum_cols);
  metrics.bitaliasing_sd = stddev(sum_cols);

  fvec intraHD_values(this->crp_table.n_rows, fill::zeros);

  for (size_t dev = 0; dev < this->crp_table.n_rows; ++dev) {
          float dist = 0;
          rowvec ref_sample = mat_ref.row(dev);
          
          #pragma omp parallel for shared(crp_table) reduction(+:dist)
          for (size_t s = 1; s < this->crp_table.n_slices; ++s) {
                  mat mat_sample = this->crp_table.slice(s).cols(genome);
                  rowvec sample = mat_sample.row(dev);
                  dist += (float)hamming_dist(ref_sample, sample) / genome.size();
          }
          intraHD_values.at(dev) = dist / this->crp_table.n_slices;
  }
  metrics.intraHD_mu = 1 - mean(intraHD_values);
  metrics.intraHD_sd = stddev(intraHD_values);

  metrics.nCRP = genome.size();
  metrics.ratio_valid = ((float)metrics.nCRP / this->crp_table.n_cols);
  return metrics;
}

rowvec
Solver::mutate_genome(const rowvec& genome, const float& mutation_rate)
{
  rowvec new_genome(genome.size());

  for (size_t i = 0; i < genome.size(); ++i) {
    if (this->bitalias_lut.at(i) >= this->bitalias_thresh) {
      if (genome.at(i) == 1 && rng_bin(gen) <= mutation_rate) {
        new_genome.at(i) = 0;
      }
    } else {
      if (rng_bin(gen) <= mutation_rate) {
        new_genome.at(i) = genome.at(i) == 0 ? 1 : 0;
      }
    }
  }
  return new_genome;
}

void
Solver::optimize(int num_generations)
{

  float highest_fitness = max(this->fitness_table);
  metrics_t highest_metrics;

  fmt::print(
    "Population [{}, {}]\n", this->population.n_rows, this->population.n_cols);

  this->log_file.open("./log_file.csv", std::ios::app);

  for (int g = 1; g <= num_generations; ++g) {
    fmt::print("[INFO] Generation {}", g + this->generations);

#pragma omp parallel for shared(metrics_table, fitness_table, crp_table)       \
  schedule(runtime)
    for (size_t iter = 0; iter < this->population.n_rows; ++iter) {
      uvec crps = find(this->population.row(iter) == 1);
      metrics_t m = this->calculate_metrics(crps);

#pragma omp critical
      {
        this->metrics_table.at(iter) = m;
        this->fitness_table.at(iter) = m.fitness();
      }
    }

    uvec indices = sort_index(fitness_table, "descend");

    if (fitness_table(indices(0)) > highest_fitness) {
      highest_fitness = fitness_table(indices(0));
      highest_metrics = metrics_table(indices(0));
    }

    rowvec best_genome = population.row(indices.at(0));
    const float mutation_rate = 1 / exp(2 * highest_fitness);

#pragma omp parallel for shared(metrics_table, fitness_table, crp_table)       \
  schedule(runtime)
    for (size_t i = 0; i < this->population.n_rows; ++i) {
      rowvec new_genome = this->mutate_genome(best_genome, mutation_rate);
      population.row(i) = new_genome;
    }

    fmt::print(this->log_file,
               "{},{},{}\n",
               g + this->generations,
               highest_fitness,
               mutation_rate);

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

  this->log_file.close();
  this->generations += num_generations;

  uvec complete(crp_table.n_cols, fill::ones);
  for (size_t i = 0; i < complete.size(); ++i)
    complete.at(i) = i;

  metrics_t raw_metrics = this->calculate_metrics(complete);
  fmt::print("Original:\n");
  fmt::print("{}\n", raw_metrics);
  fmt::print("Fitness {}\n\n", raw_metrics.fitness());

  fmt::print("Solution achieved:\n");
  fmt::print("Total generations: {}\n", this->generations);
  fmt::print("{}\n", highest_metrics);
  fmt::print("Fitness {}\n\n", highest_fitness);

  fmt::print("Dummy bitaliasing solution:\n");
  uvec sel_dummy = find(this->bitalias_lut <= this->bitalias_thresh);
  metrics_t metrics_dummy = calculate_metrics(sel_dummy);
  fmt::print("{}\n", metrics_dummy);
  fmt::print("Fitness {}\n\n", metrics_dummy.fitness());
}

uvec
Solver::best_genome()
{
  uvec indices = sort_index(this->fitness_table, "descend");
  return conv_to<uvec>::from(this->population.row(indices.at(0)));
}

float
Solver::best_fitness()
{
  uvec indices = sort_index(this->fitness_table, "descend");
  return this->fitness_table.at(indices.at(0));
}

metrics_t
Solver::best_metrics()
{
  uvec indices = sort_index(this->fitness_table, "descend");
  return this->metrics_table.at(indices.at(0));
}

// Compute the hamming distance between two vectors
template<typename T>
int
hamming_dist(T& first, T& second)
{
  assert(first.size() == second.size());
  int dist = 0;

  for (size_t b = 0; b < first.size(); ++b) {
          if (first.at(b) != second.at(b)) {
                  dist += 1;
          }
  }
  return dist;
}
