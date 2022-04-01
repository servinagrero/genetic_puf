#!/usr/bin/sh

set +xe

export OMP_MAX_ACTIVE_LEVELS="3"
export OMP_SCHEDULE="GUIDED,14"

data_file="$HOME/Programming/dumps_nucleo/M_bits_station.csv"

run_benchmark() {
    pop_size="$1"
    threshold="$2"
    num_generations="$3"

    for i in $(seq 1 20); do
        out_file="$(printf './results_genetics/genetic_%02d_%.2f_%03d_%02d.csv' ${pop_size} ${threshold} ${num_generations} ${i})"
        ./genetic_sram -f "${data_file}" -p "${pop_size}" -t "$threshold" -g "${num_generations}" -o "${out_file}"
    done
}

for pop_size in 10 30 50; do
    for threshold in 0.25 0.15 0.05 ; do
        for num_generations in 50 100 200 ; do
            printf "%d,%.2f,%d\n" ${pop_size} ${threshold} ${num_generations}
            run_benchmark ${pop_size} ${threshold} ${num_generations}
        done
    done
done


