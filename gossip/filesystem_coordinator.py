import json
import time
import os
import glob
import threading
from typing import Optional, Callable

class FilesystemCoordinator:
    """
    Coordinates population-wide evolution via the filesystem.
    - Rank 0 reads all fitness files and writes a single global ranking file.
    - All ranks read this global ranking file to determine their percentile.
    """
    def __init__(self, global_rank: int, world_size: int, output_dir: str):
        self.global_rank = global_rank
        self.world_size = world_size
        self.is_coordinator = (global_rank == 0)
        
        # Directories and files
        self.fitness_dir = os.path.join(output_dir, "fitness")
        self.fitness_file = os.path.join(self.fitness_dir, f"rank_{self.global_rank:04d}.json")
        self.ranking_file = os.path.join(output_dir, "global_ranking.json")
        
        # Rank 0 is responsible for creating the shared directory.
        if self.is_coordinator:
            os.makedirs(self.fitness_dir, exist_ok=True)
            
        # State
        self.running = False
        self.thread = None
        self.last_fitness_update = 0
        self.last_ranking_read = 0
        self.population_rankings = {}
        self.get_fitness: Callable[[], Optional[float]] = lambda: None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def set_fitness_provider(self, fitness_provider_fn: Callable[[], Optional[float]]):
        """Callback to get current fitness from the main thread's EvolutionaryNode."""
        self.get_fitness = fitness_provider_fn

    def get_my_percentile(self) -> Optional[float]:
        """Get this rank's current fitness percentile (0.0=best, 1.0=worst)."""
        # The keys in the JSON file are strings
        return self.population_rankings.get(str(self.global_rank))

    def _run(self):
        """Main loop for the coordinator thread."""
        time.sleep(5)  # Initial delay for things to settle.
        while self.running:
            try:
                # All ranks periodically report their own fitness.
                if time.time() - self.last_fitness_update > 15:
                    self._write_fitness_file()
                
                # Rank 0 is the only one that builds the global ranking.
                if self.is_coordinator:
                    self._update_global_ranking()

                # All ranks periodically read the global ranking.
                self._read_global_ranking()

            except Exception as e:
                # Use a print statement as logging might not be configured in a thread.
                print(f"[Rank {self.global_rank}] FilesystemCoordinator Error: {e}", flush=True)

            time.sleep(10)  # Main loop interval.

    def _write_fitness_file(self):
        fitness = self.get_fitness()
        if fitness is None or fitness == float('inf'):
            return
            
        # Atomic write via rename
        temp_file = self.fitness_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump({'rank': self.global_rank, 'fitness': fitness, 'timestamp': time.time()}, f)
        os.rename(temp_file, self.fitness_file)
        self.last_fitness_update = time.time()

    def _update_global_ranking(self):
        """(Rank 0 only) Reads all fitness files and creates the ranking."""
        all_fitness = []
        for f_path in glob.glob(os.path.join(self.fitness_dir, "rank_*.json")):
            try:
                with open(f_path, 'r') as f:
                    data = json.load(f)
                # Only consider data from the last 5 minutes to prune dead ranks.
                if time.time() - data['timestamp'] < 300:
                    all_fitness.append((data['rank'], data['fitness']))
            except (IOError, json.JSONDecodeError):
                continue

        if not all_fitness: return

        all_fitness.sort(key=lambda x: x[1]) # Sort by fitness (lower is better)

        num_active = len(all_fitness)
        rankings = {
            # Map each rank to its percentile (0.0 = best, 1.0 = worst)
            rank_info[0]: i / (num_active - 1) if num_active > 1 else 0.5
            for i, rank_info in enumerate(all_fitness)
        }
        
        # Atomic write
        temp_file = self.ranking_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump({'timestamp': time.time(), 'rankings': rankings}, f)
        os.rename(temp_file, self.ranking_file)

    def _read_global_ranking(self):
        """(All ranks) Reads the ranking file created by rank 0."""
        if not os.path.exists(self.ranking_file): return
        try:
            mtime = os.path.getmtime(self.ranking_file)
            if mtime <= self.last_ranking_read: return

            with open(self.ranking_file, 'r') as f:
                data = json.load(f)

            if time.time() - data['timestamp'] < 300:
                self.population_rankings = data['rankings']
                self.last_ranking_read = mtime
        except (IOError, json.JSONDecodeError):
            self.population_rankings = {}
