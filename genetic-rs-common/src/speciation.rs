/// Trait that allows a genome to be separated into species.
pub trait Speciated {
    /// The context type for speciation. This can be used to provide additional information
    /// that may be needed to calculate the divergence between two genomes.
    type Context;

    /// How structurally different this genome is to another genome. The higher the value, the more different they are.
    /// When implementing this, you should consider only the major structural differences that could
    /// take several generations to evolve an optimal solution. Do not consider minor differences that
    /// are easily evolved in a single generation, such as a single weight change.
    /// By common convention, this is typically a value between 0 and 1.
    /// Additionally, it should also be symmetric, meaning that the divergence between genome A and genome B should be
    /// the same as the divergence between genome B and genome A.
    /// This value is used to separate genomes into a species.
    fn divergence(&self, other: &Self, ctx: &Self::Context) -> f32;
}

/// A population of genomes that have been separated into species.
pub struct SpeciatedPopulation {
    /// The species in this population. Each species is a vector of indices into the original genome vector.
    /// The first genome in a species is its representation (i.e. the one that gets compared to other genomes to determine
    /// if they belong in the species)
    pub species: Vec<Vec<usize>>,

    /// The threshold used to determine if a genome belongs in a species. If the divergence between a genome and the representative genome
    /// of a species is less than this threshold, then the genome belongs in that species.
    pub threshold: f32,
}

impl SpeciatedPopulation {
    /// Inserts a genome into the speciated population.
    /// Returns whether a new species was created by this insertion.
    pub fn insert_genome<G: Speciated>(
        &mut self,
        index: usize,
        genomes: &[G],
        ctx: &G::Context,
    ) -> bool {
        if let Some(species) = self.get_species_mut(index, genomes, ctx) {
            species.push(index);
            return false;
        }
        self.species.push(vec![index]);
        true
    }

    /// Creates a new speciated population from a population of genomes.
    /// Note that this can be O(n^2) worst case, but is typically much faster in practice,
    /// especially if the genome structure doesn't mutate often.
    pub fn from_genomes<G: Speciated>(population: &[G], threshold: f32, ctx: &G::Context) -> Self {
        let mut speciated_population = SpeciatedPopulation {
            species: Vec::new(),
            threshold,
        };
        for index in 0..population.len() {
            speciated_population.insert_genome(index, population, ctx);
        }
        speciated_population
    }

    /// Gets the species that a genome belongs to, if any.
    pub fn get_species<G: Speciated>(
        &self,
        index: usize,
        genomes: &[G],
        ctx: &G::Context,
    ) -> Option<&Vec<usize>> {
        let genome = &genomes[index];
        self.species
            .iter()
            .find(|species| genome.divergence(&genomes[species[0]], ctx) < self.threshold)
    }

    /// Get a mutable reference to the species that a genome belongs to, if any.
    pub fn get_species_mut<G: Speciated>(
        &mut self,
        index: usize,
        genomes: &[G],
        ctx: &G::Context,
    ) -> Option<&mut Vec<usize>> {
        let genome = &genomes[index];
        self.species
            .iter_mut()
            .find(|species| genome.divergence(&genomes[species[0]], ctx) < self.threshold)
    }

    /// Round robin cyclic iterator over the genomes in this population.
    /// Note that the index value is the index in the original genome vector, not the index in the species vector.
    pub fn round_robin(&self) -> impl Iterator<Item = usize> + '_ {
        let species = &self.species;
        let mut idx_in_species = vec![0; self.species.len()];
        let mut species_i = 0usize;

        std::iter::from_fn(move || {
            if species.is_empty() {
                return None;
            }

            let cur_species = &species[species_i];
            let genome_index = cur_species[idx_in_species[species_i]];
            idx_in_species[species_i] += 1;
            if idx_in_species[species_i] >= cur_species.len() {
                idx_in_species[species_i] = 0;
                species_i = (species_i + 1) % species.len();
            }
            Some(genome_index)
        })
    }

    /// Round robin, but also returns the species index along with the genome index.
    /// Note that the genome index value is the index in the original genome vector, not the index in the species vector.
    /// However, the species index represents the index in [`Self::species`].
    /// Format is (species_index, genome_index).
    pub fn round_robin_enumerate(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let species = &self.species;
        let mut idx_in_species = vec![0; self.species.len()];
        let mut species_i = 0usize;

        std::iter::from_fn(move || {
            if species.is_empty() {
                return None;
            }

            let cur_species = &species[species_i];
            let genome_index = cur_species[idx_in_species[species_i]];
            idx_in_species[species_i] += 1;
            if idx_in_species[species_i] >= cur_species.len() {
                idx_in_species[species_i] = 0;
                species_i = (species_i + 1) % species.len();
            }
            Some((species_i, genome_index))
        })
    }
}
