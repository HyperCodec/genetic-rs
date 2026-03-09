/// Trait that allows a genome to be separated into species.
pub trait Speciated {
    /// How structurally different this genome is to another genome. The higher the value, the more different they are.
    /// When implementing this, you should consider only the major structural differences that could
    /// take several generations to evolve an optimal solution. Do not consider minor differences that
    /// are easily evolved in a single generation, such as a single weight change.
    /// By common convention, this is typically a value between 0 and 1.
    /// Additionally, it should also be symmetric, meaning that the divergence between genome A and genome B should be
    /// the same as the divergence between genome B and genome A.
    /// This value is used to separate genomes into a species.
    fn divergence(&self, other: &Self) -> f32;
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
    pub fn insert_genome<G: Speciated>(&mut self, index: usize, genomes: &[G]) -> bool {
        if let Some(species) = self.get_species_mut(index, genomes) {
            species.push(index);
            return false;
        }
        self.species.push(vec![index]);
        true
    }

    /// Creates a new speciated population from a population of genomes.
    /// Note that this can be O(n^2) worst case, but is typically much faster in practice,
    /// especially if the genome structure doesn't mutate often.
    pub fn from_genomes<G: Speciated>(population: &[G], threshold: f32) -> Self {
        let mut speciated_population = SpeciatedPopulation {
            species: Vec::new(),
            threshold,
        };
        for index in 0..population.len() {
            speciated_population.insert_genome(index, population);
        }
        speciated_population
    }

    /// Gets the species that a genome belongs to, if any.
    pub fn get_species<G: Speciated>(&self, index: usize, genomes: &[G]) -> Option<&Vec<usize>> {
        let genome = &genomes[index];
        for species in &self.species {
            if genome.divergence(&genomes[species[0]]) < self.threshold {
                return Some(species);
            }
        }
        None
    }

    /// Get a mutable reference to the species that a genome belongs to, if any.
    pub fn get_species_mut<G: Speciated>(
        &mut self,
        index: usize,
        genomes: &[G],
    ) -> Option<&mut Vec<usize>> {
        let genome = &genomes[index];
        for species in &mut self.species {
            if genome.divergence(&genomes[species[0]]) < self.threshold {
                return Some(species);
            }
        }
        None
    }
}
