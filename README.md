# genetic-rs
A crate for quickstarting genetic algorithm projects

### How to Use
First off, this crate comes with the `builtin` and `genrand` modules by default. If you want to add the builtin sexual reproduction extension, you can do so by adding the `sexual` feature.

Once you have eveything imported as you wish, you can define your entity and impl the required traits:

```rust
#[derive(Clone, Debug)] // clone is currently a required derive for pruning nextgens.
struct MyEntity {
    field1: f32,
}

// required in all of the builtin functions as requirements of `ASexualEntity` and `SexualEntity`
impl RandomlyMutable for MyEntity {
    fn mutate(&mut self, rate: f32) {
        self.field1 += fastrand::f32() * rate;
    }
}

// required for `asexual_pruning_nextgen`.
impl ASexualEntity for MyEntity {
    fn spawn_child(&self) -> Self {
        let mut child = self.clone();
        child.mutate(0.25); // use a constant mutation rate when spawning children in pruning algorithms.
        child
    }
}

impl Prunable for MyEntity {
    fn despawn(self) {
        println!("{:?} died", self);
    }
}

impl GenerateRandom for MyEntity {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self { field1: rng.gen() }
    }
}
```

Once you have a struct, you must create your fitness function:
```rust
fn my_fitness_fn(ent: &MyEntity) -> f32 {
    ent.field1 // this just means that the algorithm will try to create as big a number as possible due to fitness being directly taken from the field.
}
```


Once you have your reward function, you can create a `GeneticSim` object to manage and control the evolutionary steps:

```rust
fn main() {
    let mut rng = rand::thread_rng();
    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 1000), // must provide a random starting population. size will be preserved in builtin nextgen fns. note that you do not need to specify a type for `Vec::gen_random` because of the input of `my_fitness_fn`.
        my_fitness_fn,
        asexual_pruning_nextgen,
    );

    // perform evolution (100 gens)
    for _ in 0..100 {
        sim.next_generation(); // in a complex genetic algorithm, you will want to utilize `sim.entities` to test them and generate a reward.
    }

    dbg!(sim.entities);
}
```

That is the minimal code for a working pruning-based genetic algorithm. You can [read the docs](https://docs.rs/genetic-rs) or [check the examples](/examples/) for more complicated systems.

### License
This project falls under the `MIT` license.