# genetic-rs
A crate for quickstarting genetic algorithm projects

### How to Use
First off, this crate comes with the `builtin` module by default. If you want to add the builtin sexual reproduction extension, you can do so by adding the `sexual` feature.

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

// required for `asexual_scrambling_nextgen` and `asexual_pruning_nextgen`. TODO change source
impl ASexualEntity for MyEntity {
    fn spawn_child(&self) -> Self {
        let mut child = self.clone();
        child.mutate(0.25); // use a constant mutation rate when spawning children in pruning algorithms.
        child
    }
}

impl Prunable for MyEntity {
    fn despawn(&mut self) { // TODO make despawn consuming
        println!("{:?} died", self);
    }
}
```

Once you have a struct, you must create your reward function:
```rust
fn my_reward_fn(ent: &MyEntity) -> f32 {
    ent.field1 // this just means that the algorithm will try to create as big a number as possible due to reward being directly taken from the field.
}
```


Once you have your reward function, you can create a `GeneticSim` object to manage and control the evolutionary steps:

```rust
fn main() {
    // need to define a starting population of random entities. the simulation should always retain the same population size.
    let population = (0..100) // population size is 100
        .into_iter()
        .map(|_| MyEntity { field1: fastrand::f32() })
        .collect();

    let mut sim = GeneticSim::new(
        population,
        my_reward_fn,

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