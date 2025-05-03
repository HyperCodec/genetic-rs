pub extern crate rand;

pub use crate::*;

#[cfg(feature = "builtin")]
pub use crate::builtin_old::*;

#[cfg(feature = "builtin")]
pub use next_gen::*;

pub use rand::Rng as RandRng;
