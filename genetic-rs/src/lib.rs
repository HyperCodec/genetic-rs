#![allow(clippy::needless_doctest_main)]
#![cfg_attr(publish, doc = include_str!(env!("CARGO_PKG_README")))]
#![cfg_attr(not(publish), doc = include_str!(concat!("../", env!("CARGO_PKG_README"))))]

pub mod prelude {
    pub use genetic_rs_common::{self, prelude::*};

    #[cfg(feature = "derive")]
    pub use genetic_rs_macros::*;
}

pub use genetic_rs_common::*;

#[cfg(feature = "derive")]
pub use genetic_rs_macros::*;

pub extern crate genetic_rs_common;
