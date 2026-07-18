//! See [llama-cpp-4](https://crates.io/crates/llama-cpp-4) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

/// CPU instruction profile compiled into the vendored llama.cpp library.
pub const TRIMPROMPT_CPU_ENGINE: &str = if cfg!(feature = "avx512") {
    "avx512"
} else if cfg!(feature = "avx2") {
    "avx2"
} else {
    "compatible"
};

#[allow(unnecessary_transmutes)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;

pub mod common;
