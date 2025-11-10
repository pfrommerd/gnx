pub use std::io::{IoSlice, IoSliceMut};


use std::io::Result;

mod path;

pub use path::*;

pub mod local;

pub trait Origin {
    type Reader: std::io::Read + std::io::Seek;
    type Writer: std::io::Write;

    type Resource<'s>: Resource<'s, Origin = Self> + 's where Self: 's;

    fn get<'origin>(&'origin self, path: &Path) -> Result<Self::Resource<'origin>>;
}

pub trait Resource<'origin> {
    type Origin: Origin;

    fn origin(&self) -> &'origin Self::Origin;
    fn relative(&self, path: &Path) -> Result<<Self::Origin as Origin>::Resource<'origin>>;

    fn create(&self) -> Result<<Self::Origin as Origin>::Writer>;
    fn append(&self) -> Result<<Self::Origin as Origin>::Writer>;

    fn read(&self) -> Result<<Self::Origin as Origin>::Reader>;
}