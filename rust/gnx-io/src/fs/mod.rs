pub use std::io::{Error, Result, ErrorKind};
pub use std::io::{Seek, Read, Write};

mod path;

pub use path::*;

pub mod local;

pub trait Origin {
    type Reader: Read + Seek;
    type Writer: Write;

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




// A wrapper around std::io::Read that
// 
pub struct StdRead<R: std::io::Read> {
    reader: R,
    buffer: Vec<u8>
}

impl<R: std::io::Read> Read for StdRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        self.reader.read(buf)
    }
}