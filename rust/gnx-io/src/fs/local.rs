use std::fs::{File, OpenOptions};
use std::path::{PathBuf as StdPathBuf, Path as StdPath};
use std::io::{Error, ErrorKind};

use super::*;

pub struct LocalFs {
    root: StdPathBuf,
}

impl LocalFs {
    pub fn new<P: AsRef<StdPath>>(root: P) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    fn path_to_std(path: &Path) -> Result<StdPathBuf> {
        let mut std_path = StdPathBuf::new();
        for component in path.as_str().split('/') {
            if component == ".." {
                return Err(Error::new(ErrorKind::InvalidInput, "Invalid path, contains '..'"));
            } else if !component.is_empty() || component == "." {
                std_path.push(component);
            }
        }
        Ok(std_path)
    }
}


impl Origin for LocalFs {
    type Reader = File;
    type Writer = File;
    type Resource<'s> = LocalResource<'s> where Self: 's;

    fn get<'s>(&'s self, path: &Path) -> Result<Self::Resource<'s>> {
        Ok(LocalResource {
            origin: self,
            path: self.root.join(Self::path_to_std(path)?),
        })
    }
}

pub struct LocalResource<'origin> {
    origin: &'origin LocalFs,
    path: StdPathBuf,
}

impl<'origin> Resource<'origin> for LocalResource<'origin> {
    type Origin = LocalFs;

    fn origin(&self) -> &'origin Self::Origin {
        self.origin
    }

    fn relative(&self, path: &Path) -> Result<<Self::Origin as Origin>::Resource<'origin>> {
        let buf = LocalFs::path_to_std(path)?;
        Ok(LocalResource {
            origin: self.origin,
            path: self.path.join(buf)
        })
    }

    fn create(&self) -> Result<<Self::Origin as Origin>::Writer> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        Ok(file)
    }

    fn append(&self) -> Result<<Self::Origin as Origin>::Writer> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.path)?;
        Ok(file)
    }

    fn read(&self) -> Result<<Self::Origin as Origin>::Reader> {
        let file = File::open(&self.path)?;
        Ok(file)
    }
}