use std::borrow::Borrow;


// A path is a slice of slash-free components
#[repr(transparent)]
pub struct PathBuf {
    value: String
}
#[repr(transparent)]
pub struct Path {
    value: str
}

impl Path {
    pub fn new<S: AsRef<str>>(value: &S) -> &Self {
        // SAFETY: Path is #[repr(transparent)] and str is #[repr(transparent)]
        unsafe {
            &*(value.as_ref() as *const str as *const Path)
        }
    }
    pub fn as_str(&self) -> &str {
        &self.value
    }
}
impl std::ops::Deref for PathBuf {
    type Target = Path;
    #[inline]
    fn deref(&self) -> &Path {
        Path::new(&self.value)
    }
}
impl Borrow<Path> for PathBuf {
    fn borrow(&self) -> &Path { self }
}
impl AsRef<Path> for PathBuf {
    fn as_ref(&self) -> &Path { self }
}