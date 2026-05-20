use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::Dialect;

pub struct Dialects {
    dialects: BTreeMap<&'static str, &'static dyn Dialect>,
}

impl Dialects {
    pub fn get(&self, name: &str) -> Option<&'static dyn Dialect> {
        self.dialects.get(name).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &'static dyn Dialect)> + '_ {
        self.dialects
            .iter()
            .map(|(name, dialect)| (*name, *dialect))
    }

    pub fn len(&self) -> usize {
        self.dialects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dialects.is_empty()
    }
}

pub struct DialectRegistration {
    dialect: &'static dyn Dialect,
}

impl DialectRegistration {
    pub const fn new(dialect: &'static dyn Dialect) -> Self {
        Self { dialect }
    }

    pub fn dialect(&self) -> &'static dyn Dialect {
        self.dialect
    }
}

inventory::collect!(DialectRegistration);

#[macro_export]
macro_rules! register_dialect {
    ($dialect:expr $(,)?) => {
        $crate::__private::inventory::submit! {
            $crate::DialectRegistration::new($dialect)
        }
    };
}
static DIALECTS: LazyLock<Dialects> = LazyLock::new(|| Dialects {
    dialects: inventory::iter::<DialectRegistration>
        .into_iter()
        .map(|registration| (registration.dialect().name(), registration.dialect()))
        .collect(),
});

pub fn dialects() -> &'static Dialects {
    &*DIALECTS
}
