use std::fmt;

use rkyv::{
    Archive, Portable, Serialize,
    bytecheck::{self, CheckBytes},
    rend::{i16_le, i32_le, i64_le, u16_le, u32_le, u64_le},
    string::ArchivedString,
};

use crate::symbol::Symbol;
use crate::util::{
    ArchivedDyn, ArchivedTypeTable, DynRef, OwnedDyn, TypeImpls, TypeTable, TypeTableBuilder,
    TypedIn, dyn_archive,
};

/// A variable of an expression: a value, or in a high-level expression
/// possibly a cell. Variables are local to their expression.
#[derive(
    Archive,
    Serialize,
    Portable,
    CheckBytes,
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
#[rkyv(as = Self)]
#[bytecheck(crate = bytecheck)]
#[repr(transparent)]
pub struct Var(u32_le);
impl Var {
    pub fn new(id: u32) -> Self {
        Self(id.into())
    }
    /// A variable that names nothing: as an output of an equation, or as
    /// what a cell is read into, it discards the value. It cannot be used.
    pub fn hole() -> Self {
        Self::new(u32::MAX)
    }
    pub fn is_hole(self) -> bool {
        self.id() == u32::MAX
    }
    pub fn id(self) -> u32 {
        self.0.to_native()
    }
}

/// Allocates the variables and symbols of an expression.
#[derive(Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct VarScope {
    next_var: u32,
    next_symbol: u32,
}

impl VarScope {
    pub fn new() -> VarScope {
        Self::default()
    }

    pub fn create_var(&mut self) -> Var {
        let var = Var::new(self.next_var);
        assert!(!var.is_hole(), "too many variables");
        self.next_var += 1;
        var
    }

    /// Allocate a symbol independently of the variable counter.
    pub fn create_symbol(&mut self) -> Symbol {
        let symbol = Symbol::new(self.next_symbol);
        self.next_symbol = self.next_symbol.checked_add(1).expect("too many symbols");
        symbol
    }

    // Makes every variable and symbol allocated from here on come after
    // the given ones.
    pub(crate) fn reserve(
        &mut self,
        vars: impl IntoIterator<Item = Var>,
        symbols: impl IntoIterator<Item = Symbol>,
    ) {
        for var in vars.into_iter().filter(|var| !var.is_hole()) {
            self.next_var = self
                .next_var
                .max(var.id().checked_add(1).expect("too many variables"));
        }
        for symbol in symbols {
            self.next_symbol =
                (self.next_symbol).max(symbol.id().checked_add(1).expect("too many symbols"));
        }
    }
}

/// Whether a variable of a high-level expression is an immutable value
/// or a mutable cell. Its info is stored separately.
#[derive(Archive, Serialize, Portable, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[rkyv(as = Self)]
#[repr(u8)]
pub enum Kind {
    Value,
    Cell,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Value => "value",
            Self::Cell => "cell",
        })
    }
}

dyn_archive! {
    /// The dialect-defined metadata of a variable: the type of a value, or
    /// of the contents of a cell.
    ///
    /// Implemented by archived types. A concrete info is named in a
    /// dialect, and is made from a builder: a type that archives as it.
    ///
    /// Metadata can refer to symbols of its expression through an ordered
    /// list of symbol slots. The dialect owns the meaning of each slot.
    pub trait Info {
        /// The symbol in each slot, in slot order. A symbol can fill
        /// several slots. A symbol in the info of an input is bound by that
        /// input, unless the expression already declares it as a closure
        /// symbol.
        fn symbols(&self) -> &[Symbol] {
            &[]
        }

        /// Rebuild this info with `f` applied to the symbol in each slot,
        /// in slot order. `None` leaves the info unchanged, which is only
        /// correct for an info without symbols.
        fn remap_symbols(&self, _f: &mut dyn FnMut(Symbol) -> Symbol) -> Option<VarInfo> {
            None
        }

        /// Writes the info as it is shown in an expression.
        fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.type_name())
        }
    }
}

impl dyn Info {
    /// The info, if it is of the (archived) type `A`.
    pub fn downcast_ref<A: TypedIn<dyn Info>>(&self) -> Option<&A> {
        // SAFETY: a name identifies one type among the dialects of Info
        (self.type_name() == A::NAME).then(|| unsafe { &*(self as *const Self).cast::<A>() })
    }
}

impl fmt::Display for dyn Info {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f)
    }
}

/// An owned, cheaply cloned info, made from a builder with `VarInfo::new`.
pub type VarInfo = OwnedDyn<dyn Info>;

/// An info in a root.
pub type VarInfoRef<'a> = DynRef<'a, dyn Info>;

/// The archived form of an info in a root.
pub type VarInfoData = ArchivedDyn<dyn Info>;

impl VarInfo {
    /// Replace the symbol in every slot. See [`Info::remap_symbols`].
    pub fn remap_symbols(&self, mut f: impl FnMut(Symbol) -> Symbol) -> Self {
        (self.get().remap_symbols(&mut f)).unwrap_or_else(|| self.clone())
    }
}

impl From<&VarInfo> for VarInfo {
    fn from(info: &VarInfo) -> Self {
        info.clone()
    }
}

impl fmt::Display for VarInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().display(f)
    }
}

impl fmt::Display for VarInfoRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().display(f)
    }
}

/// What a typed tracer requires of the info of its values: either any
/// info at all, or one particular archived type.
pub trait InfoCast {
    fn cast(info: &VarInfo) -> Option<&Self>;
}

impl InfoCast for VarInfo {
    fn cast(info: &VarInfo) -> Option<&Self> {
        Some(info)
    }
}

impl<A: TypedIn<dyn Info>> InfoCast for A {
    fn cast(info: &VarInfo) -> Option<&Self> {
        info.downcast()
    }
}

// Serialization helpers: InfoTypeTable and InfoTypeImpls

// Used during serialization to build the InfoTypeTable
pub type InfoTypeCtx = TypeTableBuilder<dyn Info>;

// The archived version contains the (dialect, name) values.
// The unarchived version is a zero-sized type.
pub type InfoTypeTable = TypeTable<dyn Info>;
pub type InfoTypeTableData = ArchivedTypeTable<dyn Info>;

// The external table for info ids, resolved
// from an InfoTypeTableData at load time
pub type InfoTypeImpls = TypeImpls<dyn Info>;

// Plain values as symbol-free infos: `VarInfo::new(0usize)`,
// `VarInfo::new("f32")`. These are what the primitives archive as.
macro_rules! prim_infos {
    ($($name:literal => $ty:ty),* $(,)?) => {
        $(impl Info for $ty {
            fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self, f)
            }
        })*
        crate::dialect! {
            dyn Info, "prim" => { $($name => $ty),* }
        }
    };
}

prim_infos! {
    "bool" => bool,
    "u8" => u8,
    "i8" => i8,
    "u16" => u16_le,
    "i16" => i16_le,
    "u32" => u32_le,
    "i32" => i32_le,
    "u64" => u64_le,
    "i64" => i64_le,
    "str" => ArchivedString,
}
