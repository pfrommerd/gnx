//! Derive [`Serialize`] and [`Deserialize`] for structs and enums using `#[serde(...)]`
//! attributes. The generated implementations use GNX's I/O traits, including for
//! field and variant hooks. For example, this writes `{"kind":"batch","items":[1,2]}`:
//!
//! ```
//! use gnx_io::{Deserialize, Serialize};
//! #[derive(Serialize, Deserialize)]
//! #[serde(tag = "kind", content = "items", rename_all = "snake_case")]
//! enum Update { Done, Batch(Vec<u32>) }
//! assert_eq!(gnx_io::json::to_string(&Update::Batch(vec![1, 2])).unwrap(),
//!            r#"{"kind":"batch","items":[1,2]}"#);
//! ```
//!
//! Named structs ignore unknown fields unless `deny_unknown_fields` is set.
//! Missing `Option` fields default to `None`; other fields require a value or a default.
//! Untagged enums try variants in declaration order and need self-describing input.
//!
//! Internally tagged enums cannot contain tuple variants:
//! ```compile_fail
//! use gnx_io::Serialize;
//! #[derive(Serialize)]
//! #[serde(tag = "type")]
//! enum Invalid { Pair(u8, u8) }
//! ```
extern crate self as gnx_io;
pub use gnx_graph::Error;
pub use gnx_io_derive::{Deserialize, Serialize};
pub mod des;
pub mod json;
pub mod ser;
pub mod toml;
pub mod util;

pub use des::*;
pub use gnx_base::store;
pub use ser::*;
