use gnx_io::{
    Deserialize, Serialize,
    json::{JsonParser, to_string},
    util::StrSource,
};
fn parse<'a, T: Deserialize<'a>>(s: &'a str) -> T {
    T::deserialize(&mut JsonParser::new(StrSource::from(s))).unwrap()
}

#[derive(Debug, PartialEq)]
struct Foreign(i32);
impl Foreign {
    fn get(&self) -> i32 {
        self.0
    }
}
#[derive(Serialize, Deserialize)]
#[serde(remote = "Foreign")]
struct Mirror(#[serde(getter = "Foreign::get")] i32);
impl From<Mirror> for Foreign {
    fn from(m: Mirror) -> Self {
        Foreign(m.0)
    }
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct UsesRemote {
    #[serde(with = "Mirror")]
    value: Foreign,
}
#[test]
fn remote() {
    let v = UsesRemote { value: Foreign(13) };
    let s = to_string(&v).unwrap();
    assert_eq!(s, r#"{"value":13}"#);
    assert_eq!(parse::<UsesRemote>(&s), v);
}

mod pair {
    use gnx_io::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(a: &u8, b: &u8, s: S) -> Result<S::Ok, S::Error> {
        (u16::from(*a) * 256 + u16::from(*b)).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<(u8, u8), D::Error> {
        let n = u16::deserialize(d)?;
        Ok(((n >> 8) as u8, n as u8))
    }
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Hook {
    #[serde(with = "pair")]
    Packed(u8, u8),
}
#[test]
fn variant_hooks() {
    let h = Hook::Packed(1, 2);
    let s = to_string(&h).unwrap();
    assert_eq!(s, r#"{"Packed":258}"#);
    assert_eq!(parse::<Hook>(&s), h);
}
#[derive(Deserialize, Debug, PartialEq)]
#[serde(field_identifier, rename_all = "snake_case")]
enum Keys {
    KnownKey,
    Other(String),
}
#[test]
fn identifiers() {
    assert_eq!(parse::<Keys>(r#""known_key""#), Keys::KnownKey);
    assert_eq!(parse::<Keys>(r#""extra""#), Keys::Other("extra".into()));
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "i32", into = "i32")]
struct Positive(i32);
impl TryFrom<i32> for Positive {
    type Error = &'static str;
    fn try_from(v: i32) -> Result<Self, Self::Error> {
        if v > 0 {
            Ok(Self(v))
        } else {
            Err("not positive")
        }
    }
}
impl From<Positive> for i32 {
    fn from(p: Positive) -> Self {
        p.0
    }
}
#[test]
fn conversions() {
    assert_eq!(to_string(&Positive(4)).unwrap(), "4");
    assert_eq!(parse::<Positive>("4"), Positive(4));
    assert!(Positive::deserialize(&mut JsonParser::new(StrSource::from("0"))).is_err());
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
struct Transparent<T> {
    value: T,
    #[serde(skip)]
    context: usize,
}
#[test]
fn transparent() {
    assert_eq!(
        parse::<Transparent<u8>>("3"),
        Transparent {
            value: 3,
            context: 0
        }
    );
    assert_eq!(
        to_string(&Transparent {
            value: 4,
            context: 7
        })
        .unwrap(),
        "4"
    );
}
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))]
struct Node<T> {
    value: T,
    children: Vec<Node<T>>,
}
#[test]
fn borrowed_cow_and_const_generics() {
    use std::borrow::Cow;
    #[derive(Serialize, Deserialize)]
    struct Item<'a, const N: usize> {
        #[serde(borrow)]
        label: Cow<'a, str>,
        data: [u8; N],
    }
    let item: Item<2> = parse(r#"{"label":"plain","data":[2,3]}"#);
    assert!(matches!(item.label, Cow::Borrowed("plain")));
    assert_eq!(item.data, [2, 3]);
    let item: Item<0> = parse(r#"{"label":"a\nb","data":[]}"#);
    assert!(matches!(item.label, Cow::Owned(_)));
}

#[test]
fn recursive_bounds() {
    let n: Node<u8> = parse(r#"{"value":3,"children":[{"value":4,"children":[]}]}"#);
    assert_eq!(n.children[0].value, 4);
    assert!(to_string(&n).unwrap().contains("children"));
}
