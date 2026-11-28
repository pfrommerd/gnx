use gnx_io::{
    Deserialize, Serialize,
    json::{self, JsonParser},
    util::StrSource,
};

fn parse<'a, T: Deserialize<'a>>(s: &'a str) -> Result<T, json::JsonError> {
    T::deserialize(&mut JsonParser::new(StrSource::from(s)))
}
fn check<T>(item: T)
where
    T: Serialize + serde::Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
{
    let wire = serde_json::to_string(&item).unwrap();
    assert_eq!(json::to_string(&item).unwrap(), wire);
    assert_eq!(parse::<T>(&wire).unwrap(), item);
}
macro_rules! fixture {
    ($item:item) => {
        #[derive(Debug, PartialEq, Serialize, Deserialize, serde::Serialize)]
        $item
    };
}
fixture!(
    struct Unit;
);
fixture!(
    struct Id<T>(T);
);
fixture!(
    struct Point(i32, i32);
);
fixture!(
    #[serde(rename_all = "snake_case", rename_all_fields = "camelCase")]
    enum Event<T> {
        Stop,
        One(T),
        Pair(T, T),
        Named { user_id: T },
    }
);
fixture!(
    #[serde(tag = "kind")]
    enum Internal {
        Stop,
        One(Details),
        Named { x: u32 },
    }
);
fixture!(
    #[serde(tag = "t", content = "v")]
    enum Adj {
        Stop,
        One(i32),
        Pair(u32, u32),
        Named { x: u32 },
    }
);
fixture!(
    #[serde(untagged)]
    enum Untag {
        N(u32),
        S(String),
        Pair(i32, i32),
        Map { x: u32 },
        Null,
    }
);
fixture!(
    struct Details {
        #[serde(alias = "old")]
        x: u32,
    }
);
fixture!(
    struct Flat {
        first: u32,
        #[serde(flatten)]
        details: Details,
    }
);
fixture!(
    struct Envelope {
        event: Event<u32>,
    }
);
#[test]
fn shapes() {
    check(Unit);
    check(Id(7));
    check(Point(-2, 3));
    check(Event::<u32>::Stop);
    check(Event::One(9));
    check(Event::Pair(1, 2));
    check(Event::Named { user_id: 17 });
}
#[test]
fn tagging() {
    check(Internal::Stop);
    check(Internal::One(Details { x: 3 }));
    check(Internal::Named { x: 4 });
    check(Adj::Stop);
    check(Adj::One(3));
    check(Adj::Pair(4, 5));
    check(Adj::Named { x: 7 });
    assert_eq!(parse::<Adj>(r#"{"v":2,"t":"One"}"#).unwrap(), Adj::One(2));
    check(Untag::N(4));
    check(Untag::S("x".into()));
    check(Untag::Pair(1, 2));
    check(Untag::Map { x: 1 });
    check(Untag::Null);
}
#[test]
fn flatten() {
    check(Flat {
        first: 1,
        details: Details { x: 2 },
    });
    assert_eq!(
        parse::<Flat>(r#"{"old":1,"first":2}"#).unwrap().details.x,
        1
    );
}
#[test]
fn flattened_fields_share_unclaimed_entries() {
    #[derive(Deserialize)]
    struct Header<'a> {
        name: &'a str,
    }
    #[derive(Deserialize)]
    struct Message<'a> {
        #[serde(flatten, borrow)]
        header: Header<'a>,
        #[serde(flatten)]
        details: Details,
        #[serde(flatten)]
        extra: std::collections::HashMap<String, u32>,
    }
    let src = String::from(r#"{"extra":7,"old":2,"name":"test"}"#);
    let m: Message = parse(&src).unwrap();
    assert_eq!(
        m.header.name.as_ptr(),
        src[src.find("test").unwrap()..].as_ptr()
    );
    assert_eq!(m.details.x, 2);
    assert_eq!(
        m.extra.into_iter().collect::<Vec<_>>(),
        [("extra".to_owned(), 7)]
    );
}

#[test]
fn flattened_variants() {
    #[derive(Debug, PartialEq, Serialize, Deserialize, serde::Serialize)]
    struct Wrapped {
        #[serde(flatten)]
        event: Event<u32>,
    }
    for event in [
        Event::One(1),
        Event::Pair(2, 3),
        Event::Named { user_id: 4 },
    ] {
        check(Wrapped { event });
    }
}

#[test]
fn malformed_enums() {
    for s in [
        r#""one""#,
        r#"{"one":2,"stop":null}"#,
        r#"{"stop":1}"#,
        r#"{"pair":[3]}"#,
        r#"{"pair":[1,2,3]}"#,
        r#"{"missing":{}}"#,
    ] {
        assert!(parse::<Event<u32>>(s).is_err(), "{s}");
    }
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Borrowed<'a> {
    name: &'a str,
}
#[test]
fn borrowed() {
    let text = String::from(r#"{"name":"abc"}"#);
    let b: Borrowed = parse(&text).unwrap();
    assert_eq!(b.name.as_ptr(), text[9..].as_ptr());
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Attrs {
    #[serde(default = "default_n", rename = "n", alias = "old")]
    number: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    opt: Option<i32>,
    #[serde(skip)]
    hidden: String,
}
fn default_n() -> i32 {
    9
}
#[test]
fn field_policy() {
    let a: Attrs = parse(r#"{"extra":[{"nested":[0]},{}]}"#).unwrap();
    assert_eq!(a.number, 9);
    assert_eq!(json::to_string(&a).unwrap(), r#"{"n":9}"#);
    assert!(parse::<Attrs>(r#"{"n":1,"old":2}"#).is_err());
}
#[test]
fn toml_enum() {
    use gnx_io::toml::{self, TomlParser};
    for event in [
        Event::Stop,
        Event::One(1),
        Event::Pair(2, 3),
        Event::Named { user_id: 7 },
    ] {
        let e = Envelope { event };
        let wire = toml::to_string(&e).unwrap();
        assert_eq!(
            Envelope::deserialize(&mut TomlParser::new(StrSource::from(wire.as_str()))).unwrap(),
            e
        );
    }
}
