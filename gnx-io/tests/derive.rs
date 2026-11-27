use gnx_io::{
    Deserialize, Serialize,
    json::{JsonError, JsonParser, to_string},
    util::StrSource,
};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Pair<T> {
    first: T,
    second: Option<T>,
}

fn parse(text: &str) -> Result<Pair<usize>, JsonError> {
    Pair::deserialize(&mut JsonParser::new(StrSource::from(text)))
}

#[test]
fn fields_are_named_and_required() {
    let pair = Pair {
        first: 7usize,
        second: None,
    };
    assert_eq!(parse(&to_string(&pair).unwrap()).unwrap(), pair);
    assert_eq!(parse(r#"{"second":null,"first":7}"#).unwrap(), pair);
    for (text, message) in [
        (r#"{"second":null}"#, "missing field: first"),
        (
            r#"{"first":7,"first":8,"second":null}"#,
            "duplicate field: first",
        ),
    ] {
        assert!(parse(text).unwrap_err().to_string().contains(message));
    }
    assert!(parse(r#"{"first":-1,"second":null}"#).is_err());
}
