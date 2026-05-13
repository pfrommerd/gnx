use gnx_graph::{Deserialize, Serialize};
use gnx_io::json::{JsonError, JsonParser, Map, Number, Value, to_string};
use gnx_io::util::RawStrSource;

fn parse_json(json: &str) -> Result<Value, JsonError> {
    let mut parser = JsonParser::new(RawStrSource::from(json));
    Value::deserialize(&mut parser)
}

#[test]
fn test_json_parse() {
    let json = r#"{
        "name": "John",
        "age": 30,
        "city": "New York"
    }"#;
    let value = parse_json(json).unwrap();
    assert_eq!(
        value,
        Value::Object(Map::from([
            ("name".to_string(), Value::String("John".to_string())),
            ("age".to_string(), Value::Number(Number::from(30))),
            ("city".to_string(), Value::String("New York".to_string())),
        ]))
    );
}

#[test]
fn test_json_literals_and_whitespace() {
    let value = parse_json(
        r#"
        {
            "null": null,
            "true": true,
            "false": false,
            "empty_object": {},
            "empty_array": []
        }
        "#,
    )
    .unwrap();

    assert_eq!(
        value,
        Value::Object(Map::from([
            ("empty_array".to_string(), Value::Array(vec![])),
            ("empty_object".to_string(), Value::Object(Map::new())),
            ("false".to_string(), Value::Bool(false)),
            ("null".to_string(), Value::Null),
            ("true".to_string(), Value::Bool(true)),
        ]))
    );
}

#[test]
fn test_json_string_escapes_and_unicode() {
    let value = parse_json(
        r#"{
            "escapes": "\"\\\/\b\f\n\r\t",
            "latin": "\u00E9",
            "g_clef": "\uD834\uDD1E"
        }"#,
    )
    .unwrap();

    assert_eq!(
        value,
        Value::Object(Map::from([
            (
                "escapes".to_string(),
                Value::String("\"\\/\u{08}\u{0c}\n\r\t".to_string())
            ),
            ("g_clef".to_string(), Value::String("𝄞".to_string())),
            ("latin".to_string(), Value::String("é".to_string())),
        ]))
    );
}

#[test]
fn test_json_numbers() {
    let value = parse_json(
        r#"[
            0,
            -0,
            42,
            -17,
            1.25,
            -2E-2,
            1e3,
            6.022e23
        ]"#,
    )
    .unwrap();

    assert_eq!(
        value,
        Value::Array(vec![
            Value::Number(Number::from(0)),
            Value::Number(Number::from(0)),
            Value::Number(Number::from(42)),
            Value::Number(Number::from(-17)),
            Value::Number(Number::from(1.25)),
            Value::Number(Number::from(-0.02)),
            Value::Number(Number::from(1000)),
            Value::Number(Number::from(6.022e23)),
        ])
    );
}

#[test]
fn test_json_nested_arrays_and_objects_require_commas() {
    let value = parse_json(r#"{"items":[{"id":1},{"id":2}],"nested":[[true],[]]}"#).unwrap();
    assert_eq!(
        value,
        Value::Object(Map::from([
            (
                "items".to_string(),
                Value::Array(vec![
                    Value::Object(Map::from([(
                        "id".to_string(),
                        Value::Number(Number::from(1)),
                    )])),
                    Value::Object(Map::from([(
                        "id".to_string(),
                        Value::Number(Number::from(2)),
                    )])),
                ]),
            ),
            (
                "nested".to_string(),
                Value::Array(vec![
                    Value::Array(vec![Value::Bool(true)]),
                    Value::Array(vec![]),
                ]),
            ),
        ]))
    );

    for invalid in ["[1 2]", "[1,]", r#"{"a":1 "b":2}"#, r#"{"a":1,}"#] {
        assert!(
            parse_json(invalid).is_err(),
            "{invalid} should be invalid JSON"
        );
    }
}

#[test]
fn test_json_rejects_invalid_standard_forms() {
    for invalid in [
        "+1",
        "01",
        "1.",
        "1e",
        "tru",
        "nul",
        "\"unterminated",
        "\"bad\\xescape\"",
        "\"bad\nnewline\"",
        "\"bad\\uD800\"",
        "\"bad\\uD800\\u0000\"",
    ] {
        assert!(
            parse_json(invalid).is_err(),
            "{invalid:?} should be invalid JSON"
        );
    }
}

#[test]
fn test_json_write() {
    let value = Value::Object(Map::from([
        (
            "message".to_string(),
            Value::String("hello\n\"gnx\"".to_string()),
        ),
        ("ok".to_string(), Value::Bool(true)),
        (
            "items".to_string(),
            Value::Array(vec![Value::Null, Value::Number(Number::from(3))]),
        ),
    ]));

    let json = to_string(&value).unwrap();
    assert_eq!(
        json,
        r#"{"items":[null,3],"message":"hello\n\"gnx\"","ok":true}"#
    );

    let mut parser = JsonParser::new(RawStrSource::from(json.as_str()));
    assert_eq!(Value::deserialize(&mut parser).unwrap(), value);
}

#[test]
fn test_json_writer_escapes_strings_and_rejects_nonfinite_numbers() {
    let value = Value::String("quote \" slash \\ control \u{08}\n".to_string());
    assert_eq!(
        to_string(&value).unwrap(),
        r#""quote \" slash \\ control \b\n""#
    );

    #[derive(Clone, Copy)]
    struct Infinite;

    impl Serialize for Infinite {
        fn serialize<S: gnx_graph::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_f64(f64::INFINITY)
        }
    }

    assert!(to_string(&Infinite).is_err());
}
