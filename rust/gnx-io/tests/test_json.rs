use gnx_graph::Deserialize;
use gnx_io::json::{JsonParser, Map, Number, Value, to_string};
use gnx_io::util::RawStrSource;

#[test]
fn test_json_parse() {
    let json = r#"{
        "name": "John",
        "age": 30,
        "city": "New York"
    }"#;
    let mut parser = JsonParser::new(RawStrSource::from(json));
    let value = Value::deserialize(&mut parser).unwrap();
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
