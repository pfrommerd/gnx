use gnx_io::util::RawStrSource;
use gnx_io::json::{JsonParser, Value, Map, Number};
use gnx_graph::Deserialize;

#[test]
fn test_json_parse() {
    let json = r#"{
        "name": "John",
        "age": 30,
        "city": "New York"
    }"#;
    let mut parser = JsonParser::new(RawStrSource::from(json));
    let value = Value::deserialize(&mut parser).unwrap();
    assert_eq!(value, Value::Object(Map::from([
        ("name".to_string(), Value::String("John".to_string())),
        ("age".to_string(), Value::Number(Number::from(30))),
        ("city".to_string(), Value::String("New York".to_string())),
    ])));
}