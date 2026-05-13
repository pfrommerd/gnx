use gnx_graph::Deserialize;
use gnx_io::toml::{Map, TomlParser, Value, to_string, value_to_string};
use gnx_io::util::RawStrSource;

#[test]
fn test_toml_parse_document() {
    let toml = r#"
title = "TOML Example"
enabled = true
answer = 42
hex = 0x2A
pi = 3.1415
born = 1979-05-27T07:32:00Z
colors = ["red", 'green', """blue"""]
owner.name = "Tom"
owner.bio = """
Line one\
  line two"""

[database]
ports = [8000, 8001, 8002]
connection = { host = "localhost", retry = true }

[[products]]
name = "Hammer"
sku = 738594937

[[products]]
name = "Nail"
sku = 284758393
"#;

    let parser = TomlParser::new(RawStrSource::from(toml));
    let value = parser.into_value().unwrap();
    let root = value.as_table().unwrap();

    assert_eq!(root["title"], Value::String("TOML Example".to_string()));
    assert_eq!(root["enabled"], Value::Bool(true));
    assert_eq!(root["answer"], Value::Integer(42));
    assert_eq!(root["hex"], Value::Integer(42));
    assert_eq!(
        root["born"],
        Value::OffsetDateTime("1979-05-27T07:32:00Z".to_string())
    );

    let owner = root["owner"].as_table().unwrap();
    assert_eq!(owner["name"], Value::String("Tom".to_string()));
    assert_eq!(owner["bio"], Value::String("Line oneline two".to_string()));

    let database = root["database"].as_table().unwrap();
    assert_eq!(
        database["ports"],
        Value::Array(vec![
            Value::Integer(8000),
            Value::Integer(8001),
            Value::Integer(8002),
        ])
    );

    let products = match &root["products"] {
        Value::Array(products) => products,
        value => panic!("expected products array, got {value:?}"),
    };
    assert_eq!(products.len(), 2);
    assert_eq!(
        products[1].as_table().unwrap()["name"],
        Value::String("Nail".to_string())
    );
}

#[test]
fn test_toml_deserialize_value() {
    let toml = r#"
name = "gnx"
features = ["json", "toml"]
metadata = { stable = false, version = 1 }
"#;

    let mut parser = TomlParser::new(RawStrSource::from(toml));
    let value = Value::deserialize(&mut parser).unwrap();
    assert_eq!(
        value,
        Value::Table(Map::from([
            (
                "features".to_string(),
                Value::Array(vec![
                    Value::String("json".to_string()),
                    Value::String("toml".to_string()),
                ]),
            ),
            (
                "metadata".to_string(),
                Value::Table(Map::from([
                    ("stable".to_string(), Value::Bool(false)),
                    ("version".to_string(), Value::Integer(1)),
                ])),
            ),
            ("name".to_string(), Value::String("gnx".to_string())),
        ]))
    );
}

#[test]
fn test_toml_write_and_round_trip() {
    let value = Value::Table(Map::from([
        ("active".to_string(), Value::Bool(true)),
        (
            "created".to_string(),
            Value::LocalDate("2026-05-13".to_string()),
        ),
        (
            "nested".to_string(),
            Value::Table(Map::from([(
                "message".to_string(),
                Value::String("hello\n\"gnx\"".to_string()),
            )])),
        ),
        (
            "values".to_string(),
            Value::Array(vec![Value::Integer(1), Value::Float(2.5.into())]),
        ),
    ]));

    let toml = value_to_string(&value).unwrap();
    assert_eq!(
        toml,
        "active = true\ncreated = 2026-05-13\nnested = { message = \"hello\\n\\\"gnx\\\"\" }\nvalues = [1, 2.5]"
    );

    let parsed = TomlParser::new(RawStrSource::from(toml.as_str()))
        .into_value()
        .unwrap();
    assert_eq!(parsed, value);
}

#[test]
fn test_toml_generic_writer() {
    let value = Value::Table(Map::from([
        ("items".to_string(), Value::Array(vec![Value::Integer(1)])),
        ("name".to_string(), Value::String("gnx".to_string())),
    ]));

    let toml = to_string(&value).unwrap();
    assert_eq!(toml, "items = [1]\nname = \"gnx\"");
}

#[test]
fn test_toml_rejects_duplicate_keys() {
    let parser = TomlParser::new(RawStrSource::from("name = \"Tom\"\nname = \"Pradyun\""));
    assert!(parser.into_value().is_err());
}
