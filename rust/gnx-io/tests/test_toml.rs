use gnx_graph::Deserialize;
use gnx_io::toml::TomlError;
use gnx_io::toml::{Map, TomlParser, Value, to_string, value_to_string};
use gnx_io::util::RawStrSource;

fn parse_toml(toml: &str) -> Result<Value, TomlError> {
    TomlParser::new(RawStrSource::from(toml)).into_value()
}

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

    let value = parse_toml(toml).unwrap();
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
fn test_toml_strings_and_keys() {
    let toml = r#"
bare-key = "bare"
"quoted key"."spaced key" = "quoted"
literal = 'C:\Users\nodejs\templates'
basic = "tab\tnewline\nescape\ehex\xE9"
multiline_basic = """
Roses are red\
  violets are blue"""
multiline_literal = '''
The first newline is trimmed.
Backslashes \ are literal.'''
edge_basic = """"quoted at both ends""""
edge_literal = ''''quoted at both ends''''
"#;

    let value = parse_toml(toml).unwrap();
    let root = value.as_table().unwrap();

    assert_eq!(root["bare-key"], Value::String("bare".to_string()));
    assert_eq!(
        root["basic"],
        Value::String("tab\tnewline\nescape\u{1b}hexé".to_string())
    );
    assert_eq!(
        root["literal"],
        Value::String(r"C:\Users\nodejs\templates".to_string())
    );
    assert_eq!(
        root["multiline_basic"],
        Value::String("Roses are redviolets are blue".to_string())
    );
    assert_eq!(
        root["multiline_literal"],
        Value::String("The first newline is trimmed.\nBackslashes \\ are literal.".to_string())
    );
    assert_eq!(
        root["edge_basic"],
        Value::String("\"quoted at both ends\"".to_string())
    );
    assert_eq!(
        root["edge_literal"],
        Value::String("'quoted at both ends'".to_string())
    );

    let quoted = root["quoted key"].as_table().unwrap();
    assert_eq!(quoted["spaced key"], Value::String("quoted".to_string()));
}

#[test]
fn test_toml_numbers_and_datetime_values() {
    let toml = r#"
pos = +99
neg = -17
grouped = 1_000_000
hex = 0xDEAD_BEEF
oct = 0o755
bin = 0b1101_0110
flt = 224_617.445_991_228
exp = -2E-2
inf = +inf
neg_inf = -inf
nan = nan
offset = 1979-05-27 07:32Z
local_dt = 1979-05-27T07:32
local_date = 1979-05-27
local_time = 07:32
"#;

    let value = parse_toml(toml).unwrap();
    let root = value.as_table().unwrap();

    assert_eq!(root["pos"], Value::Integer(99));
    assert_eq!(root["neg"], Value::Integer(-17));
    assert_eq!(root["grouped"], Value::Integer(1_000_000));
    assert_eq!(root["hex"], Value::Integer(0xDEAD_BEEF));
    assert_eq!(root["oct"], Value::Integer(0o755));
    assert_eq!(root["bin"], Value::Integer(0b1101_0110));
    assert_eq!(root["flt"], Value::Float(224_617.445_991_228.into()));
    assert_eq!(root["exp"], Value::Float((-2E-2).into()));
    assert_eq!(root["inf"], Value::Float(f64::INFINITY.into()));
    assert_eq!(root["neg_inf"], Value::Float(f64::NEG_INFINITY.into()));
    match root["nan"] {
        Value::Float(value) => assert!(value.0.is_nan()),
        ref value => panic!("expected nan float, got {value:?}"),
    }
    assert_eq!(
        root["offset"],
        Value::OffsetDateTime("1979-05-27 07:32Z".to_string())
    );
    assert_eq!(
        root["local_dt"],
        Value::LocalDateTime("1979-05-27T07:32".to_string())
    );
    assert_eq!(
        root["local_date"],
        Value::LocalDate("1979-05-27".to_string())
    );
    assert_eq!(root["local_time"], Value::LocalTime("07:32".to_string()));
}

#[test]
fn test_toml_arrays_comments_and_trailing_commas() {
    let toml = r#"
mixed = [
  1,
  "two", # comments may appear between array elements
  { nested = true },
]
"#;

    let value = parse_toml(toml).unwrap();
    let root = value.as_table().unwrap();
    assert_eq!(
        root["mixed"],
        Value::Array(vec![
            Value::Integer(1),
            Value::String("two".to_string()),
            Value::Table(Map::from([("nested".to_string(), Value::Bool(true))])),
        ])
    );
}

#[test]
fn test_toml_nested_tables_and_arrays_of_tables() {
    let toml = r#"
[[fruits]]
name = "apple"

[fruits.physical]
color = "red"
shape = "round"

[[fruits.varieties]]
name = "red delicious"

[[fruits.varieties]]
name = "granny smith"

[[fruits]]
name = "banana"

[[fruits.varieties]]
name = "plantain"
"#;

    let value = parse_toml(toml).unwrap();
    let fruits = match &value.as_table().unwrap()["fruits"] {
        Value::Array(values) => values,
        value => panic!("expected fruits array, got {value:?}"),
    };

    assert_eq!(fruits.len(), 2);
    let apple = fruits[0].as_table().unwrap();
    assert_eq!(apple["name"], Value::String("apple".to_string()));
    assert_eq!(
        apple["physical"].as_table().unwrap()["color"],
        Value::String("red".to_string())
    );
    let apple_varieties = match &apple["varieties"] {
        Value::Array(values) => values,
        value => panic!("expected varieties array, got {value:?}"),
    };
    assert_eq!(apple_varieties.len(), 2);

    let banana = fruits[1].as_table().unwrap();
    assert_eq!(banana["name"], Value::String("banana".to_string()));
    let banana_varieties = match &banana["varieties"] {
        Value::Array(values) => values,
        value => panic!("expected banana varieties array, got {value:?}"),
    };
    assert_eq!(
        banana_varieties[0].as_table().unwrap()["name"],
        Value::String("plantain".to_string())
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
fn test_toml_multiline_inline_table_v1_1() {
    let toml = r#"
contact = {
    personal = {
        name = "Donald Duck",
        email = "donald@duckburg.com", # comments are allowed between entries
    },
    work = {
        title = "Coin cleaner",
        active = true,
    },
}
"#;

    let value = parse_toml(toml).unwrap();
    let root = value.as_table().unwrap();
    let contact = root["contact"].as_table().unwrap();

    assert_eq!(
        contact["personal"],
        Value::Table(Map::from([
            (
                "email".to_string(),
                Value::String("donald@duckburg.com".to_string()),
            ),
            ("name".to_string(), Value::String("Donald Duck".to_string())),
        ]))
    );
    assert_eq!(
        contact["work"],
        Value::Table(Map::from([
            ("active".to_string(), Value::Bool(true)),
            (
                "title".to_string(),
                Value::String("Coin cleaner".to_string()),
            ),
        ]))
    );

    let rendered = value_to_string(&value).unwrap();
    let reparsed = TomlParser::new(RawStrSource::from(rendered.as_str()))
        .into_value()
        .unwrap();
    assert_eq!(reparsed, value);
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
    assert!(parse_toml("name = \"Tom\"\nname = \"Pradyun\"").is_err());
}

#[test]
fn test_toml_rejects_invalid_standard_forms() {
    for invalid in [
        "bad = 01",
        "bad = 1_",
        "bad = _1",
        "bad = 0x_FF",
        "bad = .7",
        "bad = 7.",
        "bad = \"invalid\\qescape\"",
        "bad = \"invalid\nnewline\"",
        "= \"no key\"",
        "fruit.apple = 1\nfruit.apple.smooth = true",
        "type = { name = \"Nail\" }\ntype.edible = false",
        "[fruit]\n[fruit]",
        "bad = 1979-02-30",
    ] {
        assert!(
            parse_toml(invalid).is_err(),
            "{invalid:?} should be invalid TOML"
        );
    }
}
