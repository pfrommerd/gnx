pub trait Error : Sized + std::error::Error {
    fn custom<T: Display>(msg: T) -> Self;


    fn invalid_type(unexp_value: Option<impl Display>, unexp_type: impl Display, expected: impl Display) -> Self {
        let _ = unexp_value;
        Self::custom(format!("Invalid type, got {unexp_type}, expected {expected}"))
    }
    fn invalid_leaf() -> Self {
        Self::custom("Invalid leaf")
    }
    fn invalid_id(id: GraphId) -> Self {
        Self::custom(format!("Invalid ID: {id}"))
    }

    fn missing_leaf() -> Self { Self::custom("Missing leaf") }
    fn missing_static_leaf() -> Self { Self::custom("Missing static leaf") }
    fn missing_node() -> Self { Self::custom("Missing node") }
    fn missing_child(key: Key) -> Self { Self::custom(format!("Missing child {key}")) }

    fn expected_leaf() -> Self { Self::custom("Expected leaf") }
    fn expected_static_leaf() -> Self { Self::custom("Expected static leaf") }
    fn expected_node() -> Self { Self::custom("Expected node") }
}