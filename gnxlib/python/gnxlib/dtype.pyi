from typing import Final, Literal, TypeAlias

DType: TypeAlias = Literal[
    "bool",
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float16", "float32", "float64",
    "complex64", "complex128",
]

bool: Final = "bool"
uint8: Final = "uint8"
uint16: Final = "uint16"
uint32: Final = "uint32"
uint64: Final = "uint64"
int8: Final = "int8"
int16: Final = "int16"
int32: Final = "int32"
int64: Final = "int64"
float16: Final = "float16"
float32: Final = "float32"
float64: Final = "float64"
complex64: Final = "complex64"
complex128: Final = "complex128"
