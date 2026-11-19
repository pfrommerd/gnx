"""Names of the element types supported by GNX arrays."""

from typing import Literal as _Literal

DType = _Literal[
    "bool",
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64",
    "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu",
    "complex64", "complex128",
]

bool = "bool"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
uint64 = "uint64"
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
float16 = "float16"
float32 = "float32"
float64 = "float64"
complex64 = "complex64"
complex128 = "complex128"
bfloat16 = "bfloat16"
float8_e4m3fn = "float8_e4m3fn"
float8_e5m2 = "float8_e5m2"
float8_e8m0fnu = "float8_e8m0fnu"
