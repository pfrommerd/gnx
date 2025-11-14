use crate::util::{TextSource, Text, ScratchBuffer};

use super::JsonError;

fn hex_pair_to_u8(hex: [u8; 2]) -> Result<u8, JsonError> {
    if hex[0] < b'0' || hex[0] > b'a' || hex[1] < b'0' || hex[1] > b'f' {
        return Err(JsonError::InvalidEscape);
    }
    let a = hex[0] - b'0';
    let b = hex[1] - b'0';
    Ok(a * 16 + b)
}

// Unescape a string of bytes, returning the same string,
// but with any escaped characters replaced with their unescaped equivalents.
// Will mutate the buffer in-place!
pub fn unescape(buffer: String) -> Result<String, JsonError> {
    let mut bytes = buffer.into_bytes();
    let (mut src, mut tgt) = (0, 0);
    loop {
        if src >= bytes.len() { break; }
        if bytes[src] == b'\\' {
            // Skip over the backslash and the escaped character.
            src += 2;
            // If there is no escaped character, return an error.
            if src - 1 >= bytes.len() { return Err(JsonError::UnexpectedEOF); }
            // Handle the escaped character
            match bytes[src - 1] {
                b'"' => bytes[tgt] = b'"',
                b'\\' => bytes[tgt] = b'\\',
                b'/' => bytes[tgt] = b'/',
                b'b' => bytes[tgt] = b'\x08',
                b'f' => bytes[tgt] = b'\x0C',
                b'n' => bytes[tgt] = b'\n',
                b'r' => bytes[tgt] = b'\r',
                b't' => bytes[tgt] = b'\t',
                b'u' => {
                    // Ensure, src through src + 3 are valid indices into bytes.
                    if src + 3 >= bytes.len() { return Err(JsonError::UnexpectedEOF); }
                    let hex_a = [bytes[src], bytes[src + 1] ];
                    let hex_b = [bytes[src + 2], bytes[src + 3] ];
                    bytes[tgt] = hex_pair_to_u8(hex_a)?;
                    tgt += 1;
                    bytes[tgt + 1] = hex_pair_to_u8(hex_b)?;
                },
                _ => return Err(JsonError::InvalidEscape),
            }
            tgt += 1;
            continue;
        } else {
            bytes[tgt] = bytes[src];
            src += 1;
            tgt += 1;
        }
    }
    bytes.truncate(tgt);
    // TODO: Use std::str::from_utf8_unchecked at some point
    // when we feel good about the safety of this code.
    let result = String::from_utf8(bytes).unwrap();
    Ok(result)
}

// Consume an escaped character, erroring if there is an issue with the escaping.
// Assumes the slash has already been consumed.
fn skip_escaped<'src, S: TextSource<'src> + ?Sized>(source: &mut S) -> Result<(), JsonError> {
    let c = match source.next()? {
        None => return Err(JsonError::UnexpectedEOF),
        Some(c) => c
    };
    match c {
        '"' => Ok(()),
        '\\' => Ok(()),
        'b' => Ok(()),
        'f' => Ok(()),
        'n' => Ok(()),
        'r' => Ok(()),
        't' => Ok(()),
        'u' => {
            for _ in 0..4 {
                let c = match source.next()? {
                    Some(c) => c,
                    None => return Err(JsonError::UnexpectedEOF)
                };
                if c < '0' || c > 'f' { return Err(JsonError::InvalidEscape); }
            }
            Ok(())
        },
        _ => Err(JsonError::InvalidEscape),
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ValueType {
    String, Number, Boolean, Null, Object, Array,
}

pub trait JsonSource<'src>: TextSource<'src> {
    fn peek_type(&mut self) -> Result<ValueType, JsonError> {
        let next = match self.peek()? {
            Some(c) => c,
            None => return Err(JsonError::UnexpectedEOF)
        };
        match next {
            '"' => Ok(ValueType::String),
            '0'..='9' => Ok(ValueType::Number),
            't' => Ok(ValueType::Boolean),
            'f' => Ok(ValueType::Boolean),
            'n' => Ok(ValueType::Null),
            '{' => Ok(ValueType::Object),
            '[' => Ok(ValueType::Array),
            _ => return Err(JsonError::Unexpected(next))
        }
    }

    fn skip(&mut self) -> Result<(), JsonError> {
        let consume_item = |s: &mut Self| -> Result<Option<ValueType>, JsonError> { match s.peek_type()?{
            ValueType::String => { s.skip_string()?; Ok(None) },
            ValueType::Number => { s.skip_number()?; Ok(None) },
            ValueType::Boolean => { s.skip_boolean()?; Ok(None) },
            ValueType::Null => { s.skip_null()?; Ok(None) },
            ValueType::Object => {
                s.consume_object_start()?;
                s.consume_whitespace()?;
                Ok(Some(ValueType::Object))
            },
            ValueType::Array => {
                s.consume_array_start()?;
                s.consume_whitespace()?;
                Ok(Some(ValueType::Array))
            },
        }};
        let value_type = match consume_item(self)? {
            Some(value_type) => value_type,
            None => return Ok(()),
        };
        // Use a stack to avoid recursion.
        let mut stack = Vec::new();
        stack.push(value_type);
        while let Some(value_type) = stack.last().map(|x| *x) {
            match value_type {
                ValueType::Object => {
                    // Continue consuming the object until we hit the end,
                    // or a nested value.
                    loop {
                        if !self.is_object_end()? {
                            self.skip_object_key()?;
                            match consume_item(self)? {
                                Some(nested) => {
                                    stack.push(nested);
                                    break;
                                },
                                None => self.consume_delim_whitespace()?
                            }
                        } else {
                            self.consume_object_end()?;
                            if !stack.is_empty() {
                                // consume a delimiter and whitespace
                                // after this object if there is any
                                self.consume_delim_whitespace()?;
                            }
                        }
                    }
                },
                ValueType::Array => {
                    self.consume_array_start()?;
                    stack.push(ValueType::Array);
                },
                _ => panic!("Unexpected value type: {:?}", value_type),
            }
        }
        Ok(())
    }
    // Will skip to either (1) the idx-th object in the current
    // value if the current value is an array, or
    // (2) the "{idx}" key in the current value if the current value is an object.
    // These are useful for skipping to a particular child of an object or array.
    fn consume_whitespace(&mut self) -> Result<(), JsonError> {
        // consume characters until we peek a non-whitespace character
        loop {
            let done = self.peek_and(|c| match c {
                Some(c) => (c.is_whitespace(), c.is_whitespace()),
                None => (true, true)
            })?;
            if done { break; }
        }
        Ok(())
    }

    // Matches either whitespace or whitespace , whitespace
    fn consume_delim_whitespace(&mut self) -> Result<(), JsonError> {
        self.consume_whitespace()?;
        let is_delim = self.peek_and(|c| {
            let is_delim = c.is_some() && c.unwrap() == ',';
            (is_delim, is_delim)
        })?;
        if is_delim {
            self.consume_whitespace()?;
        }
        Ok(())
    }

    // Object reading methods
    fn consume_object_start(&mut self) -> Result<(), JsonError> {
        match self.next()? {
            Some('{') => Ok(()),
            Some(c) => Err(JsonError::Unexpected(c)),
            None => Err(JsonError::UnexpectedEOF)
        }
    }
    fn consume_object_end(&mut self) -> Result<(), JsonError> {
        match self.next()? {
            Some('}') => Ok(()),
            Some(c) => Err(JsonError::Unexpected(c)),
            None => Err(JsonError::UnexpectedEOF)
        }
    }
    fn is_object_end(&mut self) -> Result<bool, JsonError> {
        match self.peek()? {
            Some('}') => Ok(true),
            Some(_) => Ok(false),
            None => Err(JsonError::UnexpectedEOF)
        }
    }
    // Match "key" WHITESPACE ":" WHITESPACE
    fn skip_object_key(&mut self) -> Result<(), JsonError> {
        match self.next()? {
            Some('"') => (),
            Some(c) => return Err(JsonError::Unexpected(c)),
            None => return Err(JsonError::UnexpectedEOF)
        }
        self.consume_whitespace()?;
        match self.next()? {
            Some(':') => (),
            Some(c) => return Err(JsonError::Unexpected(c)),
            None => return Err(JsonError::UnexpectedEOF)
        }
        self.consume_whitespace()?;
        Ok(())
    }
    fn consume_object_key<'tmp>(&mut self, scratch: &'tmp ScratchBuffer<str>) -> Result<Text<'src, 'tmp>, JsonError> 
                                where Self: 'tmp {
        match self.next()? {
            Some('"') => (),
            Some(c) => return Err(JsonError::Unexpected(c)),
            None => return Err(JsonError::UnexpectedEOF)
        }
        let result = self.consume_string(scratch)?;
        self.consume_whitespace()?;
        match self.next()? {
            Some(':') => (),
            Some(c) => return Err(JsonError::Unexpected(c)),
            None => return Err(JsonError::UnexpectedEOF)
        }
        self.consume_whitespace()?;
        Ok(result)
    }

    // Array reading methods
    fn consume_array_start(&mut self) -> Result<(), JsonError> {
        todo!()
    }
    fn consume_array_end(&mut self) -> Result<(), JsonError> {
        todo!()
    }
    fn is_array_end(&mut self) -> Result<bool, JsonError> {
        todo!()
    }


    // The primitive types!
    fn consume_number(&mut self) -> Result<f64, JsonError> {
        todo!()
    }
    fn skip_number(&mut self) -> Result<(), JsonError> {
        todo!()
    }

    fn consume_boolean(&mut self) -> Result<bool, JsonError> {
        todo!()
    }
    fn skip_boolean(&mut self) -> Result<(), JsonError> {
        todo!()
    }

    fn consume_null(&mut self) -> Result<(), JsonError> {
        todo!()
    }
    fn skip_null(&mut self) -> Result<(), JsonError> {
        self.consume_null()
    }

    // Consume a string of escaped text, returning the unescaped version.
    fn consume_string<'tmp>(&mut self, scratch: &'tmp ScratchBuffer<str>) -> Result<Text<'src, 'tmp>, JsonError>
                        where Self: 'tmp {
        match self.next()? {
            Some(c) => if c != '"' { return Err(JsonError::Unexpected(c)); },
            None => return Err(JsonError::UnexpectedEOF)
        }
        let (buffered, escaped) = {
            let mut buf = self.buffering(scratch);
            let mut escaped = false;
            let mut escaping = false;
            loop {
                let c = match buf.next()? {
                    Some(c) => c,
                    None => return Err(JsonError::UnexpectedEOF)
                };
                if !escaping {
                    if c == '\\' {
                        escaped = true;
                        escaping = true;
                    } else if c == '"' {
                        break;
                    }
                } else {
                    escaping = false;
                }
            }
            (buf.into(), escaped)
        };
        // pop the trailing quote
        let (buffered, _) = buffered.pop();
        // If no escaping, return the buffered text as-is
        if !escaped {
            return Ok(buffered)
        } else {
            // Convert the buffer into a String and consume the rest
            // using special escape handling rules.
            let unescaped = unescape(buffered.into_string())?;
            Ok(Text::Owned(unescaped))
        }
    }
    // Skip a string of escaped text, erroring if there is an issue with the escaping.
    fn skip_string(&mut self) -> Result<(), JsonError> {
        match self.next()? {
            Some(c) => if c != '"' { return Err(JsonError::Unexpected(c)); },
            None => return Err(JsonError::UnexpectedEOF)
        }
        loop {
            let c = match self.next()? {
                Some(c) => c,
                None => return Err(JsonError::UnexpectedEOF)
            };
            if c == '\\' { skip_escaped(self)?; }
            else if c == '"' { break; }
        }
        Ok(())
    }
}
impl<'src, T: TextSource<'src>> JsonSource<'src> for T {}