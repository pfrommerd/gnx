use crate::{json::Number, util::{ScratchBuffer, Text, TextSource, BufferedSource}};

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

// Returns true if the number is fractional, false if it purely integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumberType { PureInt, SciInt, Float }


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    String, Number, Bool, Null, Object, Array,
}

pub trait JsonSource<'src>: TextSource<'src> {
    fn peek_type(&mut self) -> Result<ValueType, JsonError> {
        let next = match self.peek()? {
            Some(c) => c,
            None => return Err(JsonError::UnexpectedEOF)
        };
        match next {
            '"' => Ok(ValueType::String),
            '-' => Ok(ValueType::Number),
            '0'..='9' => Ok(ValueType::Number),
            't' => Ok(ValueType::Bool),
            'f' => Ok(ValueType::Bool),
            'n' => Ok(ValueType::Null),
            '{' => Ok(ValueType::Object),
            '[' => Ok(ValueType::Array),
            _ => return Err(JsonError::Unexpected(next))
        }
    }

    // skip a value including any nested values.
    // This will verify that the value is valid JSON
    // Note that this does not recurse.
    fn skip(&mut self) -> Result<(), JsonError> {
        let consume_item = |s: &mut Self| -> Result<Option<ValueType>, JsonError> { match s.peek_type()?{
            ValueType::String => { s.skip_string()?; Ok(None) },
            ValueType::Number => { s.skip_number()?; Ok(None) },
            ValueType::Bool=> { s.skip_bool()?; Ok(None) },
            ValueType::Null => { s.skip_null()?; Ok(None) },
            ValueType::Object => {
                s.consume_object_start()?;
                Ok(Some(ValueType::Object))
            },
            ValueType::Array => {
                s.consume_array_start()?;
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
                    loop {
                        if !self.is_array_end()? {
                            match consume_item(self)? {
                                Some(nested) => {
                                    stack.push(nested);
                                    break;
                                },
                                None => self.consume_delim_whitespace()?
                            }
                        } else {
                            self.consume_array_end()?;
                            if !stack.is_empty() {
                                self.consume_delim_whitespace()?;
                            }
                        }
                    }
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
                Some(c) => (c.is_whitespace(), !c.is_whitespace()),
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
            Some('{') => {
                self.consume_whitespace()?;
                Ok(())
            },
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
    fn consume_colon_sep(&mut self) -> Result<(), JsonError> {
        self.consume_whitespace()?;
        match self.next()? {
            Some(':') => (),
            Some(c) => return Err(JsonError::Unexpected(c)),
            None => return Err(JsonError::UnexpectedEOF)
        }
        self.consume_whitespace()?;
        Ok(())
    }

    // Array reading methods
    fn consume_array_start(&mut self) -> Result<(), JsonError> {
        match self.next()? {
            Some('[') => {
                self.consume_whitespace()?;
                Ok(())
            },
            Some(c) => Err(JsonError::Unexpected(c)),
            None => Err(JsonError::UnexpectedEOF)
        }
    }
    fn consume_array_end(&mut self) -> Result<(), JsonError> {
        match self.next()? {
            Some(']') => Ok(()),
            Some(c) => Err(JsonError::Unexpected(c)),
            None => Err(JsonError::UnexpectedEOF)
        }
    }
    fn is_array_end(&mut self) -> Result<bool, JsonError> {
        match self.peek()? {
            Some(']') => Ok(true),
            Some(_) => Ok(false),
            None => Err(JsonError::UnexpectedEOF)
        }
    }

    // consume number potentially requires a scratch buffer to handle overflowing numbers.
    fn consume_number(&mut self, scratch: &mut ScratchBuffer<str>) -> Result<Number, JsonError> {
        // TODO: Consume without buffering for better performance?
        let mut buffered = self.buffering(scratch);
        let number_type = buffered.skip_number()?;
        let text = buffered.into_buffer();
        let text = text.borrow();
        let opt = || -> Result<Number, JsonError> { match number_type {
            NumberType::PureInt => {
                if text.starts_with('-') {
                    text.parse::<i64>().map(Number::from)
                        .map_err(JsonError::from)
                } else {
                    text.parse::<u64>().map(Number::from)
                        .map_err(JsonError::from)
                }
            },
            NumberType::SciInt => {
                let exp_offset = text.find('e').or_else(|| text.find('E')).unwrap();
                // parse the exponent part
                let exp: u32 = match text[exp_offset + 1..].parse() {
                    Ok(exp) => exp,
                    Err(_) => return Err(JsonError::InvalidNumber)
                };
                // parse the pre-exponent part
                let mantisa = &text[..exp_offset];
                if mantisa.starts_with('-') {
                    let mantisa = mantisa.parse::<i64>()?;
                    Ok(Number::from(mantisa * 10_i64.pow(exp)))
                } else {
                    let mantisa = mantisa.parse::<u64>()?;
                    Ok(Number::from(mantisa * 10_u64.pow(exp)))
                }
            },
            _ => Err(JsonError::InvalidNumber),
        }}();
        match opt {
            Ok(number) => Ok(number),
            // We cannot parse as a number, so fallback 
            // to a floating point number
            Err(_) => { 
                text.parse::<f64>().map(Number::from)
                    .map_err(JsonError::from)
            }
        }
    }

    fn skip_number(&mut self) -> Result<NumberType, JsonError> {
        // skip a minus sign if it exists
        self.peek_and(|c| (c == Some('-'), ()))?;
        // skip the integer part
        let first_digit = self.peek_and(|c| match c {
            Some(c @ '0'..='9') => (true, Ok(c as u8 - b'0')),
            Some(c) => (false, Err(JsonError::Unexpected(c))),
            None => (false, Err(JsonError::UnexpectedEOF))
        })??;
        // Ensure that the first digit is 0, 
        // we cannot have a number like 01 or 001.
        if first_digit == 0 && matches!(self.peek()?, Some('0'..='9')) {
            return Err(JsonError::InvalidNumber);
        }
        // skip until we hit a non-digit character
        loop {
            let is_digit = self.peek_and(|c| match c {
                Some('0'..='9') => (true, true),
                _ => (false, false),
            })?;
            if !is_digit { break; }
        }
        // check for a fractional part
        let mut has_fractional = self.peek_and(|c| match c {
            Some('.') => (true, true),
            _ => (false, false),
        })?;
        if has_fractional {
            // skip the first digit of the fractional part
            self.peek_and(|c| match c {
                Some('0'..='9') => (true, Ok(())),
                Some(c) => (false, Err(JsonError::Unexpected(c))),
                None => (false, Err(JsonError::UnexpectedEOF))
            })??;
            loop {
                let is_digit = self.peek_and(|c| match c {
                    Some('0'..='9') => (true, true),
                    _ => (false, false),
                })?;
                if !is_digit { break; }
            }
        }
        // check for an exponent part
        let has_exponent = self.peek_and(|c| match c {
            Some('e') => (true, true),
            Some('E') => (true, true),
            _ => (false, false),
        })?;
        if has_exponent {
            // skip a + or - sign if it exists
            let nonpos_exp = self.peek_and(|c| match c {
                Some('+') => (true, false),
                Some('-') => (true, true),
                _ => (false, false),
            })?;

            // skip the first digit of the exponent
            let nonzero_exp = self.peek_and(|c| match c {
                Some('0') => (true, Ok(false)),
                Some('1'..='9') => (true, Ok(true)),
                Some(c) => (false, Err(JsonError::Unexpected(c))),
                None => (false, Err(JsonError::UnexpectedEOF))
            })??;
            // If we have a negative exponent part,
            // we do not have a "scientific" integer.
            if nonzero_exp && nonpos_exp {
                has_fractional = true;
            }
            loop {
                let is_digit = self.peek_and(|c| match c {
                    Some('0'..='9') => (true, true),
                    _ => (false, false),
                })?;
                if !is_digit { break; }
            }
        }
        Ok(match (has_fractional, has_exponent) {
            (true, _) => NumberType::Float,
            (false, true) => NumberType::SciInt,
            (false, false) => NumberType::PureInt,
        })
    }

    fn consume_bool(&mut self) -> Result<bool, JsonError> {
        // check if the next 5 characters start with "true" 
        // or are exactly "false"
        let b = self.peek_chars_and(5, |s| {
            let maybe_true = &s[..4.max(s.len())] == "true";
            let maybe_false = s == "false";
            if maybe_true { (4, Ok(true)) }
            else if maybe_false { (5, Ok(false)) }
            else { (0, Err(JsonError::Unexpected(s.chars().next().unwrap()))) }
        })??;
        Ok(b)
    }

    fn skip_bool(&mut self) -> Result<(), JsonError> {
        self.consume_bool()?;
        Ok(())
    }

    fn consume_null(&mut self) -> Result<(), JsonError> {
        // check if the next 4 characters are exactly "null"
        self.peek_chars_and(4, |s| {
            if s == "null" { (4, Ok(())) }
            else { (0, Err(JsonError::Unexpected(s.chars().next().unwrap()))) }
        })??;
        Ok(())
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