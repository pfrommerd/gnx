use std::collections::{BTreeMap, BTreeSet};

use gnx_graph::{
    DataVisitor, DeserializeSeed, Deserializer, EnumAccess, MapAccess, SeqAccess, VariantAccess,
};

use crate::util::TextSource;

use super::{Map, Result, TomlError, TomlSource, Value};

pub struct TomlParser<S> {
    source: S,
    parsed: Option<Value>,
}

#[derive(Default)]
struct DocumentState {
    current_table: Vec<String>,
    explicit_tables: BTreeSet<Vec<String>>,
    dotted_tables: BTreeSet<Vec<String>>,
    arrays_of_tables: BTreeSet<Vec<String>>,
    array_table_indices: BTreeMap<Vec<String>, usize>,
    defined_values: BTreeSet<Vec<String>>,
    inline_tables: BTreeSet<Vec<String>>,
}

impl<S> TomlParser<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            parsed: None,
        }
    }

    pub fn into_inner(self) -> S {
        self.source
    }
}

impl<'src, S: TextSource<'src>> TomlParser<S> {
    pub fn into_value(mut self) -> Result<Value> {
        self.parse_if_needed()?;
        Ok(self.parsed.take().expect("TOML document was parsed"))
    }

    pub fn parse_document(&mut self) -> Result<&Value> {
        self.parse_if_needed()?;
        Ok(self.parsed.as_ref().expect("TOML document was parsed"))
    }

    fn parse_if_needed(&mut self) -> Result<()> {
        if self.parsed.is_none() {
            self.parsed = Some(self.parse_document_inner()?);
        }
        Ok(())
    }

    fn parse_document_inner(&mut self) -> Result<Value> {
        let mut root = Value::Table(Map::new());
        let mut state = DocumentState::default();

        self.source.consume_ws_comments_newlines()?;
        while self.source.peek()?.is_some() {
            self.source.consume_ws()?;
            match self.source.peek()? {
                Some('[') => self.parse_header(&mut root, &mut state)?,
                Some(_) => self.parse_key_value(&mut root, &mut state)?,
                None => break,
            }
            self.source.consume_line_end_or_eof()?;
            self.source.consume_ws_comments_newlines()?;
        }

        Ok(root)
    }

    fn parse_header(&mut self, root: &mut Value, state: &mut DocumentState) -> Result<()> {
        self.expect_char('[')?;
        let array = if self.source.peek()? == Some('[') {
            self.source.next()?;
            true
        } else {
            false
        };

        self.source.consume_ws()?;
        let path = self.parse_key()?;
        self.source.consume_ws()?;

        self.expect_char(']')?;
        if array {
            self.expect_char(']')?;
            self.define_array_table(root, state, &path)?;
        } else {
            self.define_table(root, state, &path)?;
        }
        state.current_table = path;
        Ok(())
    }

    fn parse_key_value(&mut self, root: &mut Value, state: &mut DocumentState) -> Result<()> {
        let mut path = state.current_table.clone();
        let key = self.parse_key()?;
        self.source.consume_ws()?;
        self.expect_char('=')?;
        self.source.consume_ws()?;
        let value = self.parse_value()?;
        path.extend(key.iter().cloned());
        self.insert_value(root, state, &path, value, !key.is_empty())
    }

    fn parse_key(&mut self) -> Result<Vec<String>> {
        let mut parts = Vec::new();
        loop {
            self.source.consume_ws()?;
            parts.push(self.parse_key_part()?);
            self.source.consume_ws()?;
            if self.source.peek()? != Some('.') {
                break;
            }
            self.source.next()?;
        }
        Ok(parts)
    }

    fn parse_key_part(&mut self) -> Result<String> {
        match self.source.peek()? {
            Some('"') => {
                if self.source.peek_chars(3)? == "\"\"\"" {
                    return Err(TomlError::InvalidKey);
                }
                self.parse_basic_string(false)
            }
            Some('\'') => {
                if self.source.peek_chars(3)? == "'''" {
                    return Err(TomlError::InvalidKey);
                }
                self.parse_literal_string(false)
            }
            Some(ch) if is_bare_key_char(ch) => self.parse_bare_key(),
            Some(_) => Err(TomlError::InvalidKey),
            None => Err(TomlError::UnexpectedEOF),
        }
    }

    fn parse_bare_key(&mut self) -> Result<String> {
        let mut key = String::new();
        while let Some(ch) = self.source.peek()? {
            if !is_bare_key_char(ch) {
                break;
            }
            key.push(ch);
            self.source.next()?;
        }
        if key.is_empty() {
            Err(TomlError::InvalidKey)
        } else {
            Ok(key)
        }
    }

    fn parse_value(&mut self) -> Result<Value> {
        self.source.consume_ws()?;
        match self.source.peek()? {
            Some('"') => {
                let multiline = self.source.peek_chars(3)? == "\"\"\"";
                self.parse_basic_string(multiline).map(Value::String)
            }
            Some('\'') => {
                let multiline = self.source.peek_chars(3)? == "'''";
                self.parse_literal_string(multiline).map(Value::String)
            }
            Some('[') => self.parse_array(),
            Some('{') => self.parse_inline_table(),
            Some(_) => {
                let token = self.parse_bare_value_token()?;
                parse_bare_value(&token)
            }
            None => Err(TomlError::UnexpectedEOF),
        }
    }

    fn parse_array(&mut self) -> Result<Value> {
        self.expect_char('[')?;
        let mut values = Vec::new();
        loop {
            self.source.consume_ws_comments_newlines()?;
            if self.source.peek()? == Some(']') {
                self.source.next()?;
                break;
            }
            values.push(self.parse_value()?);
            self.source.consume_ws_comments_newlines()?;
            match self.source.peek()? {
                Some(',') => {
                    self.source.next()?;
                }
                Some(']') => {
                    self.source.next()?;
                    break;
                }
                Some(ch) => return Err(TomlError::Unexpected(ch)),
                None => return Err(TomlError::UnexpectedEOF),
            }
        }
        Ok(Value::Array(values))
    }

    fn parse_inline_table(&mut self) -> Result<Value> {
        self.expect_char('{')?;
        let mut table = Map::new();
        loop {
            self.source.consume_ws_comments_newlines()?;
            if self.source.peek()? == Some('}') {
                self.source.next()?;
                break;
            }

            let key = self.parse_key()?;
            self.source.consume_ws()?;
            self.expect_char('=')?;
            self.source.consume_ws()?;
            let value = self.parse_value()?;
            insert_inline_value(&mut table, &key, value)?;

            self.source.consume_ws_comments_newlines()?;
            match self.source.peek()? {
                Some(',') => {
                    self.source.next()?;
                }
                Some('}') => {
                    self.source.next()?;
                    break;
                }
                Some(ch) => return Err(TomlError::Unexpected(ch)),
                None => return Err(TomlError::UnexpectedEOF),
            }
        }
        Ok(Value::Table(table))
    }

    fn parse_bare_value_token(&mut self) -> Result<String> {
        let mut token = String::new();
        loop {
            match self.source.peek()? {
                Some('#' | '\n' | '\r' | ',' | ']' | '}') | None => break,
                Some(ch) => {
                    token.push(ch);
                    self.source.next()?;
                }
            }
        }
        let token = token.trim_end_matches([' ', '\t']).to_string();
        if token.is_empty() {
            Err(TomlError::Expected("a TOML value"))
        } else {
            Ok(token)
        }
    }

    fn parse_basic_string(&mut self, multiline: bool) -> Result<String> {
        if multiline {
            self.expect_str("\"\"\"")?;
            if self.source.consume_newline()? {
                // The first newline after a multi-line delimiter is trimmed.
            }
            self.parse_multiline_basic_body()
        } else {
            self.expect_char('"')?;
            self.parse_basic_body()
        }
    }

    fn parse_basic_body(&mut self) -> Result<String> {
        let mut out = String::new();
        loop {
            match self.source.next()? {
                Some('"') => return Ok(out),
                Some('\\') => out.push(self.parse_escape()?),
                Some('\n' | '\r') => return Err(TomlError::UnterminatedString),
                Some(ch) if invalid_basic_char(ch, false) => return Err(TomlError::Unexpected(ch)),
                Some(ch) => out.push(ch),
                None => return Err(TomlError::UnterminatedString),
            }
        }
    }

    fn parse_multiline_basic_body(&mut self) -> Result<String> {
        let mut out = String::new();
        loop {
            let quotes = self.source.peek_chars(5)?;
            if quotes.starts_with("\"\"\"\"\"") {
                self.expect_str("\"\"")?;
                out.push_str("\"\"");
                self.expect_str("\"\"\"")?;
                return Ok(out);
            }
            if quotes.starts_with("\"\"\"\"") {
                self.expect_char('"')?;
                out.push('"');
                self.expect_str("\"\"\"")?;
                return Ok(out);
            }
            if quotes.starts_with("\"\"\"") {
                self.expect_str("\"\"\"")?;
                return Ok(out);
            }
            if self.source.consume_newline()? {
                out.push('\n');
                continue;
            }
            match self.source.next()? {
                Some('\\') => {
                    if self.consume_line_ending_backslash()? {
                        continue;
                    }
                    out.push(self.parse_escape()?);
                }
                Some(ch) if invalid_basic_char(ch, true) => return Err(TomlError::Unexpected(ch)),
                Some(ch) => out.push(ch),
                None => return Err(TomlError::UnterminatedString),
            }
        }
    }

    fn consume_line_ending_backslash(&mut self) -> Result<bool> {
        let checkpoint = self.source.position()?;
        self.source.consume_ws()?;
        if self.source.consume_newline()? {
            loop {
                self.source.consume_ws()?;
                if !self.source.consume_newline()? {
                    break;
                }
            }
            Ok(true)
        } else {
            self.source.goto(checkpoint)?;
            Ok(false)
        }
    }

    fn parse_literal_string(&mut self, multiline: bool) -> Result<String> {
        if multiline {
            self.expect_str("'''")?;
            if self.source.consume_newline()? {
                // The first newline after a multi-line delimiter is trimmed.
            }
            self.parse_multiline_literal_body()
        } else {
            self.expect_char('\'')?;
            self.parse_literal_body()
        }
    }

    fn parse_literal_body(&mut self) -> Result<String> {
        let mut out = String::new();
        loop {
            match self.source.next()? {
                Some('\'') => return Ok(out),
                Some('\n' | '\r') => return Err(TomlError::UnterminatedString),
                Some(ch) if invalid_literal_char(ch, false) => {
                    return Err(TomlError::Unexpected(ch));
                }
                Some(ch) => out.push(ch),
                None => return Err(TomlError::UnterminatedString),
            }
        }
    }

    fn parse_multiline_literal_body(&mut self) -> Result<String> {
        let mut out = String::new();
        loop {
            let quotes = self.source.peek_chars(5)?;
            if quotes.starts_with("'''''") {
                self.expect_str("''")?;
                out.push_str("''");
                self.expect_str("'''")?;
                return Ok(out);
            }
            if quotes.starts_with("''''") {
                self.expect_char('\'')?;
                out.push('\'');
                self.expect_str("'''")?;
                return Ok(out);
            }
            if quotes.starts_with("'''") {
                self.expect_str("'''")?;
                return Ok(out);
            }
            if self.source.consume_newline()? {
                out.push('\n');
                continue;
            }
            match self.source.next()? {
                Some(ch) if invalid_literal_char(ch, true) => {
                    return Err(TomlError::Unexpected(ch));
                }
                Some(ch) => out.push(ch),
                None => return Err(TomlError::UnterminatedString),
            }
        }
    }

    fn parse_escape(&mut self) -> Result<char> {
        match self.source.next()? {
            Some('b') => Ok('\u{08}'),
            Some('t') => Ok('\t'),
            Some('n') => Ok('\n'),
            Some('f') => Ok('\u{0c}'),
            Some('r') => Ok('\r'),
            Some('e') => Ok('\u{1b}'),
            Some('"') => Ok('"'),
            Some('\\') => Ok('\\'),
            Some('x') => self.parse_unicode_escape(2),
            Some('u') => self.parse_unicode_escape(4),
            Some('U') => self.parse_unicode_escape(8),
            Some(_) => Err(TomlError::InvalidEscape),
            None => Err(TomlError::UnexpectedEOF),
        }
    }

    fn parse_unicode_escape(&mut self, digits: usize) -> Result<char> {
        let mut value = 0u32;
        for _ in 0..digits {
            let ch = self.source.next()?.ok_or(TomlError::UnexpectedEOF)?;
            let digit = ch.to_digit(16).ok_or(TomlError::InvalidEscape)?;
            value = (value << 4) | digit;
        }
        char::from_u32(value).ok_or(TomlError::InvalidEscape)
    }

    fn define_table(
        &mut self,
        root: &mut Value,
        state: &mut DocumentState,
        path: &[String],
    ) -> Result<()> {
        let scoped = scoped_path(state, path);
        if path.is_empty()
            || state.explicit_tables.contains(&scoped)
            || state.arrays_of_tables.contains(path)
            || state.dotted_tables.contains(&scoped)
            || state.defined_values.contains(&scoped)
            || has_inline_prefix(&state.inline_tables, &scoped)
        {
            return Err(TomlError::DuplicateKey(path_to_string(path)));
        }
        ensure_table_path(root, path, &state.arrays_of_tables)?;
        state.explicit_tables.insert(scoped);
        Ok(())
    }

    fn define_array_table(
        &mut self,
        root: &mut Value,
        state: &mut DocumentState,
        path: &[String],
    ) -> Result<()> {
        let scoped = scoped_path(state, path);
        if path.is_empty()
            || state.explicit_tables.contains(&scoped)
            || state.defined_values.contains(&scoped)
            || has_inline_prefix(&state.inline_tables, &scoped)
        {
            return Err(TomlError::DuplicateKey(path_to_string(path)));
        }
        append_array_table(root, path, &state.arrays_of_tables)?;
        state.arrays_of_tables.insert(path.to_vec());
        let next_index = state
            .array_table_indices
            .get(path)
            .map(|index| index + 1)
            .unwrap_or(0);
        state.array_table_indices.insert(path.to_vec(), next_index);
        Ok(())
    }

    fn insert_value(
        &mut self,
        root: &mut Value,
        state: &mut DocumentState,
        path: &[String],
        value: Value,
        dotted_key: bool,
    ) -> Result<()> {
        let scoped = scoped_path(state, path);
        if state.defined_values.contains(&scoped)
            || state.explicit_tables.contains(&scoped)
            || state.arrays_of_tables.contains(path)
            || has_inline_prefix(&state.inline_tables, &scoped)
        {
            return Err(TomlError::DuplicateKey(path_to_string(path)));
        }
        insert_document_value(root, path, value.clone(), &state.arrays_of_tables)?;
        state.defined_values.insert(scoped);
        if dotted_key && path.len() > state.current_table.len() + 1 {
            for idx in state.current_table.len() + 1..path.len() {
                state.dotted_tables.insert(scoped_path(state, &path[..idx]));
            }
        }
        if matches!(value, Value::Table(_)) {
            state.inline_tables.insert(scoped_path(state, path));
        }
        Ok(())
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        match self.source.next()? {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(TomlError::Unexpected(ch)),
            None => Err(TomlError::UnexpectedEOF),
        }
    }

    fn expect_str(&mut self, expected: &str) -> Result<()> {
        for ch in expected.chars() {
            self.expect_char(ch)?;
        }
        Ok(())
    }
}

fn is_bare_key_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
}

fn invalid_basic_char(ch: char, multiline: bool) -> bool {
    match ch {
        '\t' => false,
        '\n' | '\r' if multiline => false,
        '\u{00}'..='\u{08}' | '\u{0a}'..='\u{1f}' | '\u{7f}' => true,
        _ => false,
    }
}

fn invalid_literal_char(ch: char, multiline: bool) -> bool {
    match ch {
        '\t' => false,
        '\n' | '\r' if multiline => false,
        '\u{00}'..='\u{08}' | '\u{0a}'..='\u{1f}' | '\u{7f}' => true,
        _ => false,
    }
}

fn path_to_string(path: &[String]) -> String {
    path.join(".")
}

fn has_inline_prefix(inline_tables: &BTreeSet<Vec<String>>, path: &[String]) -> bool {
    (1..=path.len()).any(|idx| inline_tables.contains(&path[..idx]))
}

fn scoped_path(state: &DocumentState, path: &[String]) -> Vec<String> {
    let mut scoped = Vec::with_capacity(path.len());
    let mut prefix = Vec::new();
    for part in path {
        prefix.push(part.clone());
        scoped.push(part.clone());
        if state.arrays_of_tables.contains(&prefix) {
            if let Some(index) = state.array_table_indices.get(&prefix) {
                scoped.push(format!("#{index}"));
            }
        }
    }
    scoped
}

fn insert_inline_value(
    table: &mut Map<String, Value>,
    path: &[String],
    value: Value,
) -> Result<()> {
    if path.is_empty() {
        return Err(TomlError::InvalidKey);
    }
    let mut cursor = table;
    for part in &path[..path.len() - 1] {
        let entry = cursor
            .entry(part.clone())
            .or_insert_with(|| Value::Table(Map::new()));
        cursor = entry
            .as_table_mut()
            .ok_or_else(|| TomlError::DuplicateKey(path_to_string(path)))?;
    }
    let key = path.last().expect("path is not empty").clone();
    if cursor.contains_key(&key) {
        return Err(TomlError::DuplicateKey(path_to_string(path)));
    }
    cursor.insert(key, value);
    Ok(())
}

fn current_table_mut<'a>(
    root: &'a mut Value,
    path: &[String],
    array_tables: &BTreeSet<Vec<String>>,
) -> Result<&'a mut Map<String, Value>> {
    let mut cursor = root.as_table_mut().ok_or(TomlError::RootMustBeTable)?;
    let mut prefix = Vec::new();
    for part in path {
        prefix.push(part.clone());
        let value = cursor
            .get_mut(part)
            .ok_or_else(|| TomlError::Expected("an existing TOML table"))?;
        if array_tables.contains(&prefix) {
            match value {
                Value::Array(items) => {
                    let last = items
                        .last_mut()
                        .ok_or_else(|| TomlError::Expected("an array table element"))?;
                    cursor = last
                        .as_table_mut()
                        .ok_or_else(|| TomlError::Expected("a TOML table"))?;
                }
                _ => return Err(TomlError::DuplicateKey(path_to_string(&prefix))),
            }
        } else {
            cursor = value
                .as_table_mut()
                .ok_or_else(|| TomlError::DuplicateKey(path_to_string(&prefix)))?;
        }
    }
    Ok(cursor)
}

fn ensure_table_path(
    root: &mut Value,
    path: &[String],
    array_tables: &BTreeSet<Vec<String>>,
) -> Result<()> {
    let mut cursor = root.as_table_mut().ok_or(TomlError::RootMustBeTable)?;
    let mut prefix = Vec::new();
    for part in path {
        prefix.push(part.clone());
        if array_tables.contains(&prefix) {
            let value = cursor
                .get_mut(part)
                .ok_or_else(|| TomlError::Expected("an array table"))?;
            match value {
                Value::Array(items) => {
                    let last = items
                        .last_mut()
                        .ok_or_else(|| TomlError::Expected("an array table element"))?;
                    cursor = last
                        .as_table_mut()
                        .ok_or_else(|| TomlError::Expected("a TOML table"))?;
                }
                _ => return Err(TomlError::DuplicateKey(path_to_string(&prefix))),
            }
        } else {
            let entry = cursor
                .entry(part.clone())
                .or_insert_with(|| Value::Table(Map::new()));
            cursor = entry
                .as_table_mut()
                .ok_or_else(|| TomlError::DuplicateKey(path_to_string(&prefix)))?;
        }
    }
    Ok(())
}

fn insert_document_value(
    root: &mut Value,
    path: &[String],
    value: Value,
    array_tables: &BTreeSet<Vec<String>>,
) -> Result<()> {
    if path.is_empty() {
        return Err(TomlError::InvalidKey);
    }
    ensure_table_path(root, &path[..path.len() - 1], array_tables)?;
    let table = current_table_mut(root, &path[..path.len() - 1], array_tables)?;
    let key = path.last().expect("path is not empty").clone();
    if table.contains_key(&key) {
        return Err(TomlError::DuplicateKey(path_to_string(path)));
    }
    table.insert(key, value);
    Ok(())
}

fn append_array_table(
    root: &mut Value,
    path: &[String],
    array_tables: &BTreeSet<Vec<String>>,
) -> Result<()> {
    if path.is_empty() {
        return Err(TomlError::InvalidKey);
    }
    ensure_table_path(root, &path[..path.len() - 1], array_tables)?;
    let table = current_table_mut(root, &path[..path.len() - 1], array_tables)?;
    let key = path.last().expect("path is not empty").clone();
    match table.get_mut(&key) {
        Some(Value::Array(items)) if array_tables.contains(path) => {
            items.push(Value::Table(Map::new()));
        }
        Some(_) => return Err(TomlError::DuplicateKey(path_to_string(path))),
        None => {
            table.insert(key, Value::Array(vec![Value::Table(Map::new())]));
        }
    }
    Ok(())
}

fn parse_bare_value(token: &str) -> Result<Value> {
    let token = token.trim();
    match token {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        _ => {}
    }
    if let Some(value) = parse_datetime(token)? {
        return Ok(value);
    }
    parse_number(token)
}

fn parse_datetime(token: &str) -> Result<Option<Value>> {
    if parse_offset_datetime(token)? {
        return Ok(Some(Value::OffsetDateTime(token.to_string())));
    }
    if parse_local_datetime(token)? {
        return Ok(Some(Value::LocalDateTime(token.to_string())));
    }
    if parse_local_date(token)? {
        return Ok(Some(Value::LocalDate(token.to_string())));
    }
    if parse_local_time(token)? {
        return Ok(Some(Value::LocalTime(token.to_string())));
    }
    Ok(None)
}

fn parse_offset_datetime(token: &str) -> Result<bool> {
    let Some((date, rest)) = split_datetime(token) else {
        return Ok(false);
    };
    if !valid_date(date)? {
        return Err(TomlError::InvalidDateTime);
    }
    let (time, offset) = split_offset(rest);
    if offset.is_empty() {
        return Ok(false);
    }
    Ok(valid_time(time)? && valid_offset(offset)?)
}

fn parse_local_datetime(token: &str) -> Result<bool> {
    let Some((date, time)) = split_datetime(token) else {
        return Ok(false);
    };
    if !valid_date(date)? {
        return Err(TomlError::InvalidDateTime);
    }
    Ok(valid_time(time)?)
}

fn parse_local_date(token: &str) -> Result<bool> {
    if token.len() == 10 && token.as_bytes()[4] == b'-' && token.as_bytes()[7] == b'-' {
        return valid_date(token);
    }
    Ok(false)
}

fn parse_local_time(token: &str) -> Result<bool> {
    if token.len() >= 5 && token.as_bytes().get(2) == Some(&b':') {
        return valid_time(token);
    }
    Ok(false)
}

fn split_datetime(token: &str) -> Option<(&str, &str)> {
    if token.len() < 16 {
        return None;
    }
    if let Some(idx) = token.find('T') {
        return Some((&token[..idx], &token[idx + 1..]));
    }
    if let Some(idx) = token.find(' ') {
        return Some((&token[..idx], &token[idx + 1..]));
    }
    None
}

fn split_offset(time: &str) -> (&str, &str) {
    if let Some(time) = time.strip_suffix('Z') {
        return (time, "Z");
    }
    for idx in (0..time.len()).rev() {
        let byte = time.as_bytes()[idx];
        if (byte == b'+' || byte == b'-') && idx >= 5 {
            return (&time[..idx], &time[idx..]);
        }
    }
    (time, "")
}

fn valid_date(date: &str) -> Result<bool> {
    if date.len() != 10 || date.as_bytes()[4] != b'-' || date.as_bytes()[7] != b'-' {
        return Ok(false);
    }
    let year = parse_fixed_digits(&date[0..4])?;
    let month = parse_fixed_digits(&date[5..7])?;
    let day = parse_fixed_digits(&date[8..10])?;
    if !(1..=12).contains(&month) {
        return Ok(false);
    }
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => unreachable!(),
    };
    Ok((1..=max_day).contains(&day))
}

fn valid_time(time: &str) -> Result<bool> {
    if time.len() < 5 || time.as_bytes().get(2) != Some(&b':') {
        return Ok(false);
    }
    let hour = parse_fixed_digits(&time[0..2])?;
    let minute = parse_fixed_digits(&time[3..5])?;
    if hour > 23 || minute > 59 {
        return Ok(false);
    }
    let rest = &time[5..];
    if rest.is_empty() {
        return Ok(true);
    }
    if !rest.starts_with(':') || rest.len() < 3 {
        return Ok(false);
    }
    let second = parse_fixed_digits(&rest[1..3])?;
    if second > 59 {
        return Ok(false);
    }
    let fraction = &rest[3..];
    if fraction.is_empty() {
        return Ok(true);
    }
    if !fraction.starts_with('.') || fraction.len() == 1 {
        return Ok(false);
    }
    Ok(fraction[1..].chars().all(|ch| ch.is_ascii_digit()))
}

fn valid_offset(offset: &str) -> Result<bool> {
    if offset == "Z" {
        return Ok(true);
    }
    if offset.len() != 6
        || !matches!(offset.as_bytes()[0], b'+' | b'-')
        || offset.as_bytes()[3] != b':'
    {
        return Ok(false);
    }
    let hour = parse_fixed_digits(&offset[1..3])?;
    let minute = parse_fixed_digits(&offset[4..6])?;
    Ok(hour <= 23 && minute <= 59)
}

fn parse_fixed_digits(text: &str) -> Result<u32> {
    if text.chars().all(|ch| ch.is_ascii_digit()) {
        Ok(text.parse()?)
    } else {
        Err(TomlError::InvalidDateTime)
    }
}

fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn parse_number(token: &str) -> Result<Value> {
    match token {
        "inf" | "+inf" => return Ok(Value::Float(ordered_float::OrderedFloat(f64::INFINITY))),
        "-inf" => return Ok(Value::Float(ordered_float::OrderedFloat(f64::NEG_INFINITY))),
        "nan" | "+nan" => return Ok(Value::Float(ordered_float::OrderedFloat(f64::NAN))),
        "-nan" => return Ok(Value::Float(ordered_float::OrderedFloat(-f64::NAN))),
        _ => {}
    }

    if let Some(value) = parse_base_integer(token)? {
        return Ok(Value::Integer(value));
    }

    if token.contains('.') || token.contains('e') || token.contains('E') {
        validate_float(token)?;
        return Ok(Value::Float(ordered_float::OrderedFloat(
            token.replace('_', "").parse()?,
        )));
    }

    validate_decimal_integer(token)?;
    Ok(Value::Integer(token.replace('_', "").parse()?))
}

fn parse_base_integer(token: &str) -> Result<Option<i64>> {
    let (base, digits) = if let Some(digits) = token.strip_prefix("0x") {
        (16, digits)
    } else if let Some(digits) = token.strip_prefix("0o") {
        (8, digits)
    } else if let Some(digits) = token.strip_prefix("0b") {
        (2, digits)
    } else {
        return Ok(None);
    };
    validate_digit_underscores(digits, |ch| ch.is_digit(base))?;
    let compact = digits.replace('_', "");
    let value = i64::from_str_radix(&compact, base)?;
    Ok(Some(value))
}

fn validate_decimal_integer(token: &str) -> Result<()> {
    let digits = token.strip_prefix(['+', '-']).unwrap_or(token);
    validate_digit_underscores(digits, |ch| ch.is_ascii_digit())?;
    let compact = digits.replace('_', "");
    if compact.len() > 1 && compact.starts_with('0') {
        return Err(TomlError::InvalidNumber);
    }
    Ok(())
}

fn validate_float(token: &str) -> Result<()> {
    let (mantissa, exponent) = match token.find(['e', 'E']) {
        Some(idx) => (&token[..idx], Some(&token[idx + 1..])),
        None => (token, None),
    };
    if let Some(exponent) = exponent {
        let exponent_digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        validate_digit_underscores(exponent_digits, |ch| ch.is_ascii_digit())?;
    }
    if let Some(idx) = mantissa.find('.') {
        validate_decimal_integer(&mantissa[..idx])?;
        validate_digit_underscores(&mantissa[idx + 1..], |ch| ch.is_ascii_digit())?;
    } else {
        validate_decimal_integer(mantissa)?;
    }
    Ok(())
}

fn validate_digit_underscores<F>(digits: &str, valid_digit: F) -> Result<()>
where
    F: Fn(char) -> bool,
{
    if digits.is_empty() {
        return Err(TomlError::InvalidNumber);
    }
    let mut prev_digit = false;
    for ch in digits.chars() {
        if valid_digit(ch) {
            prev_digit = true;
        } else if ch == '_' {
            if !prev_digit {
                return Err(TomlError::InvalidNumber);
            }
            prev_digit = false;
        } else {
            return Err(TomlError::InvalidNumber);
        }
    }
    if !prev_digit {
        return Err(TomlError::InvalidNumber);
    }
    Ok(())
}

pub struct TomlValueDeserializer<'a> {
    value: &'a Value,
}

impl<'a> TomlValueDeserializer<'a> {
    pub fn new(value: &'a Value) -> Self {
        Self { value }
    }

    fn deserialize_integer<'de, V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::Integer(value) => visitor.visit_i64(*value),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_unsigned<'de, V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::Integer(value) if *value >= 0 => visitor.visit_u64(*value as u64),
            Value::Integer(_) => Err(TomlError::InvalidNumber),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_float<'de, V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::Float(value) => visitor.visit_f64(value.0),
            Value::Integer(value) => visitor.visit_f64(*value as f64),
            _ => self.deserialize_any(visitor),
        }
    }
}

impl<'src, S: TextSource<'src>> TomlParser<S> {
    fn deserialize_integer<V: DataVisitor<'src>>(&mut self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_integer(visitor)
    }

    fn deserialize_unsigned<V: DataVisitor<'src>>(&mut self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_unsigned(visitor)
    }

    fn deserialize_float<V: DataVisitor<'src>>(&mut self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_float(visitor)
    }
}

macro_rules! deserialize_signed {
    ($method:ident) => {
        fn $method<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
            self.deserialize_integer(visitor)
        }
    };
}

macro_rules! deserialize_unsigned {
    ($method:ident) => {
        fn $method<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
            self.deserialize_unsigned(visitor)
        }
    };
}

macro_rules! deserialize_float {
    ($method:ident) => {
        fn $method<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
            self.deserialize_float(visitor)
        }
    };
}

impl<'src> Deserializer<'src> for TomlValueDeserializer<'_> {
    type Error = TomlError;

    fn deserialize_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::Bool(value) => visitor.visit_bool(*value),
            Value::Integer(value) => visitor.visit_i64(*value),
            Value::Float(value) => visitor.visit_f64(value.0),
            Value::String(value)
            | Value::OffsetDateTime(value)
            | Value::LocalDateTime(value)
            | Value::LocalDate(value)
            | Value::LocalTime(value) => visitor.visit_str(value),
            Value::Array(values) => visitor.visit_seq(TomlSeqAccess {
                values: values.iter(),
            }),
            Value::Table(values) => visitor.visit_map(TomlMapAccess {
                iter: values.iter(),
                next_value: None,
            }),
        }
    }

    fn deserialize_shared<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        Err(TomlError::Unsupported("shared graph references"))
    }

    fn deserialize_hinted<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        Err(TomlError::Unsupported("hinted graph values"))
    }

    fn deserialize_array<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        Err(TomlError::Unsupported("graph arrays"))
    }

    fn deserialize_bool<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::Bool(value) => visitor.visit_bool(*value),
            _ => self.deserialize_any(visitor),
        }
    }

    deserialize_signed!(deserialize_i8);
    deserialize_signed!(deserialize_i16);
    deserialize_signed!(deserialize_i32);
    deserialize_signed!(deserialize_i64);
    deserialize_unsigned!(deserialize_u8);
    deserialize_unsigned!(deserialize_u16);
    deserialize_unsigned!(deserialize_u32);
    deserialize_unsigned!(deserialize_u64);
    deserialize_float!(deserialize_f32);
    deserialize_float!(deserialize_f64);

    fn deserialize_char<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_str(visitor)
    }

    fn deserialize_str<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_string(visitor)
    }

    fn deserialize_string<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::String(value)
            | Value::OffsetDateTime(value)
            | Value::LocalDateTime(value)
            | Value::LocalDate(value)
            | Value::LocalTime(value) => visitor.visit_str(value),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_bytes<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_byte_buf(visitor)
    }

    fn deserialize_byte_buf<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::String(value) => visitor.visit_bytes(value.as_bytes()),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_option<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_some(self)
    }

    fn deserialize_unit<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        Err(TomlError::Unsupported("unit/null values"))
    }

    fn deserialize_unit_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::Array(values) => visitor.visit_seq(TomlSeqAccess {
                values: values.iter(),
            }),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_tuple<V: DataVisitor<'src>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.value {
            Value::Table(values) => visitor.visit_map(TomlMapAccess {
                iter: values.iter(),
                next_value: None,
            }),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _fields: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _variants: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_enum(TomlEnumAccess { value: self.value })
    }

    fn deserialize_identifier<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_string(visitor)
    }

    fn deserialize_ignored_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_unit()
    }
}

impl<'src, S: TextSource<'src>> Deserializer<'src> for &mut TomlParser<S> {
    type Error = TomlError;

    fn deserialize_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_any(visitor)
    }

    fn deserialize_shared<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_hinted<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_array<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_bool<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_bool(visitor)
    }

    deserialize_signed!(deserialize_i8);
    deserialize_signed!(deserialize_i16);
    deserialize_signed!(deserialize_i32);
    deserialize_signed!(deserialize_i64);
    deserialize_unsigned!(deserialize_u8);
    deserialize_unsigned!(deserialize_u16);
    deserialize_unsigned!(deserialize_u32);
    deserialize_unsigned!(deserialize_u64);
    deserialize_float!(deserialize_f32);
    deserialize_float!(deserialize_f64);

    fn deserialize_char<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_str(visitor)
    }

    fn deserialize_str<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_str(visitor)
    }

    fn deserialize_string<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_byte_buf(visitor)
    }

    fn deserialize_byte_buf<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_byte_buf(visitor)
    }

    fn deserialize_option<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_option(visitor)
    }

    fn deserialize_unit<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_unit(visitor)
    }

    fn deserialize_unit_struct<V: DataVisitor<'src>>(
        self,
        name: &str,
        visitor: V,
    ) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_unit_struct(name, visitor)
    }

    fn deserialize_newtype_struct<V: DataVisitor<'src>>(
        self,
        name: &str,
        visitor: V,
    ) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_newtype_struct(name, visitor)
    }

    fn deserialize_seq<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_seq(visitor)
    }

    fn deserialize_tuple<V: DataVisitor<'src>>(self, len: usize, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_tuple(len, visitor)
    }

    fn deserialize_tuple_struct<V: DataVisitor<'src>>(
        self,
        name: &str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_tuple_struct(name, len, visitor)
    }

    fn deserialize_map<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_map(visitor)
    }

    fn deserialize_struct<V: DataVisitor<'src>>(
        self,
        name: &str,
        fields: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_struct(name, fields, visitor)
    }

    fn deserialize_enum<V: DataVisitor<'src>>(
        self,
        name: &str,
        variants: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_enum(name, variants, visitor)
    }

    fn deserialize_identifier<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.parse_if_needed()?;
        TomlValueDeserializer::new(self.parsed.as_ref().expect("TOML document was parsed"))
            .deserialize_ignored_any(visitor)
    }
}

struct TomlSeqAccess<'a> {
    values: std::slice::Iter<'a, Value>,
}

impl<'src> SeqAccess<'src> for TomlSeqAccess<'_> {
    type Error = TomlError;

    fn next_element_seed<T: DeserializeSeed<'src>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        match self.values.next() {
            Some(value) => seed
                .deserialize(TomlValueDeserializer::new(value))
                .map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.values.len())
    }
}

struct TomlMapAccess<'a> {
    iter: std::collections::btree_map::Iter<'a, String, Value>,
    next_value: Option<&'a Value>,
}

impl<'src> MapAccess<'src> for TomlMapAccess<'_> {
    type Error = TomlError;

    fn next_key_seed<K: DeserializeSeed<'src>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        match self.iter.next() {
            Some((key, value)) => {
                self.next_value = Some(value);
                seed.deserialize(TomlKeyDeserializer { key }).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'src>>(&mut self, seed: V) -> Result<V::Value> {
        let value = self
            .next_value
            .take()
            .ok_or_else(|| TomlError::Expected("a TOML map value"))?;
        seed.deserialize(TomlValueDeserializer::new(value))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

struct TomlKeyDeserializer<'a> {
    key: &'a str,
}

impl TomlKeyDeserializer<'_> {
    fn deserialize_integer<'src, V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let value = self.key.parse::<i64>()?;
        visitor.visit_i64(value)
    }

    fn deserialize_unsigned<'src, V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let value = self.key.parse::<u64>()?;
        visitor.visit_u64(value)
    }

    fn deserialize_float<'src, V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let value = self.key.parse::<f64>()?;
        visitor.visit_f64(value)
    }
}

impl<'src> Deserializer<'src> for TomlKeyDeserializer<'_> {
    type Error = TomlError;

    fn deserialize_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_str(self.key)
    }

    fn deserialize_shared<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_hinted<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_array<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_bool<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.key {
            "true" => visitor.visit_bool(true),
            "false" => visitor.visit_bool(false),
            _ => self.deserialize_any(visitor),
        }
    }

    deserialize_signed!(deserialize_i8);
    deserialize_signed!(deserialize_i16);
    deserialize_signed!(deserialize_i32);
    deserialize_signed!(deserialize_i64);
    deserialize_unsigned!(deserialize_u8);
    deserialize_unsigned!(deserialize_u16);
    deserialize_unsigned!(deserialize_u32);
    deserialize_unsigned!(deserialize_u64);
    deserialize_float!(deserialize_f32);
    deserialize_float!(deserialize_f64);

    fn deserialize_char<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_str(visitor)
    }

    fn deserialize_str<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_str(self.key)
    }

    fn deserialize_string<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_str(self.key)
    }

    fn deserialize_bytes<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_bytes(self.key.as_bytes())
    }

    fn deserialize_byte_buf<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_bytes(self.key.as_bytes())
    }

    fn deserialize_option<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_some(self)
    }

    fn deserialize_unit<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        Err(TomlError::Unsupported("unit/null map keys"))
    }

    fn deserialize_unit_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        Err(TomlError::Unsupported("sequence map keys"))
    }

    fn deserialize_tuple<V: DataVisitor<'src>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        Err(TomlError::Unsupported("map keys"))
    }

    fn deserialize_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _fields: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _variants: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_enum(TomlEnumAccess {
            value: &Value::String(self.key.to_string()),
        })
    }

    fn deserialize_identifier<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_str(self.key)
    }

    fn deserialize_ignored_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_unit()
    }
}

struct TomlEnumAccess<'a> {
    value: &'a Value,
}

struct TomlVariantAccess<'a> {
    value: Option<&'a Value>,
}

impl<'src, 'a> EnumAccess<'src> for TomlEnumAccess<'a> {
    type Error = TomlError;
    type Variant = TomlVariantAccess<'a>;

    fn variant_seed<VS: DeserializeSeed<'src>>(
        self,
        seed: VS,
    ) -> Result<(VS::Value, Self::Variant)> {
        match self.value {
            Value::String(variant) => {
                let variant = seed.deserialize(TomlKeyDeserializer { key: variant })?;
                Ok((variant, TomlVariantAccess { value: None }))
            }
            Value::Table(table) if table.len() == 1 => {
                let (variant, value) = table.iter().next().expect("table has one entry");
                let variant = seed.deserialize(TomlKeyDeserializer { key: variant })?;
                Ok((variant, TomlVariantAccess { value: Some(value) }))
            }
            _ => Err(TomlError::Expected("a TOML enum value")),
        }
    }
}

impl<'src> VariantAccess<'src> for TomlVariantAccess<'_> {
    type Error = TomlError;

    fn unit_variant(self) -> Result<()> {
        Ok(())
    }

    fn newtype_variant_seed<T: DeserializeSeed<'src>>(self, seed: T) -> Result<T::Value> {
        let value = self
            .value
            .ok_or_else(|| TomlError::Expected("a TOML enum variant value"))?;
        seed.deserialize(TomlValueDeserializer::new(value))
    }

    fn tuple_variant<V: DataVisitor<'src>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        let value = self
            .value
            .ok_or_else(|| TomlError::Expected("a TOML enum tuple variant"))?;
        TomlValueDeserializer::new(value).deserialize_seq(visitor)
    }

    fn struct_variant<V: DataVisitor<'src>>(self, fields: &[&str], visitor: V) -> Result<V::Value> {
        let value = self
            .value
            .ok_or_else(|| TomlError::Expected("a TOML enum struct variant"))?;
        TomlValueDeserializer::new(value).deserialize_struct("", fields, visitor)
    }
}
