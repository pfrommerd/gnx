use crate::util::TextSource;

use super::{Result, TomlError};

fn is_forbidden_comment_char(ch: char) -> bool {
    matches!(ch, '\u{00}'..='\u{08}' | '\u{0a}'..='\u{1f}' | '\u{7f}')
}

pub trait TomlSource<'src>: TextSource<'src> {
    fn consume_ws(&mut self) -> Result<()> {
        loop {
            let consumed = self.peek_and(|ch| match ch {
                Some(' ' | '\t') => (true, true),
                _ => (false, false),
            })?;
            if !consumed {
                return Ok(());
            }
        }
    }

    fn consume_newline(&mut self) -> Result<bool> {
        match self.peek()? {
            Some('\n') => {
                self.next()?;
                Ok(true)
            }
            Some('\r') => {
                self.next()?;
                match self.next()? {
                    Some('\n') => Ok(true),
                    Some(ch) => Err(TomlError::Unexpected(ch)),
                    None => Err(TomlError::UnexpectedEOF),
                }
            }
            _ => Ok(false),
        }
    }

    fn consume_comment(&mut self) -> Result<bool> {
        if self.peek()? != Some('#') {
            return Ok(false);
        }
        self.next()?;
        loop {
            match self.peek()? {
                Some('\n' | '\r') | None => return Ok(true),
                Some(ch) if is_forbidden_comment_char(ch) => {
                    return Err(TomlError::Unexpected(ch));
                }
                Some(_) => {
                    self.next()?;
                }
            }
        }
    }

    fn consume_ws_and_comments(&mut self) -> Result<()> {
        loop {
            self.consume_ws()?;
            if !self.consume_comment()? {
                return Ok(());
            }
        }
    }

    fn consume_ws_comments_newlines(&mut self) -> Result<()> {
        loop {
            self.consume_ws()?;
            if self.consume_comment()? {
                continue;
            }
            if self.consume_newline()? {
                continue;
            }
            return Ok(());
        }
    }

    fn consume_line_end_or_eof(&mut self) -> Result<()> {
        self.consume_ws()?;
        self.consume_comment()?;
        self.consume_ws()?;
        match self.peek()? {
            None => Ok(()),
            Some('\n' | '\r') => {
                self.consume_newline()?;
                Ok(())
            }
            Some(ch) => Err(TomlError::Unexpected(ch)),
        }
    }
}

impl<'src, T: TextSource<'src>> TomlSource<'src> for T {}
