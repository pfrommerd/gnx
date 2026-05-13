use gnx_io::util::{BufferedSource, RawStrSource, ScratchBuffer, TextSource};

use std::io::Error;

#[test]
fn test_text_source() {
    let mut source = RawStrSource::from("H\u{10000}llo!");
    let pos = || -> Result<usize, Error> {
        assert_eq!(source.next()?, Some('H'));
        let mut outer_buf = source.buffering(&mut ScratchBuffer::new());
        assert_eq!(outer_buf.next()?, Some('\u{10000}'));

        let pos = outer_buf.position().unwrap();
        let mut buf = outer_buf.buffering(&mut ScratchBuffer::new());
        assert_eq!(buf.next()?, Some('l'));
        assert_eq!(buf.next()?, Some('l'));
        assert_eq!(buf.next()?, Some('o'));
        // Get the buffered text
        assert_eq!(buf.into_buffer().into_string(), "llo");
        assert_eq!(outer_buf.next()?, Some('!'));
        assert_eq!(outer_buf.into_buffer().into_string(), "\u{10000}llo!");

        assert_eq!(source.next()?, None);
        Ok(pos)
    }().unwrap();
    // Rewind...
    source.goto(pos).unwrap();
    assert_eq!(source.next().unwrap(), Some('l'));
}