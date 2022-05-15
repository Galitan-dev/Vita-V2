use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

// FUNCTIONS

pub fn read_to_bytes<P>(path: P) -> io::Result<Vec<u8>>
where
    P: AsRef<Path>,
{
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();

    // Read file into vector.
    reader.read_to_end(&mut buffer)?;

    Ok(buffer)
}
