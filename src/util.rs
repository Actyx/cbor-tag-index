use std::{io, iter::FromIterator};

use libipld::{
    cbor::{
        decode::{read_len, read_u8},
        error::UnexpectedCode,
        DagCborCodec,
    },
    codec::Decode,
};

/// Like the one from itertools, but more convenient
pub(crate) enum EitherIter<L, R> {
    Left(L),
    Right(R),
}

impl<L, R, T> Iterator for EitherIter<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;
    fn next(&mut self) -> std::option::Option<<Self as Iterator>::Item> {
        match self {
            Self::Left(l) => l.next(),
            Self::Right(r) => r.next(),
        }
    }
}

#[allow(dead_code)]
pub(crate) type BoxedIter<'a, T> = Box<dyn Iterator<Item = T> + Send + 'a>;

/// Some convenience fns so we don't have to depend on IterTools
pub(crate) trait IterExt<'a>
where
    Self: Iterator + Sized + Send + 'a,
{
    fn boxed(self) -> BoxedIter<'a, Self::Item> {
        Box::new(self)
    }

    fn left_iter<R>(self) -> EitherIter<Self, R> {
        EitherIter::Left(self)
    }

    fn right_iter<L>(self) -> EitherIter<L, Self> {
        EitherIter::Right(self)
    }
}

impl<'a, T: Iterator + Sized + Send + 'a> IterExt<'a> for T {}

/// Create an iterator from a faillble fn
///
/// the fn will be called until it returns either Ok(None) or Err(...)
pub(crate) fn from_fallible_fn<'a, T: 'a>(
    mut f: impl FnMut() -> anyhow::Result<Option<T>> + 'a,
) -> impl Iterator<Item = anyhow::Result<T>> + 'a {
    let mut done = false;
    std::iter::from_fn(move || match f() {
        Err(cause) if !done => {
            done = true;
            Some(Err(cause))
        }
        Ok(Some(value)) if !done => Some(Ok(value)),
        _ => None,
    })
}

pub fn read_seq<C, R, T>(r: &mut R) -> C
where
    C: FromIterator<anyhow::Result<T>>,
    R: io::Read + io::Seek,
    T: Decode<DagCborCodec>,
{
    let inner = |r: &mut R| -> anyhow::Result<C> {
        let major = read_u8(r)?;
        let result = match major {
            0x80..=0x9b => {
                let len = read_len(r, major - 0x80)?;
                read_seq_fl(r, len)
            }
            0x9f => read_seq_il(r),
            _ => {
                return Err(UnexpectedCode::new::<C>(major).into());
            }
        };
        Ok(result)
    };
    // this is just so we don't have to return anyhow::Result<anyhow::Result<C>>
    match inner(r) {
        Ok(value) => value,
        Err(cause) => C::from_iter(Some(Err(cause))),
    }
}

/// read a fixed length cbor sequence into a generic collection that implements FromIterator
pub fn read_seq_fl<C, R, T>(r: &mut R, len: usize) -> C
where
    C: FromIterator<anyhow::Result<T>>,
    R: io::Read + io::Seek,
    T: Decode<DagCborCodec>,
{
    (0..len).map(|_| T::decode(DagCborCodec, r)).collect()
}

/// read an indefinite length cbor sequence into a generic collection that implements FromIterator
pub fn read_seq_il<C, R, T>(r: &mut R) -> C
where
    C: FromIterator<anyhow::Result<T>>,
    R: io::Read + io::Seek,
    T: Decode<DagCborCodec>,
{
    from_fallible_fn(|| -> anyhow::Result<Option<T>> {
        let major = read_u8(r)?;
        if major == 0xff {
            return Ok(None);
        }
        r.seek(io::SeekFrom::Current(-1))?;
        let value = T::decode(DagCborCodec, r)?;
        Ok(Some(value))
    })
    .collect()
}
