use std::{io, iter::FromIterator};

use libipld::{
    cbor::{
        cbor::MajorKind,
        decode::{read_major, read_uint},
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

pub fn read_seq<C, R, T>(r: &mut R) -> C
where
    C: FromIterator<anyhow::Result<T>>,
    R: io::Read + io::Seek,
    T: Decode<DagCborCodec>,
{
    let inner = |r: &mut R| -> anyhow::Result<C> {
        let major = read_major(r)?;
        let result = match major.kind() {
            MajorKind::Array => {
                let len = read_uint(r, major)?;
                read_seq_fl(r, len)
            }
            _ => {
                return Err(UnexpectedCode::new::<C>(major.into()).into());
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
fn read_seq_fl<C, R, T>(r: &mut R, len: u64) -> C
where
    C: FromIterator<anyhow::Result<T>>,
    R: io::Read + io::Seek,
    T: Decode<DagCborCodec>,
{
    (0..len).map(|_| T::decode(DagCborCodec, r)).collect()
}
