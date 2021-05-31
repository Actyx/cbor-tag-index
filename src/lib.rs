#![allow(clippy::type_complexity)]
#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use fmt::{Display, Write};
use fnv::FnvHashMap;
use libipld::{
    cbor::DagCbor,
    codec::{Decode, Encode},
};
use libipld_cbor::DagCborCodec;
use std::hash::Hash;
use std::{convert::TryFrom, fmt, iter::FromIterator, mem::swap, usize};
mod bitmap;
mod util;
use bitmap::*;
#[cfg(test)]
mod arb;
#[cfg(test)]
mod size_tests;
pub trait Tag: PartialEq + Eq + Hash + Ord + Clone + 'static {}

impl<T: PartialEq + Eq + Hash + Ord + Clone + 'static> Tag for T {}

/// a set of tags
#[cfg(test)]
pub type TagSet<T> = vec_collections::VecSet<[T; 4]>;

/// A compact representation of a seq of tag sets,
///
/// to be used as a DNF query.
///
/// `tags` are a sequence of strings, where the offset corresponds to the
/// set bit in the bitmap.
///
/// E.g. ("a" & "b") | ("b" & "c") | ("d") would be encoded as
///
/// {
///   tags: ["a", "b", "c", "d"],
///   sets: [
///     b0011,
///     b0110,
///     b1000,
///   ]
/// }
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DnfQuery<T: Tag> {
    tags: Vec<T>,
    sets: Bitmap,
}

impl<T: Tag + DagCbor> Encode<DagCborCodec> for DnfQuery<T> {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        w.write_all(&[0x82])?;
        self.tags.encode(c, w)?;
        self.sets.encode(c, w)?;
        Ok(())
    }
}

impl<T: Tag + DagCbor> Decode<DagCborCodec> for DnfQuery<T> {
    fn decode<R: std::io::Read + std::io::Seek>(
        c: DagCborCodec,
        r: &mut R,
    ) -> anyhow::Result<Self> {
        let (tags, sets) = <(Vec<T>, Bitmap)>::decode(c, r)?;
        Ok(Self { tags, sets })
    }
}

/// A tag index, using bitmaps to encode the distinct tag sets, and a vector
/// of offsets for each event.
///
/// A sequence of events with the following tag sets:
///
/// `[{"a"}, {"a", "b"}, {"b","c"}, {"a"}]`
///
/// would be encoded like this:
///
/// ```javascript
/// {
///   tags: {
///     tags: { "a": 0, "b": 1, "c": 2 },
///     sets: [
///       b001, //   a
///       b010, //  b
///       b110, // cb
///     ],
///   },
///   events: [
///     0, // first bitset
///     1, // 2nd bitset
///     2, // 3rd bitset
///     0, // first bitset again
///   ],
/// }
///```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagIndex<T: Tag> {
    /// efficiently encoded distinct tags
    tags: DnfQuery<T>,
    /// tag offset for each event
    events: Vec<u32>,
}

impl<T: Tag + DagCbor> Encode<DagCborCodec> for TagIndex<T> {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        w.write_all(&[0x82])?;
        self.tags.encode(c, w)?;
        self.events.encode(c, w)?;
        Ok(())
    }
}

impl<T: Tag + DagCbor> Decode<DagCborCodec> for TagIndex<T> {
    fn decode<R: std::io::Read + std::io::Seek>(
        c: DagCborCodec,
        r: &mut R,
    ) -> anyhow::Result<Self> {
        let (tags, events) = <(DnfQuery<T>, Vec<u32>)>::decode(c, r)?;
        Ok(Self { tags, events })
    }
}

impl<T: Tag> DnfQuery<T> {
    pub fn new(
        terms: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    ) -> anyhow::Result<Self> {
        let mut builder = DnfQueryBuilder::new();
        for term in terms {
            builder.push(term)?;
        }
        Ok(builder.dnf_query())
    }

    /// We use DnfQuery for both the queries and the index against which they are run.
    /// This is for the latter: assuming this is an index, execute the dnf query
    pub(crate) fn dnf_query(&self, dnf: &DnfQuery<T>, result: &mut [bool]) {
        // this mapping could be done more efficiently, since both dnf.tags and our tags are ordered
        let translate = dnf
            .tags
            .iter()
            .map(|tag| self.tags.binary_search(tag).map(|x| x as u32).ok())
            .collect::<Box<_>>();
        dnf_query0(&self.sets, dnf, &translate, result);
    }

    /// An empty dnf query which matches nothing
    pub fn empty() -> Self {
        Self {
            tags: Default::default(),
            sets: Default::default(),
        }
    }

    /// An empty dnf query which matches nothing
    pub fn is_empty(&self) -> bool {
        self.sets.rows() == 0
    }

    /// a dnf query containing an empty set, which matches everything
    pub fn all() -> Self {
        Self {
            tags: Default::default(),
            sets: Bitmap::new(vec![vec![]]),
        }
    }

    /// a dnf query containing an empty set, which matches everything
    pub fn is_all(&self) -> bool {
        self.sets.iter().any(|row| row.count() == 0)
    }

    /// Helper method to return the matching indexes, mostly for tests
    pub fn matching(&self, index: &TagIndex<T>) -> Vec<bool> {
        let mut matching = vec![true; index.len()];
        self.set_matching(index, &mut matching);
        matching
    }

    /// given a bitmap of matches, corresponding to the events in the index,
    /// set those bytes to false that do not match.
    pub fn set_matching(&self, index: &TagIndex<T>, matches: &mut [bool]) {
        // create a bool array corresponding to the distinct tagsets in the index
        let mut tmp = vec![false; index.tags.sets.rows()];
        // set the fields we need to look at to true
        for (matching, index) in matches.iter().zip(index.events.iter()) {
            if *matching {
                tmp[*index as usize] = true;
            }
        }
        // evaluate the dnf query for these fields
        index.tags.dnf_query(&self, &mut tmp);
        // write result from tmp
        for (matching, index) in matches.iter_mut().zip(index.events.iter()) {
            *matching = *matching && tmp[*index as usize];
        }
    }

    /// get back the terms making up the dnf query
    ///
    /// Note that there is no guarantee that there will be the same number of terms or that
    /// tags in each term will be ordered in the same way.
    pub fn terms(&self) -> impl Iterator<Item = impl Iterator<Item = &T> + '_> + '_ {
        self.sets
            .iter()
            .map(move |rows| rows.map(move |index| &self.tags[index as usize]))
    }

    pub fn term(&self, index: usize) -> impl Iterator<Item = &T> + '_ {
        self.sets.row(index).map(move |i| &self.tags[i as usize])
    }

    pub fn term_count(&self) -> usize {
        self.sets.rows()
    }
}

impl<T: Tag + Display> Display for DnfQuery<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let term_to_string = |term: Vec<T>| -> String {
            term.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("&")
        };
        let res = self
            .terms()
            .map(|x| term_to_string(x.cloned().collect()))
            .collect::<Vec<_>>()
            .join(" | ");
        f.write_str(&res)
    }
}

struct DebugUsingDisplay<T>(Vec<T>);

impl<T: Display> fmt::Debug for DebugUsingDisplay<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('{')?;
        for (i, x) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_char(',')?;
            }
            Display::fmt(x, f)?;
        }
        f.write_char('}')?;
        Ok(())
    }
}

impl<T> FromIterator<T> for DebugUsingDisplay<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<T: Tag> TagIndex<T> {
    pub fn new(e: impl IntoIterator<Item = impl IntoIterator<Item = T>>) -> anyhow::Result<Self> {
        let mut builder = DnfQueryBuilder::new();
        let events = e
            .into_iter()
            .map(|set| builder.push(set))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            tags: builder.dnf_query(),
            events,
        })
    }

    pub fn distinct_tags(&self) -> &[T] {
        &self.tags.tags
    }

    pub fn tags(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> + '_ {
        self.events
            .iter()
            .map(move |offset| self.tags.term(*offset as usize))
    }

    pub fn get<C: FromIterator<T>>(&self, index: usize) -> Option<C> {
        let mask_index = self.events.get(index)?;
        let mask = self.tags.sets.row(*mask_index as usize);
        let lut = &self.tags.tags;
        Some(mask.map(|i| lut[i as usize].clone()).collect())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_dense(&self) -> bool {
        self.tags.sets.is_dense()
    }

    pub fn distinct_sets(&self) -> usize {
        self.tags.sets.rows()
    }
}

impl<T: Tag + Display> Display for TagIndex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(
                self.tags()
                    .map(|ts| ts.into_iter().collect::<DebugUsingDisplay<_>>()),
            )
            .finish()
    }
}

/// Turns an std::slice::IterMut<T> into an interator of T provided T has a default
///
/// This makes sense for cases where cloning is expensive, but default is cheap. E.g. Vec<T>.
struct SliceIntoIter<'a, T>(std::slice::IterMut<'a, T>);

impl<'a, T: Default + 'a> Iterator for SliceIntoIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| {
            let mut r = T::default();
            swap(x, &mut r);
            r
        })
    }
}

/// performs a dnf query on an index, given a lookup table to translate from the dnf query to the index domain
fn dnf_query0<T: Tag>(
    index: &Bitmap,
    dnf: &DnfQuery<T>,
    translate: &[Option<u32>],
    result: &mut [bool],
) {
    match index {
        Bitmap::Sparse(index) => {
            let dnf = SparseBitmap::new(
                dnf.sets
                    .iter()
                    .filter_map(|row| {
                        row.map(|index| translate[index as usize])
                            .collect::<Option<IndexSet>>()
                    })
                    .collect(),
            );
            for (set, value) in index.iter().zip(result.iter_mut()) {
                *value = *value && { dnf.iter().any(move |query| query.is_subset(set)) }
            }
        }
        Bitmap::Dense(index) => {
            let dnf = DenseBitmap::new(
                dnf.sets
                    .iter()
                    .filter_map(|row| {
                        // if a single index in the row can not be mapped (translate[index] is None),
                        // we want to skip the entire term.
                        //
                        // The somewhat convoluted way to do this is to make the mask_from_bits_iter fail
                        // by passing an index that will make mask_from_bits_iter fail.
                        mask_from_bits_iter(
                            row.map(|index| translate[index as usize].unwrap_or(MIN_SPARSE_INDEX)),
                        )
                    })
                    .collect(),
            );
            for (mask, value) in index.iter().zip(result.iter_mut()) {
                *value = *value && { dnf.iter().any(move |query| is_subset(*query, *mask)) }
            }
        }
    }
}

#[inline]
fn is_subset(a: IndexMask, b: IndexMask) -> bool {
    a & b == a
}

pub(crate) struct TagSetSetIter<'a, T>(&'a [T], BitmapRowsIter<'a>);

impl<'a, T> Iterator for TagSetSetIter<'a, T> {
    type Item = TagRefIter<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|iter| TagRefIter(self.0, iter))
    }
}

pub(crate) struct TagRefIter<'a, T>(&'a [T], BitmapRowIter<'a>);

impl<'a, T> Iterator for TagRefIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|index| &self.0[index as usize])
    }
}

#[derive(Debug)]
pub(crate) struct DnfQueryBuilder<T: Tag> {
    tags: FnvHashMap<T, u32>,
    sets: FnvHashMap<IndexSet, u32>,
}

impl<T: Tag> DnfQueryBuilder<T> {
    pub fn new() -> Self {
        Self {
            tags: FnvHashMap::default(),
            sets: Default::default(),
        }
    }

    fn permutation_table(&self) -> Vec<u32> {
        let mut tags = self.tags.iter().collect::<Vec<_>>();
        tags.sort_unstable_by_key(|(key, _)| *key);
        let mut permutation_table = vec![0u32; tags.len()];
        for (i, (_, j)) in tags.into_iter().enumerate() {
            permutation_table[*j as usize] = i as u32;
        }
        permutation_table
    }

    /// Return the result as a [DnfQuery]
    pub fn dnf_query(self) -> DnfQuery<T> {
        let perm = self.permutation_table();
        let mut tags = vec![None; self.tags.len()];
        for (tag, index) in self.tags {
            tags[perm[index as usize] as usize] = Some(tag)
        }
        let tags = tags.into_iter().flatten().collect();
        let mut sets = vec![IndexSet::default(); self.sets.len()];
        for (set, index) in self.sets {
            sets[index as usize] = set
        }
        let sets = sets
            .into_iter()
            .map(|indexes| indexes.into_iter().map(|index| perm[index as usize]))
            .collect();
        DnfQuery { tags, sets }
    }

    pub fn push(&mut self, tags: impl IntoIterator<Item = T>) -> anyhow::Result<u32> {
        let indices = tags.into_iter().map(|tag| self.add_tag(&tag));
        let set = indices.collect::<anyhow::Result<IndexSet>>()?;
        Ok(if let Some(index) = self.sets.get(&set) {
            *index
        } else {
            let index = u32::try_from(self.sets.len())?;
            self.sets.insert(set, index);
            index
        })
    }

    fn add_tag(&mut self, tag: &T) -> anyhow::Result<u32> {
        Ok(if let Some(index) = self.tags.get(tag) {
            *index
        } else {
            let index = u32::try_from(self.tags.len())?;
            self.tags.insert(tag.clone(), index);
            index
        })
    }
}

#[cfg(test)]
mod tests {
    use libipld::{
        cbor::DagCborCodec,
        codec::{assert_roundtrip, Codec},
        ipld, DagCbor,
    };
    use quickcheck::Arbitrary;

    /// our toy tag
    #[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, DagCbor)]
    #[ipld(repr = "value")]
    pub struct TestTag(pub String);

    impl fmt::Display for TestTag {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl fmt::Debug for TestTag {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl Arbitrary for TestTag {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let tag = g.choose(TAG_NAMES).unwrap();
            TestTag((*tag).to_owned())
        }
    }

    const TAG_NAMES: &[&str] = &["a", "b", "c", "d", "e", "f"];

    // create a test tag set - each alphanumeric char will be converted to an individual tag.
    fn ts(tags: &str) -> TagSet<TestTag> {
        tags.chars()
            .filter(|c| char::is_alphanumeric(*c))
            .map(|x| TestTag(x.to_string()))
            .collect()
    }

    // create a sequence of tag sets, separated by ,
    fn tss(tags: &str) -> Vec<TagSet<TestTag>> {
        tags.split(',').map(ts).collect()
    }

    // create a dnf query, separated by |
    fn dnf(tags: &str) -> DnfQuery<TestTag> {
        let parts = tags.split('|').map(ts).collect::<Vec<_>>();
        DnfQuery::new(parts).unwrap()
    }

    // create a dnf query, separated by |
    fn ti(tags: &str) -> TagIndex<TestTag> {
        TagIndex::new(tss(tags)).unwrap()
    }

    use super::*;

    fn matches(index: &str, query: &str) -> String {
        let index = ti(index);
        let query = dnf(query);
        let mut matches = vec![true; index.len()];
        query.set_matching(&index, &mut matches);
        matches.iter().map(|x| if *x { '1' } else { '0' }).collect()
    }

    #[test]
    fn tag_index_query_tests() {
        assert_eq!(&matches(" a,ab,bc, a", "ab"), "0100");
        assert_eq!(&matches(" a, a, a, a", "ab"), "0000");
        assert_eq!(&matches(" a, a, a,ab", "ab|c|d"), "0001");
    }

    #[test]
    fn dnf_query() {
        // sequence containing empty sets, matches everything
        assert_eq!(&matches("ab,ac,bc", " ||"), "111");
        assert_eq!(&matches("ab,ac,bc", "  a"), "110");
        assert_eq!(&matches("ab,ac,bc", " ab"), "100");
        assert_eq!(&matches("ab,ac,bc", " ax"), "000");
        assert_eq!(&matches("ab,ac,bc", "  c"), "011");
        assert_eq!(&matches("ab,bc,cd", "a|d"), "101");
    }

    #[quickcheck]
    fn set_matching(index: TagIndex<TestTag>, query: DnfQuery<TestTag>) -> bool {
        let mut bits1 = vec![true; index.len()];
        let mut bits2 = vec![false; index.len()];
        query.set_matching(&index, &mut bits1);

        for (tags, matching) in index.tags().zip(bits2.iter_mut()) {
            let tags: TagSet<TestTag> = tags.cloned().collect();
            *matching = query
                .terms()
                .map(|x| x.cloned().collect::<TagSet<TestTag>>())
                .any(|q| q.is_subset(&tags))
        }
        let bt = bits1
            .iter()
            .map(|x| if *x { '1' } else { '0' })
            .collect::<String>();
        println!("{} {} {}", index, query, bt);
        bits1 == bits2
    }

    #[test]
    fn dnf_query_ipld() {
        let query = dnf("zab|bc|def|gh");
        let expected = ipld! {
            [["a", "b", "c", "d", "e", "f", "g", "h", "z"], [[0, 1, 8], [1, 2], [3, 4, 5], [6, 7]]]
        };
        let data = DagCborCodec.encode(&query).unwrap();
        println!("{}", hex::encode(data));
        assert_roundtrip(DagCborCodec, &query, &expected);
    }

    #[quickcheck]
    fn dnf_query_cbor_roundtrip(value: DnfQuery<TestTag>) -> bool {
        let bytes = DagCborCodec.encode(&value).unwrap();
        let value1 = DagCborCodec.decode(&bytes).unwrap();
        value == value1
    }

    #[quickcheck]
    fn tag_index_cbor_roundtrip(value: TagIndex<TestTag>) -> bool {
        let bytes = DagCborCodec.encode(&value).unwrap();
        let value1 = DagCborCodec.decode(&bytes).unwrap();
        value == value1
    }
}
