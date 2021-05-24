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
use vec_collections::VecSet;
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
pub type TagSet<T> = VecSet<[T; 4]>;

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

/// Same as a [DnfQuery], but with optimized support for translation.
///
/// In this representation, `tags` is a map from tag to index.
///
/// E.g. ("a" & "b") | ("b" & "c") | ("d") would be encoded as
///
/// {
///   tags: {
///     "a": 0,
///     "b": 1,
///     "c": 2,
///     "d": 3
///   },
///   sets: [
///     b0011,
///     b0110,
///     b1000,
///   ]
/// }
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TagSetSet<T: Tag> {
    tags: FnvHashMap<T, u32>,
    sets: Bitmap,
}

impl<T: Tag + DagCbor> Encode<DagCborCodec> for TagSetSet<T> {
    fn encode<W: std::io::Write>(&self, c: DagCborCodec, w: &mut W) -> anyhow::Result<()> {
        DnfQuery::from(self.clone()).encode(c, w)
    }
}

impl<T: Tag + DagCbor> Decode<DagCborCodec> for TagSetSet<T> {
    fn decode<R: std::io::Read + std::io::Seek>(
        c: DagCborCodec,
        r: &mut R,
    ) -> anyhow::Result<Self> {
        DnfQuery::decode(c, r).map(Into::into)
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
    tags: TagSetSet<T>,
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
        let (tags, events) = <(TagSetSet<T>, Vec<u32>)>::decode(c, r)?;
        Ok(Self { tags, events })
    }
}

impl<T: Tag> DnfQuery<T> {
    pub fn new<'a>(sets: impl IntoIterator<Item = &'a TagSet<T>>) -> anyhow::Result<Self> {
        let mut builder = DnfQueryBuilder::new();
        for set in sets {
            builder.push(&set)?;
        }
        Ok(builder.dnf_query())
    }

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

    /// get back the tag sets the dnf query
    pub fn terms(&self) -> impl Iterator<Item = TagSet<T>> + '_ {
        self.sets.iter().map(move |rows| {
            rows.map(|index| self.tags[index as usize].clone())
                .collect()
        })
    }

    pub fn term_count(&self) -> usize {
        self.sets.rows()
    }
}

impl<T: Tag + Display> Display for DnfQuery<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let term_to_string = |term: TagSet<T>| -> String {
            term.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("&")
        };
        let res = self
            .terms()
            .map(|term| term_to_string(term))
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
    pub fn new(e: &[TagSet<T>]) -> anyhow::Result<Self> {
        let mut builder = DnfQueryBuilder::new();
        let events = e
            .iter()
            .map(|set| builder.push(set))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            tags: builder.tag_set_set(),
            events,
        })
    }

    pub fn tags(&self) -> impl Iterator<Item = TagSet<T>> + '_ {
        let lut = self.tags.tags().collect::<Vec<_>>();
        self.events
            .iter()
            .map(move |offset| lut[*offset as usize].clone())
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

impl<T: Tag> From<DnfQuery<T>> for TagSetSet<T> {
    fn from(value: DnfQuery<T>) -> Self {
        let tags: FnvHashMap<T, u32> = value
            .tags
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, tag)| (tag, u32::try_from(index).unwrap()))
            .collect();
        Self {
            tags,
            sets: value.sets,
        }
    }
}

impl<T: Tag> From<TagSetSet<T>> for DnfQuery<T> {
    fn from(value: TagSetSet<T>) -> Self {
        let mut tags = vec![None; value.tags.len()];
        for (tag, index) in value.tags {
            tags[index as usize] = Some(tag)
        }
        let tags: Vec<T> = tags.into_iter().filter_map(|x| x).collect();
        Self {
            tags,
            sets: value.sets,
        }
    }
}

impl<T: Tag> TagSetSet<T> {
    #[cfg(test)]
    pub fn new(data: &[TagSet<T>]) -> anyhow::Result<Self> {
        let mut builder = DnfQueryBuilder::new();
        for set in data {
            builder.push(set)?;
        }
        Ok(builder.tag_set_set())
    }

    pub fn dnf_query(&self, dnf: &DnfQuery<T>, result: &mut [bool]) {
        let translate = dnf
            .tags
            .iter()
            .map(|tag| self.tags.get(tag).cloned())
            .collect::<Box<_>>();
        dnf_query0(&self.sets, dnf, &translate, result);
    }

    fn lut(&self) -> Vec<T> {
        let mut tags = vec![None; self.tags.len()];
        for (tag, index) in &self.tags {
            tags[*index as usize] = Some(tag)
        }
        tags.into_iter().filter_map(|x| x.cloned()).collect()
    }

    /// get back the tag sets the tag index was created from, in order
    pub fn tags(&self) -> impl Iterator<Item = TagSet<T>> + '_ {
        let lut = self.lut();
        self.sets
            .iter()
            .map(move |row| row.map(|i| lut[i as usize].clone()).collect::<TagSet<T>>())
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
                        mask_from_bits_iter(
                            row.map(|index| translate[index as usize].unwrap_or(128)),
                        )
                        .ok()
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

    /// Return the result as a [TagSetSet]
    pub(crate) fn tag_set_set(self) -> TagSetSet<T> {
        let tags = self.tags;
        let mut sets = vec![IndexSet::default(); self.sets.len()];
        for (set, index) in self.sets {
            sets[index as usize] = set
        }
        let sets = sets.into_iter().map(|x| x.into_iter()).collect();
        TagSetSet { tags, sets }
    }

    /// Return the result as a [DnfQuery]
    pub fn dnf_query(self) -> DnfQuery<T> {
        let mut tags = vec![None; self.tags.len()];
        for (tag, index) in self.tags {
            tags[index as usize] = Some(tag)
        }
        let tags = tags.into_iter().filter_map(|x| x).collect();
        let mut sets = vec![IndexSet::default(); self.sets.len()];
        for (set, index) in self.sets {
            sets[index as usize] = set
        }
        let sets = sets.into_iter().map(|x| x.into_iter()).collect();
        DnfQuery { tags, sets }
    }

    pub fn push(&mut self, tags: &TagSet<T>) -> anyhow::Result<u32> {
        let indices = tags.iter().map(|tag| self.add_tag(tag));
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
        DnfQuery::new(&parts).unwrap()
    }

    // create a dnf query, separated by |
    fn ti(tags: &str) -> TagIndex<TestTag> {
        TagIndex::new(&tss(tags)).unwrap()
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
    fn tag_index_query_tests() -> anyhow::Result<()> {
        assert_eq!(&matches(" a,ab,bc, a", "ab"), "0100");
        assert_eq!(&matches(" a, a, a, a", "ab"), "0000");
        assert_eq!(&matches(" a, a, a,ab", "ab|c|d"), "0001");
        Ok(())
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

        let query_tags = query.terms().collect::<Vec<_>>();
        for (tags, matching) in index.tags().zip(bits2.iter_mut()) {
            *matching = query_tags.iter().any(|q| q.is_subset(&tags))
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
            [["a", "b", "z", "c", "d", "e", "f", "g", "h"], [[0, 1, 1], [1, 2], [4, 1, 1], [7, 1]]]
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
    fn tag_set_set_cbor_roundtrip(value: TagSetSet<TestTag>) -> bool {
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
