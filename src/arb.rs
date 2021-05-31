use super::*;
use quickcheck::Arbitrary;
use std::collections::{BTreeSet, HashSet};
use vec_collections::VecSet;

type TagSet<T> = VecSet<[T; 4]>;

#[derive(Clone)]
struct TagSetHelper<T>(TagSet<T>);

impl<T: Arbitrary + Ord> Arbitrary for TagSetHelper<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let items: BTreeSet<T> = Arbitrary::arbitrary(g);
        TagSetHelper(items.into_iter().collect())
    }
}

impl Arbitrary for Bitmap {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let items: Vec<HashSet<u32>> = Arbitrary::arbitrary(g);
        Bitmap::new(items)
    }
}

impl<T: Tag + Arbitrary> Arbitrary for DnfQuery<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let tags: Vec<TagSetHelper<T>> = Arbitrary::arbitrary(g);
        let mut tags: Vec<TagSet<T>> = tags.into_iter().map(|x| x.0).collect();
        tags.truncate(3);
        DnfQuery::new(tags).unwrap()
    }
}

impl<T: Tag + Arbitrary> Arbitrary for TagIndex<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let tags: Vec<TagSetHelper<T>> = Arbitrary::arbitrary(g);
        let tags: Vec<TagSet<T>> = tags.into_iter().map(|x| x.0).collect();
        TagIndex::new(tags).unwrap()
    }
}
