use crate::{DnfQuery, TagIndex, TagSet};
use libipld::{cbor::DagCborCodec, codec::Codec};
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaChaRng;
use std::{io::Cursor, sync::Arc};
/// less than 128 distinct tags - dense index
const DENSE: &[(&str, usize)] = &[("article", 30), ("sku", 40), ("location", 40)];
// more than 128 distinct tags - sparse index
// const SPARSE: &[(&str, usize)] = &[("article", 30), ("sku", 40), ("location", 100)];
/// extra tags that are added randomly to all events
const EXTRA: &[&str] = &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];

fn create_example(
    recipe: &[(&str, usize)],
    extra: &[&str],
    seed: u64,
    n_events: usize,
    n_terms: usize,
) -> anyhow::Result<(TagIndex<Arc<String>>, DnfQuery<Arc<String>>)> {
    let extra = extra
        .iter()
        .map(|x| Arc::new((*x).to_owned()))
        .collect::<Vec<_>>();
    let mut rng = ChaChaRng::seed_from_u64(seed);
    let mut id = || -> String { hex::encode(rng.gen::<u64>().to_be_bytes()) };
    let mut create_concrete = |prefix: &str, n: usize| -> Vec<Arc<String>> {
        (0..n)
            .map(|_| format!("{}-{}", prefix, id()))
            .map(Arc::new)
            .collect()
    };
    let tags: Vec<(Arc<String>, Vec<Arc<String>>)> = recipe
        .iter()
        .map(|(prefix, n)| (Arc::new((*prefix).to_owned()), create_concrete(prefix, *n)))
        .collect();
    let events = (0..n_events)
        .filter_map(|_| {
            let (common, rare) = tags.choose(&mut rng).unwrap().clone();
            let rare = rare.choose(&mut rng)?.clone();
            let extra = extra.choose(&mut rng)?.clone();
            Some(
                vec![common, rare, extra]
                    .into_iter()
                    .collect::<TagSet<Arc<String>>>(),
            )
        })
        .collect::<Vec<_>>();
    let query = (0..n_terms)
        .filter_map(|_| {
            let (common, rare) = tags.choose(&mut rng).unwrap().clone();
            let rare = rare.choose(&mut rng)?.clone();
            Some(
                vec![common, rare]
                    .into_iter()
                    .collect::<TagSet<Arc<String>>>(),
            )
        })
        .collect::<Vec<_>>();
    let index = TagIndex::new(events)?;
    let query = DnfQuery::new(query)?;
    Ok((index, query))
}

#[test]
fn compression() -> anyhow::Result<()> {
    let (index, _query) = create_example(DENSE, EXTRA, 0, 10000, 3)?;
    let encoded = DagCborCodec.encode(&index)?;
    let compressed = zstd::encode_all(Cursor::new(&encoded), 10)?;
    println!("{} {}", encoded.len(), compressed.len());
    Ok(())
}
