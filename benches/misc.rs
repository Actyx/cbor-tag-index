#![allow(clippy::type_complexity)]
use std::sync::Arc;

use cbor_tag_index::{DnfQuery, TagIndex};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libipld::codec::Codec;
use libipld_cbor::DagCborCodec;
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaChaRng;
use vec_collections::VecSet;

/// less than 128 distinct tags - dense index
const DENSE: &[(&str, usize)] = &[("article", 10), ("sku", 20), ("location", 20)];
/// more than 128 distinct tags - sparse index
const SPARSE: &[(&str, usize)] = &[("article", 10), ("sku", 20), ("location", 100)];
/// extra tags that are added randomly to all events
const EXTRA: &[&str] = &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];

type TagSet<T> = VecSet<[T; 4]>;

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

fn encode(index: &TagIndex<Arc<String>>) -> anyhow::Result<Vec<u8>> {
    DagCborCodec.encode(index)
}

fn decode(bytes: &[u8]) -> anyhow::Result<TagIndex<Arc<String>>> {
    DagCborCodec.decode(bytes)
}

fn dense(c: &mut Criterion) {
    let (index, query) = create_example(DENSE, EXTRA, 0, 10000, 3).unwrap();
    assert!(index.is_dense());
    let bytes = DagCborCodec.encode(&index).unwrap();
    c.bench_function("decode_dense", |b| b.iter(|| decode(black_box(&bytes))));
    c.bench_function("encode_dense", |b| b.iter(|| encode(black_box(&index))));
    c.bench_function("filter_dense", |b| {
        b.iter(|| query.matching(black_box(&index)))
    });
}

fn sparse(c: &mut Criterion) {
    let (index, query) = create_example(SPARSE, EXTRA, 0, 10000, 3).unwrap();
    assert!(!index.is_dense());
    let bytes = DagCborCodec.encode(&index).unwrap();
    c.bench_function("decode_sparse", |b| b.iter(|| decode(black_box(&bytes))));
    c.bench_function("encode_sparse", |b| b.iter(|| encode(black_box(&index))));
    c.bench_function("filter_sparse", |b| {
        b.iter(|| query.matching(black_box(&index)))
    });
}

criterion_group!(benches, dense, sparse);
criterion_main!(benches);
