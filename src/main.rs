#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

mod tag_index;
use tag_index::*;

fn tag_set(tags: &[&'static str]) -> TagSet<&'static str> {
    let mut res = TagSet::default();
    for tag in tags {
        res.insert(*tag);
    }
    res
}

fn main() -> anyhow::Result<()> {
    let a = tag_set(&["a", "b", "x", "y"]);
    let b = tag_set(&["a", "x", "y"]);
    let c = tag_set(&["a", "b", "y"]);
    let mut builder = DnfQueryBuilder::new();
    builder.push(&a)?;
    builder.push(&b)?;
    builder.push(&c)?;
    let t = builder.dnf_query();
    println!("{:?}", t);
    for x in t.iter() {
        let x = x.map(|t| t.to_string()).collect::<Vec<_>>();
        println!("{:?}", x);
    }
    Ok(())
}
