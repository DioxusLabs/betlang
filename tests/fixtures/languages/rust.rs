use std::collections::BTreeMap;

fn counts(values: &[&str]) -> BTreeMap<&str, usize> {
    let mut map = BTreeMap::new();
    for value in values {
        *map.entry(*value).or_default() += 1;
    }
    map
}

fn main() {
    for (language, count) in counts(&["rust", "rust", "python"]) {
        println!("{language}={count}");
    }
}
