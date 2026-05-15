pub(crate) fn rfind_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

pub(crate) fn parse_scales(metadata: &str) -> Vec<f32> {
    let mut scales = Vec::new();
    let mut rest = metadata;
    while let Some(idx) = rest.find(r#""scale":"#) {
        let value_start = idx + r#""scale":"#.len();
        let value_end = rest[value_start..]
            .find([',', '}'])
            .map(|e| value_start + e)
            .expect("scale terminator");
        scales.push(
            rest[value_start..value_end]
                .parse::<f32>()
                .expect("scale parse"),
        );
        rest = &rest[value_end..];
    }
    scales
}

pub(crate) fn assert_tokenizer_version(metadata: &str) {
    let Some(version) = parse_usize_field(metadata, "tokenizer_version") else {
        panic!("missing wordseq tokenizer_version; runtime supports only v3")
    };
    assert!(
        version == 3,
        "unsupported wordseq tokenizer_version {version}; runtime supports only v3"
    );
}

fn parse_usize_field(metadata: &str, field: &str) -> Option<usize> {
    let key = format!(r#""{field}""#);
    let after_key = &metadata[metadata.find(&key)? + key.len()..];
    let rest = after_key[after_key.find(':')? + 1..].trim_start();
    let end = rest
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}
