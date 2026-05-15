use super::embedded::MODEL_BYTES;

pub(crate) fn read_int4_dequant(cursor: &mut usize, count: usize, scale: f32) -> Box<[f32]> {
    let bytes = count.div_ceil(2);
    let payload = &MODEL_BYTES[*cursor..*cursor + bytes];
    *cursor += bytes;
    let mut out = Vec::with_capacity(count);
    for &packed in payload {
        if out.len() < count {
            let lo = (packed & 0x0f) as i8 - 8;
            out.push(lo as f32 * scale);
        }
        if out.len() < count {
            let hi = (packed >> 4) as i8 - 8;
            out.push(hi as f32 * scale);
        }
    }
    out.into_boxed_slice()
}

pub(crate) fn read_ternary_dequant(cursor: &mut usize, count: usize, scale: f32) -> Box<[f32]> {
    let bytes = count.div_ceil(4);
    let payload = &MODEL_BYTES[*cursor..*cursor + bytes];
    *cursor += bytes;
    let mut out = Vec::with_capacity(count);
    for &packed in payload {
        for shift in [0u32, 2, 4, 6] {
            if out.len() >= count {
                break;
            }
            let code = (packed >> shift) & 0x03;
            let v = match code {
                0 => -scale,
                2 => scale,
                _ => 0.0, // 1 → 0; codes outside {0,1,2} treated as 0 too
            };
            out.push(v);
        }
    }
    out.into_boxed_slice()
}

pub(crate) fn read_f32_array<const N: usize>(cursor: &mut usize) -> [f32; N] {
    let mut out = [0.0; N];
    let (chunks, remainder) = MODEL_BYTES[*cursor..*cursor + N * 4].as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    for (value, bytes) in out.iter_mut().zip(chunks) {
        *value = f32::from_le_bytes(*bytes);
    }
    *cursor += N * 4;
    out
}
