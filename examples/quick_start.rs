fn main() {
    let detection = betlang::detect("Write a short poem about the ocean.");
    let kind = detection.kind().expect("kind prediction");

    println!("{}", kind.slug());
}
