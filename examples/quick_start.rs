fn main() {
    let detection = betlang::detect("fn main() { println!(\"hi\"); }\n");
    let language = detection.language().expect("language prediction");

    println!("{}", language.slug());
}
