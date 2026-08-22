use std::collections::HashMap;

pub struct Greeter {
    greetings: HashMap<String, String>,
}

impl Greeter {
    pub fn new() -> Self {
        Self {
            greetings: HashMap::new(),
        }
    }

    pub fn greet(&mut self, name: &str) -> String {
        let message = format!("hello, {name}");
        self.greetings.insert(name.to_string(), message.clone());
        message
    }
}

fn main() {
    let mut greeter = Greeter::new();
    println!("{}", greeter.greet("world"));
}
