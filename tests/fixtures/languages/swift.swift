import Foundation

struct Greeter {
    let name: String

    func message() -> String {
        "hello \(name)"
    }
}

let names = CommandLine.arguments.dropFirst()
let greeters = (names.isEmpty ? ["world"] : Array(names)).map { Greeter(name: $0) }

for greeter in greeters.sorted(by: { $0.name < $1.name }) {
    print(greeter.message())
}
