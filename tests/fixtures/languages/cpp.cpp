#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace betlang {
class Greeter {
public:
    explicit Greeter(std::string name) : name_(std::move(name)) {}

    void greet() const {
        std::cout << "hello " << name_ << '\n';
    }

private:
    std::string name_;
};
}

int main() {
    std::vector<std::string> names{"Ada", "Grace", "Linus"};
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
        betlang::Greeter{name}.greet();
    }
}
