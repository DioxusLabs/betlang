package main

import (
	"fmt"
	"sort"
)

type Greeter struct {
	Name string
}

func (g Greeter) Message() string {
	return fmt.Sprintf("hello %s", g.Name)
}

func main() {
	names := []string{"Ada", "Grace", "Linus"}
	sort.Strings(names)
	for _, name := range names {
		fmt.Println(Greeter{Name: name}.Message())
	}
}
