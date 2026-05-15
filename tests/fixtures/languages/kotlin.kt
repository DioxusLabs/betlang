package betlang.fixture

data class Greeter(val name: String) {
    fun message(): String = "hello $name"
}

fun main(args: Array<String>) {
    val names = if (args.isEmpty()) listOf("world") else args.toList()
    names
        .map { Greeter(it.trim()) }
        .filter { it.name.isNotEmpty() }
        .forEach { println(it.message()) }
}
