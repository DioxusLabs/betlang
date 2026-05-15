object BetlangFixture {
  final case class Greeter(name: String) {
    def message: String = s"hello $name"
  }

  def counts(values: Seq[String]): Map[String, Int] =
    values.groupBy(identity).view.mapValues(_.size).toMap

  def main(args: Array[String]): Unit = {
    val names = if (args.isEmpty) Seq("world") else args.toSeq
    names.map(Greeter.apply).foreach(greeter => println(greeter.message))
    println(counts(names))
  }
}
