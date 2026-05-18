import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class Fixture {
    public static void main(String[] args) {
        List<String> names = new ArrayList<>();
        names.add("Ada");
        names.add("Grace");
        names.add("Linus");
        names.sort(Comparator.naturalOrder());
        for (String name : names) {
            System.out.println("hello " + name);
        }
    }
}
