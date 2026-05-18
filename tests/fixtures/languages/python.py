from pathlib import Path


def count_suffixes(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file():
            counts[path.suffix] = counts.get(path.suffix, 0) + 1
    return counts


def main() -> None:
    for suffix, count in sorted(count_suffixes(Path(".")).items()):
        print(f"{suffix or '<none>'}={count}")


if __name__ == "__main__":
    main()
