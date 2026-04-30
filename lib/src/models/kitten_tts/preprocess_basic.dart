library;

List<String> basicEnglishTokenize(String text) {
  final matches = RegExp(r'\w+|[^\w\s]', unicode: true).allMatches(text);
  return matches.map((match) => match.group(0)!).toList();
}
