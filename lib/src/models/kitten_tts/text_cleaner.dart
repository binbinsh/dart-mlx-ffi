library;

final class TextCleaner {
  TextCleaner() : _indexBySymbol = _buildIndex();

  static const _pad = r'$';
  static const _punctuation = ';:,.!?¡¿—…"«»"" ';
  static const _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
  static const _ipa =
      "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

  final Map<String, int> _indexBySymbol;

  List<int> call(String text) {
    final out = <int>[];
    for (final rune in text.runes) {
      final symbol = String.fromCharCode(rune);
      final index = _indexBySymbol[symbol];
      if (index != null) {
        out.add(index);
      }
    }
    return out;
  }

  static Map<String, int> _buildIndex() {
    final symbols = <String>[
      _pad,
      ..._punctuation.runes.map(String.fromCharCode),
      ..._letters.runes.map(String.fromCharCode),
      ..._ipa.runes.map(String.fromCharCode),
    ];
    return <String, int>{
      for (var i = 0; i < symbols.length; i++) symbols[i]: i,
    };
  }
}
