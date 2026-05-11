import 'package:dart_inference/src/models/live_portrait/blazeface/anchors.dart';

void main() {
  final anchors = generateBlazeFaceAnchors();
  print('total: ${anchors.length}');
  print('first: cx=${anchors.first.cx} cy=${anchors.first.cy}');
  print('last: cx=${anchors.last.cx} cy=${anchors.last.cy}');
}
