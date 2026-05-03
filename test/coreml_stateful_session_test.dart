// Phase 2 contract tests for the iOS17+/macOS14+ stateful Core ML
// pipeline. The real assertions need a stateful test mlpackage produced
// by Phase 1; until that artifact lands the suite documents the
// contract and is skipped end-to-end.
//
// See:
//   - mise-kitchen-prep/docs/adr/0001-paddleocr-vl-coreml-rearchitecture.md (§6)
//   - mise-kitchen-prep/docs/plans/paddleocr-vl-coreml-rebuild.md (Phase 2)

import 'package:test/test.dart';

void main() {
  group('CoreMlStatefulSession', () {
    test(
      'state persists across run() calls on the same stateful stage',
      () {
        // Contract:
        //   1. Open a session over a pipeline.json v2 with at least one
        //      stage flagged stateful.
        //   2. First run() into the stateful stage allocates an MLState
        //      via [model newState] (lazy).
        //   3. Two sequential run() calls observe shared state (e.g.
        //      counter goes 1 -> 2 with the dummy counter mlpackage).
        fail('unimplemented: requires Phase 1 stateful counter mlpackage');
      },
      skip: 'requires Phase 1 stateful test mlpackage',
    );

    test(
      'resetCoremlState() clears the state so the next run starts fresh',
      () {
        // Contract:
        //   1. Drive the stateful stage once (counter -> 1).
        //   2. Call dinf_coreml_reset_state.
        //   3. Next run() observes counter == 1 again (state was
        //      reallocated).
        fail('unimplemented: requires Phase 1 stateful counter mlpackage');
      },
      skip: 'requires Phase 1 stateful test mlpackage',
    );

    test(
      'reset on a session without stateful stages is a no-op (returns 0)',
      () {
        // Contract: pipelines that declare no stateful stages still
        // accept reset calls without error so callers do not need to
        // branch on the manifest.
        fail('unimplemented: requires loaded non-stateful test session');
      },
      skip: 'requires Phase 1 stateful test mlpackage',
    );
  });
}
