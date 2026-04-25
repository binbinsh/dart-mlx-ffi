from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import device_probe


class DeviceProbeTest(unittest.TestCase):
    def test_flutter_ios_devices_parses_machine_output(self) -> None:
        machine = [
            {
                "id": "00008140-001A25961A43801C",
                "name": "Binbin's iPhone (wireless)",
                "targetPlatform": "ios",
                "sdk": "iOS 26.4.1 23E254",
                "isSupported": True,
            },
            {
                "id": "macos",
                "name": "macOS",
                "targetPlatform": "darwin",
                "sdk": "macOS 26.4.1",
                "isSupported": True,
            },
        ]
        completed = mock.Mock(returncode=0, stdout=json.dumps(machine))
        with mock.patch("device_probe.shutil.which", return_value="/opt/homebrew/bin/flutter"):
            with mock.patch("device_probe.subprocess.run", return_value=completed):
                devices = device_probe.flutter_ios_devices("ios")
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["id"], "00008140-001A25961A43801C")
        self.assertEqual(devices[0]["via"], "flutter")


if __name__ == "__main__":
    unittest.main()
