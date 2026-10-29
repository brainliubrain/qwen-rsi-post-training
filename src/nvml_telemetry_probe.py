"""
High-Frequency NVML Hardware Telemetry Probe (1Hz).
Tracks GPU Power (Watts), Core Clocks (MHz), and Throttle Reasons.
"""
import time
from typing import Dict, Any

class HardwareTelemetryProbe:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._has_pynvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._has_pynvml = True
        except Exception:
            self._has_pynvml = False

    def sample(self) -> Dict[str, Any]:
        if not self._has_pynvml:
            return {"power_w": 0.0, "clock_mhz": 0, "throttle_reasons": "0x0"}
        import pynvml
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
            return {
                "power_w": power_mw / 1000.0,
                "clock_mhz": clock,
                "throttle_reasons": hex(throttle)
            }
        except Exception as e:
            return {"error": str(e)}
