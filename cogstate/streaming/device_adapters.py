from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Protocol

import numpy as np

from .buffer import StreamSample

logger = logging.getLogger(__name__)


class DeviceAdapter(Protocol):

    source: str
    sample_rate: float

    def start(self, on_sample: Callable[[StreamSample], None]) -> None:
        """Начать приём данных; on_sample вызывается на каждый отсчёт."""
        ...

    def stop(self) -> None:
        """Остановить приём и корректно закрыть соединение с устройством."""
        ...


class LSLEEGAdapter:
    """
    Адаптер для EEG-гарнитур, вещающих через Lab Streaming Layer

    """

    source = "eeg"

    def __init__(self, stream_name: str, sample_rate: float, n_channels: int):
        self.stream_name = stream_name
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, on_sample: Callable[[StreamSample], None]) -> None:
        from pylsl import StreamInlet, resolve_byprop

        logger.info("Поиск LSL-потока '%s'...", self.stream_name)
        streams = resolve_byprop("name", self.stream_name, timeout=5.0)
        if not streams:
            raise RuntimeError(f"LSL-поток '{self.stream_name}' не найден")

        inlet = StreamInlet(streams[0])
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                chunk, timestamps = inlet.pull_chunk(timeout=1.0)
                for values, ts in zip(chunk, timestamps):
                    sample = StreamSample(
                        source=self.source,
                        timestamp=ts,
                        values=np.asarray(values, dtype=np.float32),
                    )
                    on_sample(sample)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info("LSL EEG-адаптер запущен (%d каналов, %.1f Гц)",
                    self.n_channels, self.sample_rate)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class BLEHeartRateAdapter:
    """
    Адаптер для носимых устройств со стандартным Bluetooth LE
    Heart Rate профилем (GATT-нотификации)
    """

    source = "hr"
    sample_rate = 1.0

    HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

    def __init__(self, device_address: str):
        self.device_address = device_address
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

    def start(self, on_sample: Callable[[StreamSample], None]) -> None:
        import asyncio
        from bleak import BleakClient

        self._stop_event.clear()

        def _notification_handler(_, data: bytearray):
            hr_value = _parse_heart_rate(data)
            sample = StreamSample(
                source=self.source,
                timestamp=time.time(),
                values=np.array([hr_value], dtype=np.float32),
            )
            on_sample(sample)

        async def _run():
            async with BleakClient(self.device_address) as client:
                await client.start_notify(self.HR_MEASUREMENT_UUID, _notification_handler)
                logger.info("BLE HR-адаптер подключён к %s", self.device_address)
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.2)
                await client.stop_notify(self.HR_MEASUREMENT_UUID)

        def _thread_target():
            asyncio.run(_run())

        self._loop_thread = threading.Thread(target=_thread_target, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)


def _parse_heart_rate(data: bytearray) -> int:
    flags = data[0]
    is_16bit = flags & 0x1
    return int.from_bytes(data[1:3], byteorder="little") if is_16bit else data[1]
