import asyncio
from bleak import BleakScanner

async def scan_for_devices():
    print("Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover()

    for device in devices:
        print(f"Device: {device.name}, Address: {device.address}")

loop = asyncio.get_event_loop()
loop.run_until_complete(scan_for_devices())
