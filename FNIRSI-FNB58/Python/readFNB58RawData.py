import asyncio
from bleak import BleakClient

# Replace with your FNRISI FNB58's Bluetooth address
FNB58_BLUETOOTH_ADDRESS = '98:DA:B0:08:AA:88'

# UUIDs for the characteristics
NOTIFY_CHAR_UUID = '0000ffe4-0000-1000-8000-00805f9b34fb'
WRITE_CHAR_UUID = '0000ffe9-0000-1000-8000-00805f9b34fb'

def notification_handler(sender, data):
    """Notification handler to receive data from the device."""
    print(f"Received notification from {sender}: {data.hex()}")

async def run():
    print(f"Connecting to device {FNB58_BLUETOOTH_ADDRESS}...")
    async with BleakClient(FNB58_BLUETOOTH_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        # Start receiving notifications
        await client.start_notify(NOTIFY_CHAR_UUID, notification_handler)
        print("Notifications enabled...")

        # Step 1: Send initial command \xaa\x81\x00\xf4
        command_1 = bytearray([0xaa, 0x81, 0x00, 0xf4])
        await client.write_gatt_char(WRITE_CHAR_UUID, command_1)
        print(f"Sent command 1: {command_1.hex()}")

        # Step 2: Wait for the device's response (should start with \xaa030e3a00...)
        await asyncio.sleep(2)  # Adjust timing if needed to capture response

        # Step 3: Send second command \xaa8200a7 to start data streaming
        command_2 = bytearray([0xaa, 0x82, 0x00, 0xa7])
        await client.write_gatt_char(WRITE_CHAR_UUID, command_2)
        print(f"Sent command 2: {command_2.hex()}")

        # Keep connection open to receive streaming data
        await asyncio.sleep(30)  # Adjust timing as needed to collect streaming data

        # Stop notifications when done
        await client.stop_notify(NOTIFY_CHAR_UUID)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())