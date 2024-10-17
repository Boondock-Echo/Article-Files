import asyncio
from bleak import BleakClient

# Replace with your FNRISI FNB58 Bluetooth address
FNB58_BLUETOOTH_ADDRESS = "98:DA:B0:08:AA:88"  # Insert actual device address here
# Replace with the characteristic UUID for notifications and write
NOTIFY_CHARACTERISTIC_UUID = "0000ffe4-0000-1000-8000-00805f9b34fb"
WRITE_CHARACTERISTIC_UUID = "0000ffe9-0000-1000-8000-00805f9b34fb"

# Data to send to start notifications (replace with appropriate command if known)
START_COMMAND = bytearray([0x01])  # Adjust the command as necessary

def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    print(f"Data received from {sender}: {data}")

async def run():
    print(f"Connecting to {FNB58_BLUETOOTH_ADDRESS}...")
    async with BleakClient(FNB58_BLUETOOTH_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")
        
        # Write to the characteristic to enable notifications (if needed)
        await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, START_COMMAND)
        print(f"Sent start command to {WRITE_CHARACTERISTIC_UUID}")
        
        # Start receiving notifications
        await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification_handler)
        print(f"Receiving notifications from {NOTIFY_CHARACTERISTIC_UUID}...")

        # Keep the connection alive to continue receiving data
        await asyncio.sleep(30)  # Adjust the time as needed or use a loop

        # Stop notifications when done
        await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
