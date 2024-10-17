import asyncio
from bleak import BleakClient

# Bluetooth address of your FNRISI FNB58 device
FNB58_BLUETOOTH_ADDRESS = "98:DA:B0:08:AA:88"

# UUIDs for the characteristics
NOTIFY_CHAR_UUID = "0000ffe4-0000-1000-8000-00805f9b34fb"
WRITE_CHAR_UUID = "0000ffe9-0000-1000-8000-00805f9b34fb"

def process_frame(data):
    """Process the incoming data frame and extract voltage and current if type 0x07 is found, handling little-endian byte order."""
    try:
        # Look for the pattern \xaa\x07 anywhere in the data
        for i in range(len(data) - 6):  # Ensure enough bytes are left for a full frame
            if data[i] == 0xaa and data[i + 1] == 0x07 and data[i + 2] == 0x04:
                # Extract voltage and current (little-endian)
                voltage_raw = (data[i + 4] << 8) | data[i + 3]
                current_raw = (data[i + 6] << 8) | data[i + 5]
                
                # Convert to actual values
                voltage = voltage_raw / 1000
                current = current_raw / 1000
                
                # Display voltage and current with three decimal places of accuracy
                print(f"{voltage:.3f}, {current:.3f}")
                return  # Stop after finding the first valid frame
    except IndexError:
        pass  # Handle incomplete frames gracefully

def notification_handler(sender, data):
    """Notification handler to receive data from the device."""
    process_frame(data)

async def run():
    print("Voltage (V), Current (A)")  # Data header
    print("-------------------------")

    print(f"Connecting to device {FNB58_BLUETOOTH_ADDRESS}...")
    async with BleakClient(FNB58_BLUETOOTH_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        # Start receiving notifications
        await client.start_notify(NOTIFY_CHAR_UUID, notification_handler)
        print("Notifications enabled...")

        # Step 1: Send initial command \xaa\x81\x00\xf4
        command_1 = bytearray([0xaa, 0x81, 0x00, 0xf4])
        await client.write_gatt_char(WRITE_CHAR_UUID, command_1)

        # Step 2: Wait for the device's response
        await asyncio.sleep(2)

        # Step 3: Send second command \xaa8200a7 to start data streaming
        command_2 = bytearray([0xaa, 0x82, 0x00, 0xa7])
        await client.write_gatt_char(WRITE_CHAR_UUID, command_2)

        # Keep connection open to receive streaming data
        await asyncio.sleep(30)

        # Stop notifications when done
        await client.stop_notify(NOTIFY_CHAR_UUID)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
