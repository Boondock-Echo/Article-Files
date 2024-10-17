import asyncio
from bleak import BleakClient

# Replace with the Bluetooth address of your FNRISI FNB58
FNB58_BLUETOOTH_ADDRESS = "98:DA:B0:08:AA:88"  # Insert the actual device address here

async def list_services_and_characteristics():
    print(f"Connecting to {FNB58_BLUETOOTH_ADDRESS}...")
    
    async with BleakClient(FNB58_BLUETOOTH_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")
        
        services = await client.get_services()
        print("Listing services and characteristics...")
        
        for service in services:
            print(f"Service: {service.uuid}")
            for characteristic in service.characteristics:
                print(f"  Characteristic: {characteristic.uuid}, Properties: {characteristic.properties}")

loop = asyncio.get_event_loop()
loop.run_until_complete(list_services_and_characteristics())