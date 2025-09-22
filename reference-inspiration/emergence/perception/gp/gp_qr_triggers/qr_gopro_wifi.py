import json
import qrcode

ssid = "CenturyLink5830"
password = "vj9wa9zm5bi5qy"
save_copy = True

config = {
    'ssid': ssid,
    'pass': password
}

#cmd = f"^c2{ssid};{password}^l"  # c2: join WiFi; ^w: turn WiFi on
#qr = qrcode.make(cmd)

payload = json.dumps(config)
img = qrcode.make(payload)

# Save or show it
img.save("qr_gopro_wifi.png")
img.show()
