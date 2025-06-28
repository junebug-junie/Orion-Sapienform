import json
import qrcode

url = "rtmp://192.168.02/live/stream"
resolution = "1080p"
save_copy = True
config = {
    'rtmpurl': url,
    'resolution': resolution,
    'save_copy': save_copy
}

#cmd = f"^c2{ssid};{password}^l"  # c2: join WiFi; ^w: turn WiFi on
#qr = qrcode.make(cmd)

payload = json.dumps(config)
img = qrcode.make(payload)

# Save or show it
img.save("qr_gopro_rtmp_address.png")
img.show()
