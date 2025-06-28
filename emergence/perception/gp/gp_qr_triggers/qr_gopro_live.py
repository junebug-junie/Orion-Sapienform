import json
import qrcode

cmd = f"^oA4!oW1mVr1080p30b4000!W!GL"
qr = qrcode.make(cmd)

# Save or show it
qr.save("qr_gopro_live.png")
qr.show()
