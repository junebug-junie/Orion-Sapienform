import qrcode
#qr = qrcode.make("^w")
qr = qrcode.make("^oW1")

qr.save("wifi_on_png")
qr.show()