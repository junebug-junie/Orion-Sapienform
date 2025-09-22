import qrcode
#qr = qrcode.make("^w")
qr = qrcode.make("^!PA")

qr.save("pair.png")
qr.show()