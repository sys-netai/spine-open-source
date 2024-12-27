from re import U
import struct


def u32_from_bytes(bytes):
    # Full big endian: 
    tmp = struct.unpack("<I", bytes)[0]
    print(tmp)

    b1 = tmp & 0xff000000 # GH
    b2 = tmp & 0x00ff0000 # EF
    b3 = tmp & 0x0000ff00 # CD
    b4 = tmp & 0x000000ff # AB 
    return (b1 >> 8) + (b2 << 8) + (b3 >> 8) + (b4 << 8)


print(u32_from_bytes(b'\x85\xe8\x00\x00'))