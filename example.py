import numpy as np
from aes import aes_128_encrypt, aes_128_decrypt


text = 'ABCDEFGHIJKLMNOP'
plain_text = np.array([ord(c) for c in text], dtype=np.uint8)
print('plain text:', plain_text)

key = np.full(16, ord('B'), dtype=np.uint8)

cipher_text = aes_128_encrypt(plain_text, key)
print('cipher text:', cipher_text)

decrypted_plain_text = aes_128_decrypt(cipher_text, key)
print('decrypted plain text:', decrypted_plain_text)
