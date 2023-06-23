import numpy as np
from aes import aes_128_encrypt, aes_128_decrypt


plain_text = np.arange(start=0, stop=16, dtype=np.uint8)
print('plain text:', plain_text)

key = np.ones(16, dtype=np.uint8)

cipher_text = aes_128_encrypt(plain_text, key)
print('cipher text:', cipher_text)

decrypted_plain_text = aes_128_decrypt(cipher_text, key)
print('decrypted plain text:', decrypted_plain_text)
