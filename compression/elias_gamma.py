class EliasGammaPostings:
    """
    Bit-level compression using Elias-Gamma Encoding.

    Like VBEPostings, the postings list is first converted to a gap-based
    representation before encoding. The TF list is encoded as raw values
    (no gaps).

    Elias-Gamma encoding of a positive integer n:
      1. Let k = floor(log2(n))
      2. Write k zeros (unary prefix)
      3. Write a single 1 (separator)
      4. Write the binary representation of (n - 2^k) using k bits

    All bits are packed MSB-first into bytes, zero-padded to a full byte
    at the end. The first 4 bytes of every encoded block store the total
    number of integers encoded (as an unsigned 32-bit big-endian integer)
    so that the decoder knows when to stop.
    """

    @staticmethod
    def _encode_number(n):
        """
        Encodes a single non-negative integer n using Elias-Gamma coding.
        Since Elias-Gamma requires n > 0, we encode (n + 1).
        Returns a list of bits (0 or 1), MSB first.
        """
        n = n + 1
        k = n.bit_length() - 1          # floor(log2(n))
        bits = [0] * k                  # k leading zeros
        bits.append(1)                  # separator '1'
        if k > 0:
            # k-bit binary representation of (n - 2^k)
            remainder = n - (1 << k)
            for i in range(k - 1, -1, -1):
                bits.append((remainder >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_bytes(bits):
        """Pack a list of bits (MSB-first) into a bytes object. Zero-pads to byte boundary."""
        # Pad to multiple of 8
        pad = (8 - len(bits) % 8) % 8
        bits = bits + [0] * pad
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)

    @staticmethod
    def _bytes_to_bits(data):
        """Unpack bytes into a list of bits (MSB-first)."""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _decode_number(bits, pos):
        """
        Decode one Elias-Gamma number from bits starting at position pos.
        Returns (decoded_value, new_pos).
        """
        # Count leading zeros to find k
        k = 0
        while pos < len(bits) and bits[pos] == 0:
            k += 1
            pos += 1
        pos += 1  # skip the '1' separator
        # Read k more bits for the remainder
        remainder = 0
        for _ in range(k):
            remainder = (remainder << 1) | bits[pos]
            pos += 1
        n = (1 << k) + remainder
        return n - 1, pos

    @staticmethod
    def _encode_list(numbers):
        """Encode a list of positive integers with Elias-Gamma. Prepends a 4-byte count."""
        count = len(numbers)
        all_bits = []
        for n in numbers:
            all_bits.extend(EliasGammaPostings._encode_number(n))
        packed = EliasGammaPostings._bits_to_bytes(all_bits)
        # Prepend count as 4-byte big-endian integer
        return count.to_bytes(4, 'big') + packed

    @staticmethod
    def _decode_list(encoded_bytes):
        """Decode a list of Elias-Gamma encoded integers. Reads 4-byte count prefix."""
        count = int.from_bytes(encoded_bytes[:4], 'big')
        bits = EliasGammaPostings._bytes_to_bits(encoded_bytes[4:])
        numbers = []
        pos = 0
        for _ in range(count):
            n, pos = EliasGammaPostings._decode_number(bits, pos)
            numbers.append(n)
        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list using gap-based Elias-Gamma Encoding.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (must be positive integers, sorted ascending)

        Returns
        -------
        bytes
        """
        gap_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_list.append(postings_list[i] - postings_list[i - 1])
        return EliasGammaPostings._encode_list(gap_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from Elias-Gamma encoded bytes.

        Parameters
        ----------
        encoded_postings_list: bytes

        Returns
        -------
        List[int]
        """
        gap_list = EliasGammaPostings._decode_list(encoded_postings_list)
        postings = [gap_list[0]]
        for i in range(1, len(gap_list)):
            postings.append(postings[-1] + gap_list[i])
        return postings

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode TF list using Elias-Gamma Encoding (raw values, no gap).

        Parameters
        ----------
        tf_list: List[int]

        Returns
        -------
        bytes
        """
        return EliasGammaPostings._encode_list(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode TF list from Elias-Gamma encoded bytes.

        Parameters
        ----------
        encoded_tf_list: bytes

        Returns
        -------
        List[int]
        """
        return EliasGammaPostings._decode_list(encoded_tf_list)
