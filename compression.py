import array

class StandardPostings:
    """ 
    Class with static methods, to change the representation of the postings list
    which originally is a List of integers, into a sequence of bytes.
    We use the array library in Python.

    ASSUMPTION: the postings_list for a term FITS in memory!

    Please study:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray representing the sequence of integers in postings_list
        """
        # For the standard one, use L for unsigned long, because docID
        # will not be negative. And we assume the largest docID
        # can fit in a 4-byte unsigned representation.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list from a stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray representing the encoded postings list as output
            from the encode static method above.

        Returns
        -------
        List[int]
            list of docIDs that is the result of decoding encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies into a stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray representing the raw TF value of term occurrences in each
            document in the postings list
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies from a stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray representing the encoded term frequencies list as output
            from the encode_tf static method above.

        Returns
        -------
        List[int]
            List of term frequencies that is the result of decoding encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Unlike StandardPostings, where for a postings list,
    what is stored on disk is the original sequence of integers from the postings
    list as is.

    In VBEPostings, this time, what is stored is its gap, except
    the first posting. Only after that is it encoded with the Variable-Byte
    Encoding algorithm to a bytestream.

    Example:
    postings list [34, 67, 89, 454] will first be converted to gap-based,
    which is [34, 33, 22, 365]. Only then is it encoded with the
    Variable-Byte Encoding compression algorithm, and then converted to a bytestream.

    ASSUMPTION: the postings_list for a term FITS in memory!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        See our textbook!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend to the front
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # the first bit of the last byte is changed to 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Performs encoding (of course with compression) on a
        list of numbers, with Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a stream of bytes (with Variable-Byte
        Encoding). DO NOT FORGET to convert it first to a gap-based list, before
        encoding it and converting it to a bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray representing the sequence of integers in postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies into a stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray representing the raw TF value of term occurrences in each
            document in the postings list
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding a bytestream that was previously encoded with
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list from a stream of bytes. DO NOT FORGET
        the bytestream decoded from encoded_postings_list is still a
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray representing the encoded postings list as output
            from the encode static method above.

        Returns
        -------
        List[int]
            list of docIDs that is the result of decoding encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies from a stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray representing the encoded term frequencies list as output
            from the encode_tf static method above.

        Returns
        -------
        List[int]
            List of term frequencies that is the result of decoding encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

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
        Encodes a single positive integer n using Elias-Gamma coding.
        Returns a list of bits (0 or 1), MSB first.
        """
        if n == 0:
            raise ValueError("Elias-Gamma cannot encode 0")
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
        return n, pos

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


if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("encoded postings bytes: ", encoded_postings_list)
        print("encoded postings size   : ", len(encoded_postings_list), "bytes")
        print("encoded TF list bytes : ", encoded_tf_list)
        print("encoded postings size   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("decoding result (postings): ", decoded_posting_list)
        print("decoding result (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "decoding result is not the same as original postings"
        assert decoded_tf_list == tf_list, "decoding result is not the same as original postings"
        print()
